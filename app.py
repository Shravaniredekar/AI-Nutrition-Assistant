"""
NutriBot - AI Nutrition Assistant
AICTE EduNet Project Submission | Problem Statement No. 8: Nutrition Agent
Multi-Agent System powered by IBM watsonx.ai (ibm/granite-3-8b-instruct)
"""

import os
import json
import logging
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template, session

# LangChain imports
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FakeEmbeddings

# IBM watsonx.ai
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "nutribot-secret-2024")

# ---------------------------------------------------------------------------
# IBM watsonx.ai client
# ---------------------------------------------------------------------------
IBM_API_KEY   = os.getenv("IBM_CLOUD_API_KEY", "")
PROJECT_ID    = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL   = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
MODEL_ID      = "meta-llama/llama-3-1-8b"

_wx_model: ModelInference | None = None

def get_watsonx_model() -> ModelInference:
    """Lazy-initialise the watsonx ModelInference client."""
    global _wx_model
    if _wx_model is None:
        credentials = Credentials(url=WATSONX_URL, api_key=IBM_API_KEY)
        client = APIClient(credentials=credentials, project_id=PROJECT_ID)
        _wx_model = ModelInference(
            model_id=MODEL_ID,
            api_client=client,
            params={
                GenParams.MAX_NEW_TOKENS: 1024,
                GenParams.TEMPERATURE: 0.7,
                GenParams.TOP_P: 0.9,
                GenParams.REPETITION_PENALTY: 1.1,
            },
        )
        logger.info("watsonx ModelInference client initialised (%s)", MODEL_ID)
    return _wx_model


def generate_response(prompt: str) -> str:
    """Send a prompt to Granite, print token metrics, and return generated text."""
    try:
        model = get_watsonx_model()
        
        # Call generate() instead of generate_text() to get full payload metadata
        response = model.generate(prompt=prompt)
        
        # Extract response text and token counts safely
        results = response.get("results", [{}])[0]
        generated_text = results.get("generated_text", "").strip()
        
        input_tokens = results.get("input_token_count", 0)
        output_tokens = results.get("generated_token_count", 0)
        total_tokens = input_tokens + output_tokens

        # Print token metrics directly in your terminal console
        print("\n==================================================")
        print("           WATSONX TOKEN USAGE METRICS            ")
        print("==================================================")
        print(f" Prompt Input Tokens : {input_tokens}")
        print(f" Model Output Tokens : {output_tokens}")
        print(f" Total Tokens Used   : {total_tokens}")
        print("==================================================\n")

        return generated_text if generated_text else "I'm unable to generate a response right now."
        
    except Exception as exc:
        logger.error("watsonx generation error: %s", exc)
        return f"I encountered an error while processing your request. Please check your API credentials. ({exc})"


# ---------------------------------------------------------------------------
# RAG Knowledge Base (Nutrition Knowledge Agent)
# ---------------------------------------------------------------------------
NUTRITION_KNOWLEDGE = """
Macronutrients: Carbohydrates provide 4 kcal/g and are the body's primary energy source.
Proteins provide 4 kcal/g and are essential for muscle repair and immune function.
Fats provide 9 kcal/g and are vital for hormone production and fat-soluble vitamin absorption.

Micronutrients: Vitamin C (ascorbic acid) boosts immunity and is found in citrus fruits.
Vitamin D supports calcium absorption; sources include sunlight and fortified dairy.
Iron is critical for haemoglobin synthesis; found in red meat, spinach, and legumes.
Calcium supports bone density; sources include dairy, broccoli, and almonds.
Omega-3 fatty acids reduce inflammation; found in salmon, walnuts, and flaxseed.

Dietary Guidelines: Adults need 0.8 g protein per kg of body weight daily.
The recommended daily fibre intake is 25–38 g for adults.
Added sugar should not exceed 10% of total daily caloric intake.
Sodium intake should be kept below 2,300 mg per day.
Hydration: 8–10 cups (2–2.5 L) of water per day is generally recommended.

Superfoods: Berries are rich in antioxidants. Leafy greens contain folate and iron.
Quinoa is a complete protein containing all essential amino acids.
Turmeric contains curcumin with potent anti-inflammatory properties.
Greek yoghurt is high in protein and beneficial probiotics.

Weight Management: A caloric deficit of 500 kcal/day leads to approximately 0.5 kg of fat loss per week.
Mindful eating and portion control are evidence-based strategies for weight management.
Regular physical activity (150 min/week moderate intensity) complements dietary efforts.

Special Diets: Vegan diets require careful planning for B12, iron, calcium, and zinc.
Keto diets are high in fat, moderate in protein, and very low in carbohydrates (<50 g/day).
The Mediterranean diet is associated with reduced cardiovascular disease risk.
Gluten-free diets are medically necessary for those with coeliac disease.

Children & Adolescents: Calcium and Vitamin D are especially important during growth.
Iron needs increase during adolescence, particularly for females.
Breakfast consumption is linked to improved concentration and academic performance.

Elderly Nutrition: Protein needs increase with age to prevent sarcopenia.
B12 absorption decreases with age; supplementation may be required.
Adequate hydration is crucial as thirst sensation diminishes with age.
"""


def build_vector_store():
    """Build an in-memory FAISS vector store from nutrition knowledge."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.create_documents([NUTRITION_KNOWLEDGE])
    embeddings = FakeEmbeddings(size=512)
    return FAISS.from_documents(docs, embeddings)


_vector_store = build_vector_store()


def retrieve_context(query: str, k: int = 3) -> str:
    """Retrieve the top-k relevant chunks from the vector store."""
    try:
        results = _vector_store.similarity_search(query, k=k)
        return "\n".join(doc.page_content for doc in results)
    except Exception as exc:
        logger.warning("Vector search failed: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# In-memory stores (replace with a DB for production)
# ---------------------------------------------------------------------------
food_logs: dict[str, list] = {}          # session_id -> list of log entries
family_profiles: dict[str, list] = {}    # session_id -> list of profiles


# ===========================================================================
# AGENT 1 – Nutrition Knowledge Agent (RAG-based)
# ===========================================================================
def nutrition_knowledge_agent(query: str) -> str:
    """Answer factual nutrition questions using RAG over the knowledge base."""
    context = retrieve_context(query)
    prompt = f"""You are NutriBot, an expert nutrition knowledge assistant.
Use the following verified nutrition knowledge to answer the user's question accurately and concisely.

Knowledge Base Context:
{context}

User Question: {query}

Provide a clear, evidence-based answer. If the context does not cover the question, say so honestly.

Answer:"""
    return generate_response(prompt)


# ===========================================================================
# AGENT 2 – Diet Recommendation Agent
# ===========================================================================
def diet_recommendation_agent(user_data: dict) -> str:
    """Generate a personalised meal plan and diet recommendations."""
    name   = user_data.get("name", "User")
    age    = user_data.get("age", "unknown")
    weight = user_data.get("weight", "unknown")
    height = user_data.get("height", "unknown")
    goal   = user_data.get("goal", "maintain weight")
    dietary_pref = user_data.get("dietary_preference", "no restrictions")
    allergies    = user_data.get("allergies", "none")
    activity     = user_data.get("activity_level", "moderate")

    prompt = f"""You are NutriBot's Diet Recommendation Agent — a certified nutritionist AI.

Create a comprehensive, personalised 7-day meal plan for:
- Name: {name}
- Age: {age} years
- Weight: {weight} kg | Height: {height} cm
- Goal: {goal}
- Activity Level: {activity}
- Dietary Preference: {dietary_pref}
- Allergies / Restrictions: {allergies}

Include:
1. Daily caloric target and macronutrient split (carbs / protein / fat %)
2. A structured 7-day meal plan (Breakfast, Lunch, Dinner, Snacks)
3. Hydration recommendation
4. 3–5 practical tips aligned with the user's goal
5. Foods to emphasise and foods to avoid

Format the plan clearly with headings and bullet points.

Meal Plan:"""
    return generate_response(prompt)


# ===========================================================================
# AGENT 3 – Health Advisory Agent
# ===========================================================================
def health_advisory_agent(health_data: dict) -> str:
    """Provide health and nutritional advice for specific conditions."""
    condition = health_data.get("condition", "general health")
    symptoms  = health_data.get("symptoms", "none")
    current_diet = health_data.get("current_diet", "not specified")
    medications  = health_data.get("medications", "none")

    prompt = f"""You are NutriBot's Health Advisory Agent — a clinical nutrition specialist AI.

Provide evidence-based nutritional guidance for:
- Health Condition: {condition}
- Current Symptoms: {symptoms}
- Current Diet: {current_diet}
- Current Medications: {medications}

Include:
1. Nutritional impact of this condition
2. Key nutrients to prioritise (with food sources)
3. Foods and substances to avoid
4. Meal timing and portion recommendations
5. Lifestyle modifications that complement nutrition
6. When to consult a healthcare professional

⚠️ IMPORTANT DISCLAIMER: This is informational guidance only and does not replace professional medical advice. Always consult a qualified healthcare provider for medical decisions.

Health Advisory:"""
    return generate_response(prompt)


# ===========================================================================
# AGENT 4 – Food Log Agent
# ===========================================================================
def food_log_agent(action: str, session_id: str, log_entry: dict | None = None) -> dict:
    """Manage the user's food log and provide nutritional analysis."""
    if session_id not in food_logs:
        food_logs[session_id] = []

    if action == "add" and log_entry:
        log_entry["timestamp"] = datetime.now().isoformat()
        food_logs[session_id].append(log_entry)
        # Ask Granite to estimate nutrition for the logged item
        food_name = log_entry.get("food", "the food item")
        quantity  = log_entry.get("quantity", "1 serving")
        prompt = f"""You are NutriBot's Food Log Agent. Estimate the nutritional content for:
Food: {food_name}
Quantity: {quantity}

Provide:
- Estimated calories
- Protein (g)
- Carbohydrates (g)
- Fat (g)
- Key micronutrients
- Health rating (1–10) with a brief justification

Keep the response concise.

Nutritional Estimate:"""
        analysis = generate_response(prompt)
        return {
            "status": "logged",
            "entry": log_entry,
            "analysis": analysis,
            "total_entries": len(food_logs[session_id]),
        }

    elif action == "view":
        logs = food_logs.get(session_id, [])
        if not logs:
            return {"status": "empty", "message": "No food entries logged yet.", "logs": []}
        # Daily summary via Granite
        log_summary = "\n".join(
            f"- {e.get('food','?')} ({e.get('quantity','?')}) at {e.get('timestamp','?')}"
            for e in logs[-10:]  # last 10 entries
        )
        prompt = f"""You are NutriBot's Food Log Agent. Analyse the following recent food log:

{log_summary}

Provide:
1. Estimated daily calorie intake
2. Macronutrient balance assessment
3. Nutritional gaps or excesses identified
4. 3 actionable improvement suggestions

Daily Log Analysis:"""
        analysis = generate_response(prompt)
        return {"status": "success", "logs": logs, "analysis": analysis}

    elif action == "clear":
        food_logs[session_id] = []
        return {"status": "cleared", "message": "Food log cleared successfully."}

    return {"status": "error", "message": "Unknown action."}


# ===========================================================================
# Flask Routes
# ===========================================================================

@app.route("/")
def index():
    """Serve the NutriBot frontend."""
    if "session_id" not in session:
        session["session_id"] = f"session_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """Main chat endpoint — routes messages to the appropriate agent."""
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    agent   = data.get("agent", "nutrition_knowledge")

    if not message:
        return jsonify({"error": "Message is required."}), 400

    session_id = session.get("session_id", "default")
    response_text = ""

    try:
        if agent == "nutrition_knowledge":
            response_text = nutrition_knowledge_agent(message)

        elif agent == "diet_recommendation":
            user_data = data.get("user_data", {})
            user_data.setdefault("goal", message)
            response_text = diet_recommendation_agent(user_data)

        elif agent == "health_advisory":
            health_data = data.get("health_data", {})
            health_data.setdefault("condition", message)
            response_text = health_advisory_agent(health_data)

        elif agent == "food_log":
            action    = data.get("action", "view")
            log_entry = data.get("log_entry")
            result    = food_log_agent(action, session_id, log_entry)
            return jsonify({"success": True, "agent": agent, "result": result})

        else:
            # Fallback — treat as a general nutrition question
            response_text = nutrition_knowledge_agent(message)

        return jsonify({
            "success": True,
            "agent": agent,
            "response": response_text,
            "timestamp": datetime.now().isoformat(),
        })

    except Exception as exc:
        logger.error("Chat route error: %s", exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/meal-plan", methods=["POST"])
def meal_plan():
    """Generate a personalised meal plan."""
    user_data = request.get_json(silent=True) or {}
    if not user_data:
        return jsonify({"error": "User data is required."}), 400
    result = diet_recommendation_agent(user_data)
    return jsonify({"success": True, "meal_plan": result, "timestamp": datetime.now().isoformat()})


@app.route("/api/health-advice", methods=["POST"])
def health_advice():
    """Provide health-condition-specific nutritional advice."""
    health_data = request.get_json(silent=True) or {}
    if not health_data:
        return jsonify({"error": "Health data is required."}), 400
    result = health_advisory_agent(health_data)
    return jsonify({"success": True, "advice": result, "timestamp": datetime.now().isoformat()})


@app.route("/api/food-log", methods=["POST"])
def food_log_route():
    """Add, view, or clear the food log."""
    data      = request.get_json(silent=True) or {}
    action    = data.get("action", "view")
    log_entry = data.get("log_entry")
    session_id = session.get("session_id", "default")
    result    = food_log_agent(action, session_id, log_entry)
    return jsonify({"success": True, "result": result})


@app.route("/api/family-profile", methods=["GET", "POST", "DELETE"])
def family_profile():
    """Manage family member nutrition profiles."""
    session_id = session.get("session_id", "default")
    if session_id not in family_profiles:
        family_profiles[session_id] = []

    if request.method == "GET":
        return jsonify({"success": True, "profiles": family_profiles[session_id]})

    if request.method == "POST":
        profile = request.get_json(silent=True) or {}
        if not profile.get("name"):
            return jsonify({"error": "Profile name is required."}), 400
        profile["id"] = f"profile_{len(family_profiles[session_id]) + 1}"
        profile["created_at"] = datetime.now().isoformat()
        family_profiles[session_id].append(profile)
        return jsonify({"success": True, "profile": profile, "total": len(family_profiles[session_id])})

    if request.method == "DELETE":
        profile_id = request.args.get("id")
        before = len(family_profiles[session_id])
        family_profiles[session_id] = [
            p for p in family_profiles[session_id] if p.get("id") != profile_id
        ]
        removed = before - len(family_profiles[session_id])
        return jsonify({"success": True, "removed": removed, "total": len(family_profiles[session_id])})


@app.route("/api/nutrition-facts", methods=["GET"])
def nutrition_facts():
    """Quick nutrition facts lookup via the knowledge agent."""
    food = request.args.get("food", "").strip()
    if not food:
        return jsonify({"error": "Food parameter is required."}), 400
    query  = f"What are the nutritional facts and health benefits of {food}?"
    result = nutrition_knowledge_agent(query)
    return jsonify({"success": True, "food": food, "facts": result})


@app.route("/api/status", methods=["GET"])
def status():
    """Health-check endpoint."""
    return jsonify({
        "status": "online",
        "service": "NutriBot AI Nutrition Assistant",
        "model": MODEL_ID,
        "version": "1.0.0",
        "agents": [
            "Nutrition Knowledge Agent",
            "Diet Recommendation Agent",
            "Health Advisory Agent",
            "Food Log Agent",
        ],
        "timestamp": datetime.now().isoformat(),
    })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    logger.info("Starting NutriBot on %s:%s", host, port)
    app.run(host=host, port=port, debug=debug)
