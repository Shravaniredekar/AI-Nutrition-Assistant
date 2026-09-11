# 🥦 NutriBot – AI Nutrition Assistant

**AICTE EduNet Project Submission | Problem Statement No. 8: Nutrition Agent**

> An intelligent, multi-agent nutrition assistant powered by **IBM watsonx.ai** (Granite 3 8B Instruct) that provides personalised dietary guidance, health advisories, and real-time food tracking.

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Multi-Agent Architecture](#-multi-agent-architecture)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Configuration](#-configuration)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [Features](#-features)
- [Screenshots](#-screenshots)
- [Disclaimer](#-disclaimer)

---

## 🌟 Project Overview

NutriBot is a full-stack AI-powered nutrition assistant that leverages IBM watsonx.ai's Granite 3 8B Instruct large language model alongside a Retrieval-Augmented Generation (RAG) pipeline. It deploys four specialised agents working in concert to deliver:

- Factual, evidence-based nutrition information
- Personalised 7-day meal plans
- Condition-specific health and dietary advisories
- Real-time meal tracking with nutritional analysis

---

## 🤖 Multi-Agent Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NutriBot Orchestrator                   │
│                    (Flask API Gateway)                      │
└──────────────┬───────────────────────────────┬─────────────┘
               │                               │
    ┌──────────▼──────────┐       ┌────────────▼────────────┐
    │  Agent 1            │       │  Agent 2                │
    │  Nutrition Knowledge│       │  Diet Recommendation    │
    │  Agent (RAG)        │       │  Agent                  │
    │                     │       │                         │
    │  • FAISS vector DB  │       │  • 7-day meal plans     │
    │  • Semantic search  │       │  • Caloric targets      │
    │  • Fact retrieval   │       │  • Macro splits         │
    │  • LangChain RAG    │       │  • Dietary preferences  │
    └─────────────────────┘       └─────────────────────────┘

    ┌─────────────────────┐       ┌─────────────────────────┐
    │  Agent 3            │       │  Agent 4                │
    │  Health Advisory    │       │  Food Log Agent         │
    │  Agent              │       │                         │
    │  • Condition-based  │       │  • Meal tracking        │
    │  • Clinical guidance│       │  • Instant analysis     │
    │  • Drug-nutrient    │       │  • Daily summaries      │
    │    interactions     │       │  • Gap identification   │
    └─────────────────────┘       └─────────────────────────┘
                           │
              ┌────────────▼────────────────────┐
              │   IBM watsonx.ai                │
              │   Model: ibm/granite-3-8b-instruct │
              │   Endpoint: us-south.ml.cloud.ibm.com │
              └─────────────────────────────────┘
```

### Agent Details

| # | Agent | Trigger | Key Capabilities |
|---|-------|---------|-----------------|
| 1 | **Nutrition Knowledge Agent** | `agent=nutrition_knowledge` | RAG over curated knowledge base, semantic retrieval via FAISS, evidence-based Q&A |
| 2 | **Diet Recommendation Agent** | `agent=diet_recommendation` | Personalised meal plans, caloric targeting, macro-nutrient splits, goal-based planning |
| 3 | **Health Advisory Agent** | `agent=health_advisory` | Condition-specific nutrition, drug-nutrient interactions, clinical dietary guidance |
| 4 | **Food Log Agent** | `agent=food_log` | Real-time meal logging, per-item nutritional estimates, daily intake analysis |

---

## 🛠 Technology Stack

| Layer | Technology |
|-------|-----------|
| **LLM** | IBM watsonx.ai – `ibm/granite-3-8b-instruct` |
| **Backend** | Python 3.10+, Flask |
| **RAG Pipeline** | LangChain, FAISS (CPU), FakeEmbeddings |
| **IBM SDK** | `ibm-watsonx-ai` |
| **Frontend** | HTML5, CSS3 (vanilla), JavaScript (ES2020) |
| **Styling** | Custom purple theme, Font Awesome 6, Google Fonts (Inter) |
| **Config** | `python-dotenv` |

---

## 📁 Project Structure

```
AI-Nutrition-Assistant/
├── app.py                  # Flask application + 4 AI agents
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (do NOT commit)
├── .env.example            # Template for environment variables
├── README.md               # This file
├── templates/
│   └── index.html          # NutriBot frontend SPA
└── static/
    └── style.css           # Purple NutriBot theme
```

---

## ⚙️ Setup & Installation

### Prerequisites

- Python 3.10 or higher
- An active [IBM Cloud](https://cloud.ibm.com) account
- An [IBM watsonx.ai](https://www.ibm.com/watsonx) project with the Granite model enabled
- `pip` (Python package manager)

### 1. Clone / Download the Project

```bash
git clone <repository-url>
cd AI-Nutrition-Assistant
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Configuration

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```env
IBM_CLOUD_API_KEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
FLASK_SECRET_KEY=your_flask_secret_key_here
PORT=5000
HOST=0.0.0.0
```

### Obtaining IBM Credentials

| Variable | Where to find it |
|----------|-----------------|
| `IBM_CLOUD_API_KEY` | IBM Cloud → Manage → Access (IAM) → API Keys → Create |
| `WATSONX_PROJECT_ID` | watsonx.ai → Projects → your project → Manage → General → Project ID |
| `WATSONX_URL` | Use `https://us-south.ml.cloud.ibm.com` for Dallas region (default) |

---

## 🚀 Running the Application

```bash
python app.py
```

Open your browser at: **[http://localhost:5000](http://localhost:5000)**

For production deployment:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## 📡 API Reference

### Base URL: `http://localhost:5000`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Serve the NutriBot UI |
| `GET`  | `/api/status` | Health check & model info |
| `POST` | `/api/chat` | Route to any agent via chat |
| `POST` | `/api/meal-plan` | Generate personalised meal plan |
| `POST` | `/api/health-advice` | Get condition-specific advice |
| `POST` | `/api/food-log` | Add / view / clear food log |
| `GET`  | `/api/family-profile` | List family profiles |
| `POST` | `/api/family-profile` | Add a family profile |
| `DELETE` | `/api/family-profile?id=<id>` | Remove a family profile |
| `GET`  | `/api/nutrition-facts?food=<name>` | Quick nutrition lookup |

### POST `/api/chat` — Request Body Examples

**Nutrition Knowledge Agent:**
```json
{
  "message": "What are the best sources of plant-based protein?",
  "agent": "nutrition_knowledge"
}
```

**Diet Recommendation Agent:**
```json
{
  "message": "I want to lose weight",
  "agent": "diet_recommendation",
  "user_data": {
    "name": "Shravani",
    "age": 25,
    "weight": 65,
    "height": 162,
    "goal": "lose weight",
    "activity_level": "moderate",
    "dietary_preference": "vegetarian",
    "allergies": "none"
  }
}
```

**Health Advisory Agent:**
```json
{
  "message": "PCOS management",
  "agent": "health_advisory",
  "health_data": {
    "condition": "PCOS",
    "symptoms": "irregular periods, weight gain",
    "current_diet": "no restrictions",
    "medications": "metformin"
  }
}
```

**Food Log Agent:**
```json
{
  "agent": "food_log",
  "action": "add",
  "log_entry": {
    "food": "Grilled salmon",
    "quantity": "200g",
    "meal_type": "dinner"
  }
}
```

---

## ✨ Features

### 🤖 Chat Interface
- Natural language conversation with 4 selectable specialised agents
- Quick-prompt suggestions per agent for rapid interaction
- Message history with role-differentiated bubbles
- Auto-resizing input textarea

### 🍽️ Meal Plan Visualiser
- Full user profile form (age, weight, height, activity, goal, preferences)
- IBM Granite-generated 7-day meal plans with caloric targets
- Interactive macronutrient bar chart (carbs / protein / fat)

### 📋 Food Log
- Log meals with food name, quantity, and meal type
- Per-entry nutritional estimates via Granite
- Daily summary with gap analysis and recommendations
- One-click log clear

### 👨‍👩‍👧 Family Profile Manager
- Add unlimited family member profiles
- Per-member personalised health advisory generation
- Profile management (add / delete / view)

---

## 🖥️ Screenshots

> Run the application and visit `http://localhost:5000` to view the full UI.

---

## ⚠️ Disclaimer

NutriBot is an **educational project** developed for the AICTE EduNet internship programme. All nutritional information and health advisories generated by this system are for **informational purposes only** and do not constitute professional medical, dietary, or clinical advice. Always consult a qualified healthcare professional or registered dietitian before making significant changes to your diet or health regimen.

---

## 📜 License

MIT License — free for educational and non-commercial use.

---

*Built with ❤️ for AICTE EduNet | Problem Statement No. 8: Nutrition Agent*
