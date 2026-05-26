# Smart Retail Assistant

A production-grade, multi-agent AI backend for retail analytics and customer support — deployed on **Azure Web Apps** with CI/CD via GitHub Actions.

---

##  Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Agents](#agents)
- [Services](#services)
- [API Endpoints](#api-endpoints)
- [Environment Variables](#environment-variables)
- [Local Setup](#local-setup)
- [Docker](#docker)
- [Deployment](#deployment)
- [Data Pipeline](#data-pipeline)
- [ML Models](#ml-models)

---

## Overview

The Smart Retail Assistant is an intelligent backend platform that combines **LLM-powered multi-agent orchestration**, **RAG (Retrieval-Augmented Generation)**, **ML-based demand forecasting**, and **anomaly detection** — all tailored for retail operations.

Users can ask natural language questions like:
- *"Which store had the highest predicted sales?"*
- *"What is the return policy for electronics?"*
- *"How many anomalies were detected last season?"*

The system intelligently routes each query to the most appropriate AI agent using **MCP (Model Context Protocol)** tool-calling.

---

## Architecture

```
User Query
    │
    ▼
Azure AI Text Analytics
(Sentiment + Key Phrases + Language Detection)
    │
    ▼
MCP Orchestrator  ──── AzureChatOpenAI (tool selection via function calling)
    │
    ├──► Customer QA Agent     → ChromaDB (RAG on retail knowledge base)
    ├──► Data Analyst Agent    → MongoDB (forecast + anomaly data)
    └──► Document Search Agent → ChromaDB (RAG on store policy PDFs)
    │
    ▼
FastAPI Response
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| **API Framework** | FastAPI + Uvicorn (ASGI) |
| **LLM** | Azure OpenAI (GPT-4 via `AzureChatOpenAI`) |
| **Embeddings** | Azure OpenAI `text-embedding-3-small` |
| **Agent Framework** | LangChain  |
| **Orchestration** | MCP-style tool-calling via `llm.bind_tools()` |
| **Vector Store** | ChromaDB (two separate collections) |
| **RAG** | LangChain LCEL RAG chains + PyPDFLoader |
| **Database** | MongoDB Atlas (via Motor async driver + PyMongo) |
| **ML Models** | scikit-learn (RandomForestRegressor, IsolationForest) |
| **Azure ML** | Azure Machine Learning real-time endpoint |
| **NLP Analytics** | Azure AI Language (Text Analytics) |
| 
| **Deployment** | Azure Web Apps (App Service) |
| **CI/CD** | GitHub Actions  |
| **Config** | python-dotenv, Pydantic Settings |
| **Testing** | pytest |

---

## Project Structure

```
Sprint-project/
├── backend/
│   ├── app/
│   │   ├── main.py                  # FastAPI app, router registration
│   │   ├── config.py                # Env var loading (MONGO_URI, DB name)
│   │   ├── database.py              # Motor async MongoDB client
│   │   ├── orchestrator.py          # MCP orchestrator — routes queries to agents
│   │   ├── agents/
│   │   │   ├── customer_qa_agent.py # RAG agent for general retail Q&A
│   │   │   ├── analyst_agent.py     # Data analyst agent (MongoDB → LLM)
│   │   │   └── document_agent.py    # RAG agent for store policy PDFs
│   │   ├── services/
│   │   │   ├── rag_service.py           # PDF ingestion + ChromaDB builder
│   │   │   ├── forecasting.py           # RandomForest training + MongoDB save
│   │   │   ├── anomaly.py               # IsolationForest training + MongoDB save
│   │   │   ├── azure_ml_service.py      # Azure ML real-time endpoint client
│   │   │   └── text_analytics_service.py # Azure AI Language client
│   │   ├── routes/
│   │   │   ├── agent_routes.py      # POST /api/agent/query
│   │   │   ├── ml_routes.py         # POST /api/ml/predict
│   │   │   ├── analytics_routes.py  # POST /api/analytics/analyze
│   │   │   └── upload.py            # POST /upload-data (CSV ingestion)
│   │   ├── mcp/
│   │   │   └── agent_tools.py       # MCP tool schemas
│   │   └── models/
│   │       ├── forecast.pkl         # Trained RandomForest model
│   │       └── anomaly.pkl          # Trained IsolationForest model
│   ├── knowledge_base/
│   │   ├── retail_knowledge_base.pdf
│   │   └── retail_store_documents.pdf
│   ├── chroma_db/                   # Vector store for customer Q&A
│   ├── chroma_db_docs/              # Vector store for store policy docs
│   ├── tests/
│   │   └── test_main.py
│   ├
│   ├── requirements.txt
│   └── .github/
│       └── workflows/
│           └── main_smartretailassistant.yml
└── data/
    └── raw/
        └── Walmart.csv              # Historical weekly sales dataset
```

---

## Agents

### 1. Customer QA Agent (`customer_qa_agent.py`)
Answers general retail questions using RAG over the `retail_knowledge_base.pdf`.
- Retrieves top-3 relevant chunks from ChromaDB (`chroma_db/`)
- Builds answer using `AzureChatOpenAI` with a retail-specific prompt
- Triggered when the query is about product availability, store hours, order tracking, or loyalty programs

### 2. Data Analyst Agent (`analyst_agent.py`)
Answers data-driven questions by querying MongoDB and feeding structured summaries to the LLM.
- Fetches forecast records (store-wise avg predicted/actual sales, error)
- Fetches anomaly records (flagged weeks, anomaly scores, contextual features)
- Generates precise numeric answers — LLM never does raw math
- Triggered for sales trends, forecasting, anomaly, or performance questions

### 3. Document Search Agent (`document_agent.py`)
Answers policy and compliance questions using RAG over `retail_store_documents.pdf`.
- Retrieves top-4 relevant chunks from ChromaDB (`chroma_db_docs/`)
- Strict grounding: answers only from the document context
- Triggered for questions on return policy, warranties, payment methods, Walmart+ benefits

---

## Services

### MCP Orchestrator (`orchestrator.py`)
The central routing brain. Works in two steps:
1. **Pre-routing enrichment** — runs Azure AI Text Analytics on every incoming query (sentiment, key phrases, language, complaint flag)
2. **MCP tool selection** — calls `llm.bind_tools(MCP_TOOLS)` so the LLM picks the right agent via function calling (not a hard-coded classifier)
3. **Priority escalation** — prepends `[PRIORITY COMPLAINT]` to the query if negative sentiment > 0.7

### Azure AI Text Analytics (`text_analytics_service.py`)
Wraps Azure AI Language with three capabilities:
- Sentiment analysis (positive / negative / neutral / mixed with confidence scores)
- Key phrase extraction
- Language detection 

### Azure ML Service (`azure_ml_service.py`)
Calls the deployed Azure Machine Learning real-time endpoint for sales forecasting. Accepts store features (store ID, date, holiday flag, temperature, fuel price, CPI, unemployment) and returns `predicted_sales`.

### Forecasting Service (`forecasting.py`)
Offline training script:
- Trains a `RandomForestRegressor` on Walmart weekly sales data
- Features: store, holiday_flag, temperature, fuel_price, CPI, unemployment, year, month, week
- Saves model as `forecast.pkl` and persists predictions to `smart_retail.forecasts` in MongoDB

### Anomaly Detection Service (`anomaly.py`)
Offline training script:
- Trains `IsolationForest` on weekly sales features
- Labels records as `Normal` or `Anomaly` and computes anomaly scores
- Saves model as `anomaly.pkl` and persists results to `smart_retail.anomalies` in MongoDB

### RAG Service (`rag_service.py`)
One-time ingestion script:
- Loads PDFs via `PyPDFLoader`, chunks via `RecursiveCharacterTextSplitter` (500 tokens, 50 overlap)
- Embeds chunks using `AzureOpenAIEmbeddings` (`text-embedding-3-small`)
- Persists to ChromaDB for reuse by agents

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check — returns API name |
| `POST` | `/upload-data` | Upload a CSV file; preprocesses and inserts into MongoDB |
| `POST` | `/api/agent/query` | Main NLP endpoint — routes query through MCP orchestrator |
| `GET` | `/api/agent/health` | Returns status of all three agents and orchestrator |
| `POST` | `/api/ml/predict` | Call Azure ML endpoint for a single sales forecast |
| `POST` | `/api/analytics/analyze` | Run Azure AI Text Analytics on arbitrary text |

### Example — Agent Query

```bash
curl -X POST https://<your-app>.azurewebsites.net/api/agent/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Which store had the highest predicted sales?"}'
```

```json
{
  "query": "Which store had the highest predicted sales?",
  "intent": "DATA_ANALYST_TOOL",
  "answer": "Store 20 had the highest average predicted weekly sales at $2,104,322."
}
```

### Example — Sales Forecast

```bash
curl -X POST https://<your-app>.azurewebsites.net/api/ml/predict \
  -H "Content-Type: application/json" \
  -d '{
    "store": 1,
    "date": "2024-06-01",
    "holiday_flag": 0,
    "temperature": 72.5,
    "fuel_price": 3.45,
    "cpi": 211.0,
    "unemployment": 7.8,
    "day": 1, "month": 6, "year": 2024
  }'
```

---

## Environment Variables

Create a `.env` file in `backend/`:

```env
# MongoDB
MONGO_URI=mongodb+srv://<user>:<password>@cluster.mongodb.net/
DATABASE_NAME=smart_retail

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_OPENAI_API_VERSION=2024-02-01
AZURE_OPENAI_CHAT_DEPLOYMENT=<gpt-4-deployment-name>
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=<embedding-deployment-name>

# Azure AI Language
AZURE_LANGUAGE_ENDPOINT=https://<resource>.cognitiveservices.azure.com/
AZURE_LANGUAGE_KEY=<key>

# Azure ML
AZURE_ML_ENDPOINT=https://<endpoint>.inference.ml.azure.com/score
AZURE_ML_API_KEY=<key>
```

---

## Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/Sprint-project.git
cd Sprint-project/backend

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Fill in all values in .env

# 5. Run the server
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### One-time setup steps (run once before starting)

```bash
# Build ChromaDB vector stores from PDFs
python app/services/rag_service.py

# Train ML models and persist results to MongoDB
python app/services/forecasting.py
python app/services/anomaly.py
```

---


## Deployment

The app is deployed to **Azure App Service (Web Apps)** under the name `SmartRetailAssistant`.

### CI/CD Pipeline (`.github/workflows/main_smartretailassistant.yml`)

> **Note:** This workflow file was **automatically generated by Azure** when the GitHub repo was connected to Azure Web Apps via the Azure Portal (Deployment Center). It was not written manually — Azure committed it directly into the repository and set up all the required secrets.

Triggers on every push to `main`:

1. **Build job** — sets up Python 3.13, creates a virtual environment, installs dependencies, and uploads the app as a GitHub artifact
2. **Deploy job** — logs in to Azure via OIDC (passwordless, using federated credentials) and deploys to the `Production` slot using `azure/webapps-deploy@v3`

Azure's **Oryx build engine** handles `pip install` on the server side during deployment (`SCM_DO_BUILD_DURING_DEPLOYMENT=true`).

GitHub secrets (auto-configured by Azure Portal):
- `AZUREAPPSERVICE_CLIENTID_*`
- `AZUREAPPSERVICE_TENANTID_*`
- `AZUREAPPSERVICE_SUBSCRIPTIONID_*`

---

## Data Pipeline

The project uses the **Walmart Weekly Sales dataset** (`data/raw/Walmart.csv`) with the following flow:

```
Walmart.csv (raw)
    │
    ▼
/upload-data endpoint
    │  • Lowercases column names
    │  • Drops duplicates and NaN rows
    │  • Parses date column (dayfirst=True)
    ▼
MongoDB → smart_retail.sales_data
    │
    ▼
forecasting.py → smart_retail.forecasts
anomaly.py     → smart_retail.anomalies
```

---

## ML Models

### Sales Forecasting — `RandomForestRegressor`
| Parameter | Value |
|---|---|
| `n_estimators` | 100 |
| `random_state` | 42 |
| `test_size` | 20% |
| Features | store, holiday_flag, temperature, fuel_price, CPI, unemployment, year, month, week |
| Target | `weekly_sales` |
| Metrics | MAE, RMSE, R² |

### Anomaly Detection — `IsolationForest`
| Parameter | Value |
|---|---|
| `n_estimators` | 100 |
| `contamination` | 0.02 (2%) |
| `random_state` | 42 |
| Features | weekly_sales, temperature, fuel_price, CPI, unemployment |
| Output | `Normal` / `Anomaly` label + anomaly score |

Both models are serialized as `.pkl` files in `app/models/` and also served live via the **Azure ML real-time endpoint** for on-demand inference.

---

## Running Tests

```bash
cd backend
pytest tests/
```

---

## License

This project was built as part of a sprint. All Azure service credentials are managed through environment variables and GitHub Secrets — no keys are committed to the repository.
