from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pymongo import MongoClient
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# -----------------------------
# Azure OpenAI LLM
# -----------------------------
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0
)

# -----------------------------
# Fetch data from MongoDB
# -----------------------------
def fetch_forecast_summary() -> str:
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client["smart_retail"]
    records = list(db["forecasts"].find({}, {"_id": 0}))
    client.close()

    if not records:
        return "No forecast data available."

    import pandas as pd
    df = pd.DataFrame(records)

    # Pre-compute stats so LLM doesn't do raw math
    avg_error      = df["error"].mean()
    total_records  = len(df)
    best_store     = df.groupby("store")["predicted_sales"].mean().idxmax()
    best_sales     = df.groupby("store")["predicted_sales"].mean().max()
    worst_error_store = df.groupby("store")["error"].mean().idxmax()
    worst_error_val   = df.groupby("store")["error"].mean().max()

    # Per-store summary (not raw rows)
    store_summary = df.groupby("store").agg(
        avg_predicted=("predicted_sales", "mean"),
        avg_actual=("actual_sales", "mean"),
        avg_error=("error", "mean"),
        total_weeks=("week", "count")
    ).reset_index()

    store_lines = "\n".join([
        f"Store {int(r['store'])} | Avg predicted: {r['avg_predicted']:.0f} "
        f"| Avg actual: {r['avg_actual']:.0f} | Avg error: {r['avg_error']:.0f} "
        f"| Weeks: {int(r['total_weeks'])}"
        for _, r in store_summary.iterrows()
    ])

    return f"""
SUMMARY STATS:
- Total records: {total_records}
- Average prediction error: {avg_error:.0f}
- Best performing store: Store {best_store} (avg predicted sales: {best_sales:.0f})
- Highest error store: Store {worst_error_store} (avg error: {worst_error_val:.0f})

PER-STORE BREAKDOWN:
{store_lines}
"""


def fetch_anomaly_summary() -> str:
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client["smart_retail"]

    # Only fetch anomalies (not normal records) to keep context focused
    records = list(db["anomalies"].find(
        {"anomaly": "Anomaly"},
        {"_id": 0}
    ).limit(100))
    client.close()

    if not records:
        return "No anomalies detected."

    df_text = "\n".join([
        f"Store {r['store']} | Date {r['date']} | Sales {r['weekly_sales']:.0f} "
        f"| Score {r['anomaly_score']:.4f} | Temp {r['temperature']:.1f} "
        f"| CPI {r['cpi']:.2f} | Unemployment {r['unemployment']:.2f}"
        for r in records
    ])
    return df_text


# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a smart retail data analyst assistant.
You have access to two datasets:

1. FORECAST DATA (predicted vs actual weekly sales per store):
{forecast_data}

2. ANOMALY DATA (unusual sales patterns detected, with anomaly scores):
{anomaly_data}

Answer the user's question clearly and concisely using this data.
Give specific numbers, store names, and dates where relevant.
If the question is not answerable from the data, say so honestly.

Question: {question}

Answer:
""")

# -----------------------------
# Chain
# -----------------------------
chain = prompt | llm | StrOutputParser()


# -----------------------------
# Public function for orchestrator
# -----------------------------
def ask_analyst_agent(question: str) -> str:
    forecast_data = fetch_forecast_summary()
    anomaly_data  = fetch_anomaly_summary()

    return chain.invoke({
        "forecast_data": forecast_data,
        "anomaly_data":  anomaly_data,
        "question":      question
    })


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    test_questions = [
        "Which store had the highest predicted sales?",
        "How many anomalies were detected and which store had the worst one?",
        "What is the average prediction error across all stores?"
    ]
    for q in test_questions:
        print(f"\n❓ {q}")
        print(f"🤖 {ask_analyst_agent(q)}")



# pandas code to verify the question:- wch store has the highest predicted sales?

# import pandas as pd
# from pymongo import MongoClient
# import os
# from dotenv import load_dotenv
# load_dotenv()

# client = MongoClient(os.getenv("MONGO_URI"))
# df = pd.DataFrame(list(client["smart_retail"]["forecasts"].find({}, {"_id": 0})))

# print(df.groupby("store")["predicted_sales"].mean().sort_values(ascending=False).head(5))        