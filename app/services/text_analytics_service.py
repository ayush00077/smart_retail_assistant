from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from dotenv import load_dotenv
import os

load_dotenv()

ENDPOINT = os.getenv("AZURE_LANGUAGE_ENDPOINT")
KEY      = os.getenv("AZURE_LANGUAGE_KEY")

# -----------------------------
# Initialize client
# -----------------------------
def get_client():
    return TextAnalyticsClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY)
    )

# -----------------------------
# 1. Sentiment Analysis
# -----------------------------
def analyze_sentiment(text: str) -> dict:
    client = get_client()
    result = client.analyze_sentiment([text])[0]

    return {
        "sentiment":  result.sentiment,        # positive / negative / neutral / mixed
        "scores": {
            "positive": round(result.confidence_scores.positive, 3),
            "neutral":  round(result.confidence_scores.neutral,  3),
            "negative": round(result.confidence_scores.negative, 3)
        }
    }

# -----------------------------
# 2. Key Phrase Extraction
# -----------------------------
def extract_key_phrases(text: str) -> list:
    client = get_client()
    result = client.extract_key_phrases([text])[0]
    return list(result.key_phrases)

# -----------------------------
# 3. Language Detection
# -----------------------------
def detect_language(text: str) -> dict:
    client = get_client()
    result = client.detect_language([text])[0]
    return {
        "language": result.primary_language.name,
        "iso_code": result.primary_language.iso6391_name,
        "confidence": round(result.primary_language.confidence_score, 3)
    }

# -----------------------------
# Combined — runs all 3 at once
# Used by orchestrator before routing
# -----------------------------
def analyze_query(text: str) -> dict:
    sentiment   = analyze_sentiment(text)
    key_phrases = extract_key_phrases(text)
    language    = detect_language(text)

    # Flag negative queries for priority handling
    is_complaint = sentiment["sentiment"] == "negative" and \
                   sentiment["scores"]["negative"] > 0.7

    return {
        "sentiment":    sentiment,
        "key_phrases":  key_phrases,
        "language":     language,
        "is_complaint": is_complaint
    }


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    test_queries = [
        "I am very unhappy with my order, it was damaged and late!",
        "What is the return policy for electronics?",
        "Which store had the highest sales this month?"
    ]
    for q in test_queries:
        print(f"\n📝 Query: {q}")
        result = analyze_query(q)
        print(f"   Sentiment:   {result['sentiment']['sentiment']} {result['sentiment']['scores']}")
        print(f"   Key phrases: {result['key_phrases']}")
        print(f"   Language:    {result['language']['language']}")
        print(f"   Complaint:   {result['is_complaint']}")