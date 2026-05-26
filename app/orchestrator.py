# from langchain_openai import AzureChatOpenAI
# from langchain_core.messages import HumanMessage, SystemMessage
# from dotenv import load_dotenv
# import os

# load_dotenv()

# # -----------------------------
# # Import all agents
# # -----------------------------
# from agents.customer_qa_agent import ask_customer_agent
# from agents.analyst_agent import ask_analyst_agent
# from agents.document_agent import ask_document_agent

# # -----------------------------
# # Router LLM (fast, just classifies)
# # -----------------------------
# router_llm = AzureChatOpenAI(
#     azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#     api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#     api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
#     azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
#     temperature=0
# )

# # -----------------------------
# # Intent classifier
# # -----------------------------
# def classify_intent(query: str) -> str:
#     response = router_llm.invoke([
#         SystemMessage(content="""
# You are a query router for a Smart Retail AI platform.
# Classify the user's query into exactly ONE of these categories:

# - CUSTOMER_QA     : General retail questions, product availability, store hours, orders
# - DATA_ANALYST    : Sales trends, demand forecasting, anomalies, store performance, predictions
# - DOCUMENT_SEARCH : Store policies, return policy, Walmart+, payment methods, warranties, pharmacy

# Reply with ONLY the category label — no explanation, no punctuation.
# """),
#         HumanMessage(content=query)
#     ])
#     return response.content.strip()

# # -----------------------------
# # Main orchestrator function
# # -----------------------------
# def route_query(query: str) -> dict:
#     intent = classify_intent(query)

#     print(f"   🔀 Routed to: {intent}")

#     if intent == "CUSTOMER_QA":
#         answer = ask_customer_agent(query)

#     elif intent == "DATA_ANALYST":
#         answer = ask_analyst_agent(query)

#     elif intent == "DOCUMENT_SEARCH":
#         answer = ask_document_agent(query)

#     else:
#         # Fallback — try document agent
#         intent = "DOCUMENT_SEARCH (fallback)"
#         answer = ask_document_agent(query)

#     return {
#         "query":  query,
#         "intent": intent,
#         "answer": answer
#     }


# # -----------------------------
# # Quick test
# # -----------------------------
# if __name__ == "__main__":
#     test_queries = [
#         "What is demand forecasting in retail?",           # → CUSTOMER_QA
#         "Which store had the highest predicted sales?",    # → DATA_ANALYST
#         "What is the return policy for electronics?",      # → DOCUMENT_SEARCH
#         "How many anomalies were detected last season?",   # → DATA_ANALYST
#         "How much does Walmart+ membership cost?",         # → DOCUMENT_SEARCH
#     ]

#     print("\n" + "="*60)
#     print("  SMART RETAIL MULTI-AGENT ORCHESTRATOR — TEST")
#     print("="*60)

#     for q in test_queries:
#         print(f"\n❓ {q}")
#         result = route_query(q)
#         print(f"🤖 [{result['intent']}]\n{result['answer']}")
#         print("-"*60)


# This is the MCP style tool calling:-


from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from services.text_analytics_service import analyze_query
from dotenv import load_dotenv
import os, json

load_dotenv()

from agents.customer_qa_agent import ask_customer_agent
from agents.analyst_agent import ask_analyst_agent
from agents.document_agent import ask_document_agent

# -----------------------------
# LLM
# -----------------------------
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0
)

# -----------------------------
# MCP Tools defined as schemas
# -----------------------------
MCP_TOOLS = [
    {
        "name": "customer_qa_tool",
        "description": "For general retail customer questions: product availability, store hours, order tracking, loyalty programs.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The customer question"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "data_analyst_tool",
        "description": "For data analytics questions: sales trends, demand forecasting, anomaly detection, store performance.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The analytics question"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "document_search_tool",
        "description": "For store policy and product document questions: return policy, Walmart+ benefits, payment methods, warranties.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The document search question"}
            },
            "required": ["query"]
        }
    }
]

# -----------------------------
# Tool executor
# -----------------------------
def execute_tool(tool_name: str, query: str) -> str:
    if tool_name == "customer_qa_tool":
        return ask_customer_agent(query)
    elif tool_name == "data_analyst_tool":
        return ask_analyst_agent(query)
    elif tool_name == "document_search_tool":
        return ask_document_agent(query)
    return "Unknown tool"

# -----------------------------
# MCP Orchestrator
# — LLM selects tool via function calling
# -----------------------------


def route_query(query: str) -> dict:
    # Step 1: Enrich query with Text Analytics
    analytics = analyze_query(query)

    print(f"   📊 Sentiment: {analytics['sentiment']['sentiment']}")
    print(f"   🔑 Key phrases: {analytics['key_phrases']}")

    # Step 2: MCP tool selection
    llm_with_tools = llm.bind_tools(MCP_TOOLS)
    selection = llm_with_tools.invoke(query)

    if selection.tool_calls:
        tool_call  = selection.tool_calls[0]
        tool_name  = tool_call["name"]
        tool_query = tool_call["args"].get("query", query)

        print(f"   🔧 MCP Tool selected: {tool_name}")

        # Step 3: Priority handling for complaints
        if analytics["is_complaint"]:
            tool_query = f"[PRIORITY COMPLAINT] {tool_query}"

        answer = execute_tool(tool_name, tool_query)
        intent = tool_name.upper().replace("_TOOL", "")
    else:
        answer = selection.content
        intent = "DIRECT"

    return {
        "query":      query,
        "intent":     intent,
        "answer":     answer,
        "analytics":  analytics    # ← included in every response
    }

# -----------------------------
# Test
# -----------------------------
if __name__ == "__main__":
    queries = [
        "Which store had the highest predicted sales?",
        "What is the return policy for electronics?",
        "How many anomalies were detected?",
        "How much does Walmart+ cost?",
    ]

    print("\n" + "="*60)
    print("  SMART RETAIL — MCP MULTI-AGENT ORCHESTRATOR")
    print("="*60)

    for q in queries:
        print(f"\n❓ {q}")
        result = route_query(q)
        print(f"🤖 [{result['intent']}]\n{result['answer']}")
        print("-"*60)