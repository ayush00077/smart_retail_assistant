from langchain.tools import tool
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.customer_qa_agent import ask_customer_agent
from agents.analyst_agent import ask_analyst_agent
from agents.document_agent import ask_document_agent

# -----------------------------
# Each agent wrapped as MCP Tool
# -----------------------------
@tool
def customer_qa_tool(query: str) -> str:
    """
    Use this tool for general retail customer questions.
    Examples: product availability, store hours, order tracking,
    loyalty programs, general shopping queries.
    """
    return ask_customer_agent(query)


@tool
def data_analyst_tool(query: str) -> str:
    """
    Use this tool for data-driven retail analytics questions.
    Examples: sales trends, demand forecasting results, anomaly detection,
    store performance comparisons, prediction accuracy.
    """
    return ask_analyst_agent(query)


@tool
def document_search_tool(query: str) -> str:
    """
    Use this tool to search official store policy and product documents.
    Examples: return policy, Walmart+ benefits, payment methods,
    warranty information, pharmacy hours, price match policy.
    """
    return ask_document_agent(query)


# Export all tools as a list
agent_tools = [
    customer_qa_tool,
    data_analyst_tool,
    document_search_tool
]