from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

# -----------------------------
# Load LLM
# -----------------------------
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
    temperature=0
)

# -----------------------------
# Load Embedding Model
# -----------------------------
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    model="text-embedding-3-small"
)

# -----------------------------
# Load existing Chroma DB (no rebuild)
# -----------------------------
# vector_store = Chroma(
#     persist_directory="../../chroma_db",   # same path as rag_service.py
#     embedding_function=embedding_model
# )


BASE_DIR    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)


retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# -----------------------------
# Retail-specific prompt
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a helpful Smart Retail Assistant.
Answer the customer's question using only the context provided.
If you don't know the answer, say "I'm sorry, I don't have that information right now."
Keep your answer friendly, clear, and concise.

Context: {context}

Customer Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -----------------------------
# RAG Chain
# -----------------------------
customer_qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------
# Public function for orchestrator
# -----------------------------
def ask_customer_agent(question: str) -> str:
    return customer_qa_chain.invoke(question)


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    test_questions = [
        "What is demand forecasting in retail?",
        "How do I track my order?",
        "What is your return policy?"
    ]
    for q in test_questions:
        print(f"\n❓ {q}")
        print(f"🤖 {ask_customer_agent(q)}")