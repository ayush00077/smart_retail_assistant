from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

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
# Azure Embedding Model
# -----------------------------
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    model="text-embedding-3-small"
)

# -----------------------------
# Build or load Chroma DB
# — separate collection from customer_qa_agent
# -----------------------------
# CHROMA_PATH = "../../chroma_db_docs"   # different path — keeps collections separate
# PDF_PATH    = "/Users/ayush/Desktop/Sprint-project/backend/knowledge_base/retail_store_documents.pdf"


BASE_DIR  = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_PATH  = os.path.join(BASE_DIR, "knowledge_base", "retail_store_documents.pdf")
CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db_docs")

def build_or_load_vector_store():
    # If already built, just load it
    if os.path.exists(CHROMA_PATH):
        return Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embedding_model
        )

    # First run — load PDF, chunk, embed, save
    print("Building document vector store...")
    loader   = PyPDFLoader(PDF_PATH)
    docs_raw = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(docs_raw)

    vector_store = Chroma.from_documents(
        docs,
        embedding_model,
        persist_directory=CHROMA_PATH
    )
    print(f"✅ Document vector store built with {len(docs)} chunks")
    return vector_store

vector_store = build_or_load_vector_store()
retriever    = vector_store.as_retriever(search_kwargs={"k": 4})

# -----------------------------
# Prompt
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
You are a Walmart store assistant with access to the official store policy
and product knowledge base documents.

Answer the question using only the context provided below.
Be specific — include prices, timeframes, and policy details where available.
If the answer is not in the documents, say "I couldn't find that in our store documents."

Context: {context}

Question: {question}

Answer:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -----------------------------
# RAG Chain
# -----------------------------
doc_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------
# Public function for orchestrator
# -----------------------------
def ask_document_agent(question: str) -> str:
    return doc_chain.invoke(question)


# -----------------------------
# Quick test
# -----------------------------
if __name__ == "__main__":
    test_questions = [
        "What is the return policy for electronics?",
        "How much does Walmart+ membership cost and what are its benefits?",
        "What payment methods does Walmart accept?",
        "How is demand forecasting done at Walmart?"
    ]
    for q in test_questions:
        print(f"\n❓ {q}")
        print(f"🤖 {ask_document_agent(q)}")