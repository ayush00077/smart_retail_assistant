

from dotenv import load_dotenv
import os
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

# -----------------------------
# Azure OpenAI Chat Model
# -----------------------------
llm = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),  # ← 'azure_deployment' not 'deployment_name'
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

test = embedding_model.embed_query("test")
print("✅ Embedding works! Vector length:", len(test))
# Should print 1536 for text-embedding-3-small

# -----------------------------
# Load & Split PDF
# -----------------------------
loader = PyPDFLoader("/Users/ayush/Desktop/Sprint-project/backend/knowledge_base/retail_knowledge_base.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Paste this right before vector_store = Chroma.from_documents(...)
print("Endpoint:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("Embedding deployment:", os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))
print("API version:", os.getenv("AZURE_OPENAI_API_VERSION"))

# Test embedding directly
test = embedding_model.embed_query("test")
print("Embedding works! Vector length:", len(test))

# -----------------------------
# Create Vector Store
# -----------------------------
vector_store = Chroma.from_documents(
    docs,
    embedding_model,
    persist_directory="../../chroma_db"
)

retriever = vector_store.as_retriever()

# -----------------------------
# LCEL RAG Chain (replaces RetrievalQA)
# -----------------------------
prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If you don't know, say "I don't have enough information."

Context: {context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------
# Ask Question
# -----------------------------
query = "What is demand forecasting in retail?"
response = rag_chain.invoke(query)

print("\nANSWER:\n")
print(response)