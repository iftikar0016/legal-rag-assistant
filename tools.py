import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Support both local .env and Streamlit Cloud secrets
def get_secret(key: str, default: str = None) -> str:
    """Get secret from Streamlit secrets or environment variables."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)

# Load OpenAI credentials from environment variables
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_BASE_URL = get_secret("OPENAI_BASE_URL")

def retrieve_legal_context(query: str) -> str:
    """Retrieve relevant legal context from the indexed legal documents."""
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    db = FAISS.load_local("rag_faiss_store", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=3)
    return "\n\n".join([doc.page_content for doc in docs])

# Test the function
if __name__ == "__main__":
    query = "Are the pets allowed in the property?"
    context = retrieve_legal_context(query)
    print("Retrieved Context:\n", context)