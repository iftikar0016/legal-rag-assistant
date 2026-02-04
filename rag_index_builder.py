try:
    import fitz
except ImportError:
    fitz = None

import os

# Load environment variables (optional for local dev with .env)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# Support both local .env and Streamlit Cloud secrets
def get_secret(key: str, default: str = None) -> str:
    """Get secret from Streamlit secrets or environment variables."""
    if HAS_STREAMLIT:
        try:
            return st.secrets[key]
        except (KeyError, FileNotFoundError):
            pass
    return os.getenv(key, default)

# Load OpenAI credentials from environment variables
OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_BASE_URL = get_secret("OPENAI_BASE_URL")

def extract_text_from_pdf(pdf_path:str):
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is not installed. Install it locally to use PDF upload feature.")
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def build_index_from_pdf(pdf_path: str, persist_dir: str = "./rag_faiss_store"):
    full_text = extract_text_from_pdf(pdf_path)
    

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    documents = text_splitter.split_documents([Document(page_content=full_text)])

    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )
    db = FAISS.from_documents(documents, embeddings)
    os.makedirs(persist_dir, exist_ok=True)
    db.save_local(persist_dir)




if __name__ == "__main__":
    build_index_from_pdf("./docs/sample_rental_agreement.pdf")