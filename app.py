import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from rag_index_builder import build_index_from_pdf
from tools import retrieve_legal_context
from openai import OpenAI

# Load environment variables (for local development)
load_dotenv()

# Support both local .env and Streamlit Cloud secrets
def get_secret(key: str, default: str = None) -> str:
    """Get secret from Streamlit secrets or environment variables."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.getenv(key, default)

OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
OPENAI_BASE_URL = get_secret("OPENAI_BASE_URL")

# Initialize OpenAI client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

# Core chat function
def run_agent(query):
    """Run the legal assistant to answer queries using RAG."""
    
    # Step 1: Retrieve relevant context from the vector store
    context = retrieve_legal_context(query)
    
    # Step 2: Create a prompt with the context
    prompt = f"""You are a helpful legal assistant. Use the following context from legal documents to answer the user's question.

Context:
{context}

Question: {query}

Please provide a clear and accurate answer based on the context above. If the context doesn't contain relevant information, say so."""

    # Step 3: Get response from OpenAI
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "You are a helpful legal assistant that provides accurate answers based on legal documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_completion_tokens=1000
        )
        
        answer = response.choices[0].message.content
        return answer, [{"role": "assistant", "content": answer}]
    except Exception as e:
        return f"Error: {str(e)}", []

# Streamlit app layout
st.set_page_config(page_title="📄 Legal RAG Assistant", layout="wide")
st.markdown("""
    <div style="
        background: linear-gradient(135deg, #e0f7fa, #ffdde1);
        padding: 25px 35px;
        border-radius: 15px;
        max-width: 850px;
        margin: 0 auto 30px auto;
        text-align: center;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    ">
        <h1 style="
            color: #1a237e;
            font-size: 2.8em;
            margin-bottom: 12px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        ">
            ⚖️ Legal RAG Assistant
        </h1>
        <p style="
            color: #37474f;
            font-size: 1.15em;
            font-weight: 500;
            line-height: 1.6;
        ">
            Ask your legal questions with confidence 💼<br/>
            Powered by Retrieval-Augmented Generation (RAG) 🧠🔍
        </p>
    </div>
""", unsafe_allow_html=True)



st.markdown("Upload your **legal documents** and ask your legal questions. Our AI assistant will respond with contextual answers extracted directly from your document. 🔍📘")

# File upload
uploaded_file = st.file_uploader("📎 Upload a legal PDF document", type=["pdf"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_pdf_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")

    with st.spinner("🔧 Building FAISS index from uploaded PDF..."):
        build_index_from_pdf(tmp_pdf_path, persist_dir="rag_faiss_store")
    st.success("📚 Index built and ready to go!")

    # Query section
    st.markdown("---")
    st.subheader("💬 Ask Your Legal Question")
    query = st.text_input("Type your legal query here:")

    if query:
        with st.spinner("🤖 Retrieving answer from Legal Assistant..."):
            answer, chat_history = run_agent(query)

        st.markdown("### 🧠 Agent's Answer:")
        st.success(answer)

        # Optional: Expandable chat history
        with st.expander("📜 View Full Chat History"):
            for msg in chat_history:
                content = msg.get("content", "") if isinstance(msg, dict) else ""
                if content:
                    st.markdown(f"**Assistant**: {content}")
else:
    st.info("📂 Please upload a legal PDF to begin.")