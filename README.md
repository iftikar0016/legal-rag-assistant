# ContractLens — Legal RAG Assistant ⚖️

A Retrieval‑Augmented Generation (RAG) app that turns legal PDFs into an interactive question‑answering experience. Upload a contract, build a fast vector index, and get grounded answers backed by relevant document context.

---

## ✨ Highlights
- **Practical RAG pipeline**: PDF → chunking → embeddings → FAISS → answer synthesis.
- **Fast local retrieval**: FAISS index for quick semantic search.
- **Clean UX**: Streamlit interface for uploads and Q&A.
- **Cloud‑ready**: Streamlit Cloud compatible with simple secrets config.

---

## 🧠 How it works
1. **Ingest**: The PDF is parsed and split into context‑preserving chunks.
2. **Embed**: Each chunk is converted into embeddings using OpenAI.
3. **Index**: Vectors are stored locally in FAISS for similarity search.
4. **Retrieve + Answer**: Top‑K chunks are retrieved and sent to the LLM to craft a grounded response.

---

## ✅ Features
- Upload a legal PDF and build a FAISS index on the fly
- Ask natural‑language questions and receive context‑grounded answers
- Simple, production‑style separation of indexing and retrieval

---

## 🧰 Tech Stack
- **Frontend**: Streamlit
- **RAG**: LangChain + FAISS
- **PDF parsing**: PyMuPDF
- **LLM**: OpenAI (Chat Completions)
- **Runtime**: Python 3.11

---

## 📁 Project Structure
- [app.py](app.py) — Streamlit UI and RAG workflow
- [rag_index_builder.py](rag_index_builder.py) — PDF ingestion and FAISS index creation
- [tools.py](tools.py) — Retrieval utilities
- [requirements.txt](requirements.txt) — Python dependencies

---

## 🚀 Local Setup
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Add environment variables:

Create a .env file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
```

3. Run the app:

```bash
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud
1. Push the repo to GitHub.
2. In Streamlit Cloud, set the following secrets:

```toml
OPENAI_API_KEY = "your_api_key_here"
OPENAI_BASE_URL = "https://api.openai.com/v1"
```

3. Deploy using the default entry point: [app.py](app.py)

---

## 🧪 Example Questions
- “What are the termination clauses?”
- “Is there a pet policy?”
- “What are the late fees?”
- “When is the security deposit refundable?”

---

## 🧭 Roadmap
- OCR support for scanned PDFs
- Multi‑document comparison
- Exportable answer summaries
- Metadata‑aware filtering

---

## 📜 License
MIT — see [LICENSE](LICENSE)
