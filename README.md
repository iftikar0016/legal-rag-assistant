# ContractLens: Legal RAG Assistant ⚖️🤖

An intelligent Retrieval-Augmented Generation (RAG) system designed to extract sections from legal documents and provide contextual answers using a multi-agent orchestration framework.

---

## 🚀 Overview
**ContractLens** is a specialized tool for legal professionals and individuals to query complex legal documents (like rental agreements or contracts) using natural language. It leverages **LangChain** for document processing, **FAISS** for vector search, and **Microsoft AutoGen** for a multi-agent conversation flow that ensures accuracy and citations.

### Key Features
- **Semantic Retrieval**: Uses OpenAI Embeddings and FAISS to find the most relevant clauses, even if keywords don't match exactly.
- **Multi-Agent Orchestration**: Utilizes a `LegalAssistant` agent (tasked with retrieval) and a `UserProxyAgent` (tasked with execution) to mirror a professional consultation workflow.
- **On-the-Fly Indexing**: Upload a PDF and build a searchable vector database in seconds.
- **Interactive UI**: Built with Streamlit for a seamless, responsive user experience.

---

## 🛠️ Tech Stack
- **Frameworks**: [LangChain](https://www.langchain.com/), [Microsoft AutoGen](https://microsoft.github.io/autogen/)
- **Core AI**: OpenAI GPT-4/5, OpenAI Embeddings
- **Vector Store**: [FAISS](https://github.com/facebookresearch/faiss)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **PDF Processing**: PyMuPDF (fitz)
- **Environment**: Python 3.11+, Docker

---

## 🏗️ Technical Architecture
1. **Ingestion**: PDFs are parsed and split into chunks using `RecursiveCharacterTextSplitter` to maintain legal context.
2. **Embedding**: Chunks are converted into high-dimensional vectors via `text-embedding-3-small`.
3. **Storage**: Vectors are indexed locally using FAISS for high-speed similarity search.
4. **Agentic Logic**: 
    - The **User Agent** receives the query and triggers the retrieval tool.
    - The **Legal Assistant** retrieves the top 3 relevant context blocks.
    - The LLM synthesizes an answer strictly based on the provided context.

---

## 🎯 Example Queries
Try asking the assistant:
- "What are the termination conditions for this lease?"
- "Is there a pet policy mentioned in the agreement?"
- "What is the security deposit amount and when is it refundable?"
- "Are there any late fees for rent payment?"

---

cd legal-rag-assistant
```

### 2. Set Up Environment
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1  # Or your custom proxy
```

### 3. Using Docker (Recommended)
```bash
docker build -t contractlens .
docker run -p 8501:8501 --env-file .env contractlens
```

### 4. Running Locally
```bash
python -m venv .venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows
pip install -r requirements.txt
streamlit run app.py
```

---

## 🧑‍💻 Interview Highlights (Talking Points)
- **Why AutoGen?** "I chose AutoGen to separate the *retrieval* logic from the *reasoning* logic. This allows the system to verify the context before answering, reducing hallucinations common in simple RAG pipelines."
- **Why FAISS?** "For a local document assistant, FAISS provides the best balance of speed and simplicity without the overhead of a cloud-native vector database like Pinecone."
- **Challenges Overcome**: "Handling legal text requires careful chunking. I used a recursive splitter with overlap to ensure that clauses spanning multiple pages aren't lost."

---

## � Future Roadmap
- [ ] **OCR Integration**: Support for scanned image-based PDFs using Tesseract or Azure Form Recognizer.
- [ ] **Multi-Document Chat**: Ability to upload multiple files and compare clauses across different contracts.
- [ ] **Exportable Summaries**: Generate a one-page summary of key risks and dates in the contract.
- [ ] **Metadata Filtering**: Filter searches by document type, date, or specific legal category.

---

## �📜 License
Distributable under the MIT License. See `LICENSE` for more information.
