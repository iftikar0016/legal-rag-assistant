import os
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from autogen import AssistantAgent, register_function

# Load environment variables (optional for local dev with .env)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Load OpenAI credentials from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL")

# Define the assistant agent
legal_assistant = AssistantAgent(
    name="LegalAssistant",
    llm_config={
        "config_list": [
            {
                "api_key": OPENAI_API_KEY,
                "base_url": OPENAI_BASE_URL,
                "model": "gpt-5-mini",  # Set to your preferred OpenAI model
            }
        ],
        "temperature": 0
    }
)

def retrieve_legal_context(query: str) -> str:
    embeddings = OpenAIEmbeddings(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    db = FAISS.load_local("rag_faiss_store", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(query, k=3)
    print(docs)
    return "\n\n".join([doc.page_content for doc in docs])

# ✅ Register it manually using function call style
register_function(
    retrieve_legal_context,
    caller=legal_assistant,
    executor=legal_assistant,
    description="Retrieve relevant legal context from the indexed legal documents based on the query."
)

# Test the function
if __name__ == "__main__":
    query = "Are the pets allowed in the property?"
    context = retrieve_legal_context(query)
    print("Retrieved Context:\n", context)