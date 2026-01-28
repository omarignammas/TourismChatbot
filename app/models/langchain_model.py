from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
import os
import warnings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

warnings.filterwarnings("ignore")

HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize LLM with error handling
try:
    endpoint = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=HF_TOKEN,
        task="text-generation",
        max_new_tokens=512,
    )
    llm = ChatHuggingFace(llm=endpoint)
    print("✅ LLM initialized successfully")
except Exception as e:
    print(f"⚠️  Warning: LLM initialization failed: {e}")
    llm = None

# Initialize embeddings
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    print("✅ Embeddings model loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Embeddings initialization failed: {e}")
    embeddings = None

# Load vector store
try:
    vector_store = FAISS.load_local("faiss_index3", embeddings, allow_dangerous_deserialization=True)
    print("✅ Vector store loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Vector store loading failed: {e}")
    vector_store = None
"""
def retrieve_context(query: str) -> str:
    retrieved_docs = vector_store.similarity_search(query, k=4)
    return "\n\n".join(doc.page_content for doc in retrieved_docs)

tools = [  
    Tool(
        name="RetrieveContext",
        func=retrieve_context,
        description="Retrieves context about Tangier-Tetouan-Hoceima region to answer tourist questions."
    )
]"""
