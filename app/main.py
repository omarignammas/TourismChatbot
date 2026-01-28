from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routers.api import router as api_router
from app.models.langchain_model import llm, vector_store
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_chain
    
    # Check if components are available
    if llm is None or vector_store is None:
        print("⚠️  Warning: LLM or vector store not available. QA chain will not be loaded.")
        qa_chain = None
        app.state.qa_chain = None
    else:
        try:
            # Import RetrievalQA from langchain_classic
            from langchain_classic.chains import RetrievalQA
            
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
                return_source_documents=True
            )
            app.state.qa_chain = qa_chain
            print("✅ QA chain loaded successfully!")
        except Exception as e:
            print(f"⚠️  Error loading QA chain: {e}")
            qa_chain = None
            app.state.qa_chain = None
    
    yield  

app = FastAPI(
    title="Tourism ChatBot API",
    description="A RAG-based chatbot for Tangier-Tetouan-Hoceima region tourism information",
    version="1.0.0",
    lifespan=lifespan
)
app.include_router(api_router)
