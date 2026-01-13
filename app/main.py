from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.routers.api import router as api_router
from app.models.langchain_model import llm, vector_store, RetrievalQA

@asynccontextmanager
async def lifespan(app: FastAPI):
    global qa_chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    app.state.qa_chain = qa_chain
    print("QA chain loaded!")
    yield  # <-- important
    # optional shutdown logic here

app = FastAPI(lifespan=lifespan)
app.include_router(api_router)
