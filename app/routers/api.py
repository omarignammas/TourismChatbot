# app/routers/api.py
from fastapi import APIRouter, Form
from pydantic import BaseModel
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from fastapi import Request 

router = APIRouter()
class Query(BaseModel):
    question: str

@router.get("/")
async def home():
    return {"message": "Welcome dear tourist!"}

@router.get("/chat", response_class=HTMLResponse)
async def chat_form():
    return """
    <html>
        <head>
            <title>Ask a Question</title>
        </head>
        <body>
            <h2>Tourist Assistant</h2>
            <form action="/chat/ask" method="post">
                <input type="text" name="question" placeholder="Type your question..." style="width:300px;" />
                <button type="submit">Ask</button>
            </form>
        </body>
    </html>
    """
@router.post("/chat/ask")
async def ask_question(request: Request,question: str = Form(...)):
    try:
        if not hasattr(request.app.state, 'qa_chain') or request.app.state.qa_chain is None:
            return {"error": "Model still loading, try again in a few seconds"}
        
        result = request.app.state.qa_chain.invoke({"query": question})
        return result["result"]
    except Exception as e:
        return {"error": str(e)}