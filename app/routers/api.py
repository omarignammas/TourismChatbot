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
async def ask_question(request: Request, question: str = Form(...)):
    try:
        if not hasattr(request.app.state, 'qa_chain') or request.app.state.qa_chain is None:
            return {
                "error": "QA chain not available. This could be due to missing HF_TOKEN, vector store, or model initialization issues.",
                "message": "Please check the server logs and ensure all components are properly configured.",
                "demo_response": f"I'm a tourism assistant for the Tangier-Tetouan-Hoceima region. You asked about: '{question}'. In a full setup, I would provide detailed information about this beautiful region in Morocco."
            }
        
        result = request.app.state.qa_chain.invoke({"query": question})
        # Return clean text response instead of JSON with sources
        return {"message": result["result"]}
    except Exception as e:
        return {
            "error": str(e),
            "demo_response": f"I'm a tourism assistant for the Tangier-Tetouan-Hoceima region. You asked about: '{question}'. In a full setup, I would provide detailed information about this beautiful region in Morocco."
        }