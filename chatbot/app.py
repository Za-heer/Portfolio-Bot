import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
load_dotenv()

from .rag import RAGStore
from .model import LLMClient
from .memory import ConversationMemory


app = FastAPI(title="Portfolio Chatbot (RAG + HF Inference)")

# ✅ CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # you can restrict to your domain later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Health check endpoint (for Render/UptimeRobot)
@app.get("/healthz")
def healthz():
    """
    Simple health check endpoint for uptime monitoring.
    Returns 200 OK when app is running.
    """
    return {"status": "ok", "message": "Chatbot API is live."}


rag = RAGStore(portfolio_path="data/portfolio.json")
try:
    rag.load_index()
except Exception as e:
    # index might not exist yet; app still runs
    print("RAG index not loaded:", e)

llm = LLMClient()
sessions = {}  # session_id -> ConversationMemory


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    top_k: Optional[int] = 5


@app.post("/chat")
def chat(req: ChatRequest):
    if not req.message:
        raise HTTPException(status_code=400, detail="message required")

    sid = req.session_id or "default"
    mem = sessions.setdefault(sid, ConversationMemory(max_messages=12))
    mem.add_user(req.message)

    # retrieve context
    try:
        if rag.index is None and not hasattr(rag, "_fallback_embeddings"):
            # build index on the fly (only if small)
            rag.build_index()
        top_k = max(1, min(10, req.top_k or 5))
        retrieved = rag.query(req.message, top_k=top_k)
    except Exception as e:
        retrieved = []
        print("RAG retrieve error:", e)

    context = "\n\n---\n\n".join(
        [f"[{r['section']}] {r['text']}" for r in retrieved]
    ) if retrieved else "NO_RELEVANT_CONTEXT_FOUND"

    convo_text = mem.formatted()

    try:
        answer = llm.chat_with_context(
            context=context, conversation=convo_text, user_message=req.message
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {e}")

    if "<think>" in answer:
        answer = answer.split("</think>")[-1].strip()

    mem.add_assistant(answer)
    return {"response": answer, "sources": [r.get("section") for r in retrieved]}


@app.post("/build-index")
def build_index():
    try:
        rag.build_index(rebuild=True)
        return {"status": "index built"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
