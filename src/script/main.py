"""
Insurance Chatbot - FastAPI Backend
Imports directly from query_engine.py (your provided file)

Run with:
    uvicorn main:app --reload --port 8000
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import uuid
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ─── Import your existing engine directly ────────────────────────────────────
# Make sure query_engine.py is in the same directory as this file
from query_engine import InsuranceQueryEngine


# =====================================================
# ---------------------- APP SETUP --------------------
# =====================================================

app = FastAPI(
    title="Insurance Chatbot API",
    description="Structured FAQ + RAG fallback chatbot for insurance queries",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared engine instance — holds all in-memory sessions
engine = InsuranceQueryEngine()


# =====================================================
# ---------------------- SCHEMAS ----------------------
# =====================================================

class ChatRequest(BaseModel):
    session_id: Optional[str] = None    # auto-generated if omitted
    product: Optional[str] = "zucora"  # e.g. "car", "health", "general"
    user_query: str                      # free-text OR a predefined question label


class ChatResponse(BaseModel):
    session_id: str
    source: str                                  # "structured_faq" | "rag"
    topic: Optional[str] = None                  # filled for structured FAQ hits
    answer_blocks: Optional[List[Any]] = None    # structured blocks (FAQ path)
    answer: Optional[str] = None                 # plain-text answer (RAG path)


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[str]


# =====================================================
# ---------------------- ROUTES -----------------------
# =====================================================

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Insurance Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    """
    Main chat endpoint — used for BOTH:
      • Predefined clickable questions  e.g. "Raise a claim"
      • Free-form typed queries         e.g. "what does my plan cover?"

    Flow (mirrors your InsuranceQueryEngine.handle_query):
      1. Try structured FAQ  →  ChromaDB vector search → MongoDB fetch
      2. Fallback to RAG     →  OpenAI LLM + retrieved context
    """
    # Auto-generate session_id when the frontend opens a new chat popup
    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = engine.handle_query(
            session_id=session_id,
            product=request.product,
            user_query=request.user_query,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return ChatResponse(
        session_id=session_id,
        source=result.get("source"),
        topic=result.get("topic"),
        answer_blocks=result.get("answer_blocks"),  # populated for structured_faq
        answer=result.get("answer"),                 # populated for rag
    )


@app.get("/session/{session_id}/history", response_model=SessionHistoryResponse, tags=["Session"])
def get_session_history(session_id: str):
    """
    Returns the full query history for a session.
    Call this on popup re-open to restore conversation context.
    """
    session = engine.memory.get(session_id)
    return SessionHistoryResponse(
        session_id=session_id,
        history=session.get("history", []),
    )


@app.delete("/session/{session_id}", tags=["Session"])
def clear_session(session_id: str):
    """
    Wipes a session from memory.
    Call this when the user closes the chat popup (optional clean-up).
    """
    if session_id in engine.memory.sessions:
        del engine.memory.sessions[session_id]
        return {"status": "cleared", "session_id": session_id}
    return {"status": "not_found", "session_id": session_id}


@app.get("/predefined-questions", tags=["Config"])
def get_predefined_questions(product: str = "zucora"):
    """
    Returns the first 2 topics from MongoDB structured_faqs collection,
    each with their list of FAQ questions.

    Frontend renders these as clickable items — clicking sends the question
    text as `user_query` to POST /chat, which runs through your full
    InsuranceQueryEngine (FAQ match → RAG fallback).

    Query param:
        product (str): filters by product field in MongoDB. Default: "general"
    """
    try:
        # Reuse the already-open mongo connection inside faq_engine
        collection = engine.faq_engine.collection

        # Fetch first 2 topic documents for the given product
        topics_cursor = collection.find(
            {"product": product},
            {"_id": 0, "topic_id": 1, "topic_name": 1, "faqs": 1}
        ).limit(2)

        topics = list(topics_cursor)

        if not topics:
            raise HTTPException(
                status_code=404,
                detail=f"No structured FAQs found for product='{product}'"
            )

        result = []
        for topic in topics:
            # Extract just the question text from each FAQ entry
            questions = [faq["question"] for faq in topic.get("faqs", [])]

            result.append({
                "topic_id":   topic.get("topic_id"),
                "topic_name": topic.get("topic_name", topic.get("topic_id")), # fallback to topic_id if no display name
                "questions":  questions
            })

        return {"product": product, "topics": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# ---------------------- ENTRY ------------------------
# =====================================================

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)