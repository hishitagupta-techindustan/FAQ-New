"""
Insurance Chatbot - FastAPI Backend
Imports directly from query_engine.py (your provided file)

Run with:
    uvicorn main:app --reload --port 8000
"""

import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

# _PROJECT_ROOT = Path(__file__).resolve().parent.parent
# sys.path.insert(0, str(_PROJECT_ROOT))

import uuid
from typing import Optional, List, Any
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from suggestions_engine import SuggestionEngine
from ingest_document_new import run_ingestion


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
suggestion_engine = SuggestionEngine()


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

    answer: Optional[str] = None                 # plain-text answer (RAG path)
    link_id: Optional[str] = None
    link_url: Optional[str] = None


class SessionHistoryResponse(BaseModel):
    session_id: str
    history: List[str]
    

class SuggestionRequest(BaseModel):
    partial_query: str
    product: Optional[str] = "zucora",
    is_followup: bool


class SuggestionItem(BaseModel):
    question: str
    topic_id: str
    topic_name: str
    match_type:   str = "semantic" 
    score: float


class SuggestionResponse(BaseModel):
    suggestions: List[SuggestionItem]
    query: str


class IngestResponse(BaseModel):
    status: str
    product: str
    filename: str



# =====================================================
# ---------------------- ROUTES -----------------------
# =====================================================

@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "message": "Insurance Chatbot API is running"}


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
def chat(request: ChatRequest):
    # Use provided session_id or generate a new one
    session_id = request.session_id or str(uuid.uuid4())

    # Check if existing session has history and log/restore if needed
    existing_session = engine.memory.get(session_id)
    if existing_session:
        history = existing_session.get("history", [])
    else:
        history = []

    try:
        result = engine.handle_query(
            session_id=session_id,
            product=request.product,
            user_query=request.user_query,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    print(result)
    
    return ChatResponse(
        session_id=session_id,
        source=result.get("source"),
        topic=result.get("topic"),
        answer=result.get("answer"),
        link_id=result.get("link_id"),
        link_url=result.get("link_url")
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
    Returns the first 2 topics from the vector store,
    each with their list of FAQ questions.

    Frontend renders these as clickable items — clicking sends the question
    text as `user_query` to POST /chat, which runs through your full
    InsuranceQueryEngine (FAQ match → RAG fallback).

    Query param:
        product (str): filters by product field in vector metadata. Default: "general"
    """
    try:
        docs = engine.faq_engine.vector_store.get_documents(
            filter_metadata={"product": product, "question_type": "original"},
            limit=200
        )

        if not docs:
            raise HTTPException(
                status_code=404,
                detail=f"No structured FAQs found for product='{product}'"
            )

        topics = {}
        order = []
        for doc in docs:
            meta = doc.get("metadata", {}) or {}
            topic_id = meta.get("topic_id", "")
            topic_name = meta.get("topic_name", "") or topic_id
            question = meta.get("question", "")
            if not topic_id or not question:
                continue
            if topic_id not in topics:
                topics[topic_id] = {
                    "topic_id": topic_id,
                    "topic_name": topic_name,
                    "questions": []
                }
                order.append(topic_id)
            topics[topic_id]["questions"].append(question)

        result = [topics[tid] for tid in order[:2]]

        return {"product": product, "topics": result}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    
@app.post("/suggest", response_model=SuggestionResponse, tags=["Suggestions"])
def suggest(request: SuggestionRequest):
    """
    Live typeahead suggestions endpoint.

    Call this on every keystroke (debounced ~300ms on frontend).
    Returns 4-5 ranked FAQ questions matching the partial input
    via keyword + semantic search.

    Body:
        partial_query (str): whatever the user has typed so far
        product (str):       product scope, default "zucora"

    Returns:
        suggestions: list of ranked question matches with source + score
    """
    if not request.partial_query or not request.partial_query.strip():
        return SuggestionResponse(suggestions=[], query=request.partial_query)

    try:
        results = suggestion_engine.suggest(
            partial_query=request.partial_query.strip(),
            product=request.product,
            is_followup=request.is_followup
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return SuggestionResponse(
        suggestions=[SuggestionItem(**r) for r in results],
        query=request.partial_query,
    )


@app.post("/ingest-pdf", response_model=IngestResponse, tags=["Ingestion"])
def ingest_pdf(
    file: UploadFile = File(None),
    pdf_url: Optional[str] = Form(None),
    product: str = Form("zucora"),
    reset_vector_store: bool = Form(False)
):
    """
    Upload a PDF or provide a PDF URL and run structured FAQ ingestion.
    """
    if file and file.filename:
        if not file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    elif not pdf_url:
        raise HTTPException(status_code=400, detail="Provide a PDF file or a pdf_url.")

    upload_dir = "data" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = 50 * 1024 * 1024  # 50 MB

    if file and file.filename:
        safe_name = f"{uuid.uuid4()}_{Path(file.filename).name}"
        pdf_path = upload_dir / safe_name
        try:
            with pdf_path.open("wb") as f:
                f.write(file.file.read())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save upload: {e}")
    else:
        parsed = urlparse(pdf_url)
        if parsed.scheme not in {"http", "https"}:
            raise HTTPException(status_code=400, detail="pdf_url must be http or https.")

        url_name = Path(parsed.path).name or "document.pdf"
        if not url_name.lower().endswith(".pdf"):
            url_name = f"{url_name}.pdf"

        safe_name = f"{uuid.uuid4()}_{url_name}"
        pdf_path = upload_dir / safe_name

        try:
            req = Request(pdf_url, headers={"User-Agent": "ZucoraIngest/1.0"})
            with urlopen(req, timeout=20) as resp, pdf_path.open("wb") as f:
                content_type = resp.headers.get("Content-Type", "")
                if "pdf" not in content_type.lower():
                    raise HTTPException(status_code=400, detail="pdf_url did not return a PDF.")

                total = 0
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise HTTPException(status_code=400, detail="PDF exceeds size limit (50 MB).")
                    f.write(chunk)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to download pdf_url: {e}")

    try:
        run_ingestion(
            pdf_path=pdf_path,
            product_name=product,
            reset_vector_store=reset_vector_store
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return IngestResponse(
        status="ok",
        product=product,
        filename=pdf_path.name
    )


# =====================================================
# ---------------------- ENTRY ------------------------
# =====================================================

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
