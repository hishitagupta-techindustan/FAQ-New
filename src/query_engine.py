import uuid
from typing import Dict, Any, Optional
from loguru import logger
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from vectorstore import VectorStore


# =====================================================
# ------------------ Memory Manager -------------------
# =====================================================

class SessionMemory:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}

    def get(self, session_id: str) -> Dict[str, Any]:
        if session_id not in self.sessions:
            self.sessions[session_id] = {
                "current_topic": None,
                "last_question": None,
                "history": []
            }
        return self.sessions[session_id]

    def update(self,
               session_id: str,
               topic: Optional[str] = None,
               question: Optional[str] = None,
               user_query: Optional[str] = None,
               ):

        session = self.get(session_id)

        if topic:
            session["current_topic"] = topic

        if question:
            session["last_question"] = question

        if user_query:
            session["history"].append(user_query)

        return session


# =====================================================
# ---------------- Structured FAQ Engine --------------
# =====================================================

class StructuredFAQEngine:

    def __init__(self):
        self.vector_store = VectorStore(
            collection_name=settings.chroma_collection_name_questions,
            persist_directory=settings.chroma_persist_directory,
            embedding_model=settings.embedding_model
        )

    def search(self, query: str, product: str, k: int = 1):
        """
        Uses vector search over structured FAQ embeddings.
        """

        results = self.vector_store.similarity_search(
            query,
            k=k,
            filter_metadata={"product": product}
        )

        if not results:
            return None

        best = results[0]
        
        if best.get("score") is None or best.get("score") < 0.95:
            logger.info("No strong FAQ match found.")
            return None

        meta = best.get("metadata", {}) or {}
        topic_id = meta.get("topic_id")
        question = meta.get("question")

        

      

        return {
            "topic_id": topic_id,
            "question": question,
           
            "link_id": meta.get("link_id"),
            "link_url": meta.get("link_url")
        }

    def _normalize_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            import ast
            try:
                parsed = ast.literal_eval(value)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
        return []


# =====================================================
# -------------------- RAG Engine ---------------------
# =====================================================

class RAGEngine:

    def __init__(self):
        self.vector_store = VectorStore(
            collection_name=settings.chroma_collection_name_rag,
            persist_directory=settings.chroma_persist_directory,
            embedding_model=settings.embedding_model
        )

        self.llm = ChatOpenAI(
            model=settings.openai_model_large,
            temperature=0.2,
            api_key=settings.openai_api_key
        )
        self.link_min_score = 0.55

    def generate(self, query: str, memory_context: str):

        docs = self.vector_store.similarity_search(query, k=10)
        print(docs)

        context_text = "\n\n".join([doc["text"] for doc in docs])
        


        prompt = f"""
 You are a helpful insurance assistant. Answer the user's question based on the provided context.

Context from insurance documents:
{memory_context}

Retrieved knowledge:
{context_text}

User question:
{query}


Instructions:
1. If the question is not related to insurance, answer the question in short along with prompting the user to explore Zucora in a witty manner.
2. Answer based ONLY on the provided context
3. Do NOT say "the context doesn't specify" if any relevant information exists — use what's there.
4. Be clear, concise, and accurate
5. If the question is ambiguous, ask for clarification
6. Answer in exactly 3 small blocks of under 30-40 words without using the em dash.
7. Answers should be user centric yet professional.
"""

        response = self.llm.invoke(prompt)
        link_id = None
        link_url = None
        if docs:
            # Pick the best-scoring doc with a link_id above threshold
            for doc in docs:
                score = doc.get("score")
                meta = doc.get("metadata", {})
                cand_id = meta.get("link_id")
                if cand_id and score is not None and score >= self.link_min_score:
                    link_id = cand_id
                    link_url = meta.get("link_url")
                    break

        return {"answer": response.content, "link_id": link_id, "link_url": link_url}


# =====================================================
# -------------------- Orchestrator -------------------
# =====================================================

class InsuranceQueryEngine:

    def __init__(self):
        self.memory = SessionMemory()
        self.faq_engine = StructuredFAQEngine()
        self.rag_engine = RAGEngine()
        

    def handle_query(self, session_id: str, product: str, user_query: str):
        logger.info(f"Processing query: {user_query}")

        session = self.memory.get(session_id)
        product = settings.default_product_name

        memory_context = f"""
    Current topic: {session.get("current_topic")}
    Last question: {session.get("last_question")}
    Recent history: {session.get("history")[-3:]}
    """

        rag_answer = self.rag_engine.generate(
            query=user_query,
            memory_context=memory_context
        )

        self.memory.update(session_id, user_query=user_query)

        return {
            "source": "rag",
            "answer": rag_answer.get("answer") if isinstance(rag_answer, dict) else rag_answer,
            "link_id": rag_answer.get("link_id") if isinstance(rag_answer, dict) else None,
            "link_url": rag_answer.get("link_url") if isinstance(rag_answer, dict) else None
        }


# =====================================================
# ---------------------- MAIN -------------------------
# =====================================================

if __name__ == "__main__":

    logger.add("logs/query_engine.log", rotation="1 day")

    engine = InsuranceQueryEngine()

    session_id = str(uuid.uuid4())
    product = settings.default_product_name

    print("Insurance Assistant Ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        response = engine.handle_query(
            session_id=session_id,
            product=product,
            user_query=user_input
        )

        print("\nAssistant:", response, "\n")
