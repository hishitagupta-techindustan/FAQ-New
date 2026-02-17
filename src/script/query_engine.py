import uuid
from typing import Dict, Any, Optional
from loguru import logger
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from retrieval.vectorstore import VectorStore


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
               user_query: Optional[str] = None):

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
            collection_name=settings.chroma_collection_name,
            persist_directory=settings.chroma_persist_directory,
            embedding_model=settings.embedding_model
        )

        self.mongo = MongoClient(settings.mongodb_uri)
        self.db = self.mongo[settings.mongodb_db]
        self.collection = self.db["structured_faqs"]

    def search(self, query: str, product: str, k: int = 1):
        """
        Uses vector search over structured FAQ embeddings.
        """

        results = self.vector_store.similarity_search(query, k=k)

        if not results:
            return None

        best = results[0]
        
        if best["score"] is None or best["score"] < 0.50:
            logger.info("No strong FAQ match found.")
            return None

        score = best["metadata"].get("score", 0.0) if hasattr(best, "metadata") else 0.0

        # You may adjust threshold
        if score and score < 0.50:
            return None

        topic_id = best["metadata"].get("topic_id")

        question = best["metadata"].get("question")

        topic_data = self.collection.find_one({
            "product": product,
            "topic_id": topic_id
        })

        if not topic_data:
            return None

        for faq in topic_data["faqs"]:
            if faq["question"] == question:
                return {
                    "topic_id": topic_id,
                    "question": question,
                    "answer_blocks": faq["answer_blocks"]
                }

        return None


# =====================================================
# -------------------- RAG Engine ---------------------
# =====================================================

class RAGEngine:

    def __init__(self):
        self.vector_store = VectorStore(
            collection_name=settings.chroma_collection_name,
            persist_directory=settings.chroma_persist_directory,
            embedding_model=settings.embedding_model
        )

        self.llm = ChatOpenAI(
            model=settings.openai_model_large,
            temperature=0.2,
            api_key=settings.openai_api_key
        )

    def generate(self, query: str, memory_context: str):

        docs = self.vector_store.similarity_search(query, k=4)
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
3. If you cannot find the answer in the context, say so clearly
4. Be clear, concise, and accurate
5. If the question is ambiguous, ask for clarification
6. Answer in 3 small blocks of under 20 words.
"""

        response = self.llm.invoke(prompt)
        return response.content


# =====================================================
# -------------------- Orchestrator -------------------
# =====================================================

class InsuranceQueryEngine:

    def __init__(self):
        self.memory = SessionMemory()
        self.faq_engine = StructuredFAQEngine()
        self.rag_engine = RAGEngine()
        

    def handle_query(self,
                     session_id: str,
                     product: str,
                     user_query: str):

        logger.info(f"Processing query: {user_query}")

        session = self.memory.get(session_id)
        product = settings.default_product_name

        # ===============================
        # 1️⃣ Try Structured FAQ First
        # ===============================
        faq_match = self.faq_engine.search(user_query, product)

        if faq_match:
            logger.info("Structured FAQ match found.")

            self.memory.update(
                session_id,
                topic=faq_match["topic_id"],
                question=faq_match["question"],
                user_query=user_query
            )

            return {
                "source": "structured_faq",
                "topic": faq_match["topic_id"],
                "answer_blocks": faq_match["answer_blocks"]
            }

        # ===============================
        # 2️⃣ Fallback to RAG
        # ===============================

        logger.info("Falling back to RAG...")

        memory_context = f"""
Current topic: {session.get("current_topic")}
Last question: {session.get("last_question")}
Recent history: {session.get("history")[-3:]}
"""

        rag_answer = self.rag_engine.generate(
            query=user_query,
            memory_context=memory_context
        )

        self.memory.update(
            session_id,
            user_query=user_query
        )

        return {
            "source": "rag",
            "answer": rag_answer
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
