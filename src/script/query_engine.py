import uuid
from typing import Dict, Any, Optional
from loguru import logger
from pymongo import MongoClient
from langchain_openai import ChatOpenAI
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from script.vectorstore import VectorStore


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

        self.mongo = MongoClient(settings.mongodb_uri)
        self.db = self.mongo[settings.mongodb_db]
        self.collection = self.db["structured_faqs"]
        self.link_collection = self.db["link_metadata"]

    def search(self, query: str, product: str, k: int = 1):
        """
        Uses vector search over structured FAQ embeddings.
        """

        results = self.vector_store.similarity_search(query, k=k)

        if not results:
            return None

        best = results[0]
        
        if best["score"] is None or best["score"] < 0.60:
            logger.info("No strong FAQ match found.")
            return None

        score = best["metadata"].get("score", 0.0) if hasattr(best, "metadata") else 0.0

        # You may adjust threshold
        if score and score < 0.60:
            return None

        topic_id = best["metadata"].get("topic_id")
        faq_id = best["metadata"].get("faq_id")
        question = best["metadata"].get("question")

        topic_data = self.collection.find_one({ # mongodb collection
            "product": product,
            "topic_id": topic_id
        })

        if not topic_data:
            return None

        for faq in topic_data["faqs"]:
            if (faq_id and faq.get("faq_id") == faq_id) or (question and faq["question"] == question):
                print({
                    "topic_id": topic_id,
                    "question": question,
                    "answer_blocks": faq["answer_blocks"],
                    "related":faq["related"]
                })
                return {
                    "topic_id": topic_id,
                    "question": question,
                    "answer_blocks": faq["answer_blocks"],
                    "related":faq["related"],
                    "link_id": faq.get("link_id"),
                    "link_url": faq.get("link_url")
                }
                
                

        return None


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
        self.mongo = MongoClient(settings.mongodb_uri)
        self.db = self.mongo[settings.mongodb_db]
        self.link_collection = self.db["link_metadata"]

        self.llm = ChatOpenAI(
            model=settings.openai_model_large,
            temperature=0.2,
            api_key=settings.openai_api_key
        )
        self.link_min_score = 0.55

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
6. Answer in exactly 3 small blocks of under 30 words without using the em dash.
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
                    link_doc = self.link_collection.find_one(
                        {"product": settings.default_product_name, "link_id": link_id},
                        {"_id": 0, "link_url": 1}
                    )
                    link_url = link_doc.get("link_url") if link_doc else None
                    if link_url:
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
        
        print(faq_match)

        if faq_match:
            logger.info("Structured FAQ match found.")

            self.memory.update(
                session_id,
                topic=faq_match["topic_id"],
                question=faq_match["question"],
                user_query=user_query,
                
            )

            response_data = {
                "source": "structured_faq",
                "topic": faq_match["topic_id"],
                "answer_blocks": faq_match["answer_blocks"],
                "related": faq_match["related"],
            }

            # Add only if present
            if faq_match.get("link_id"):
                response_data["link_id"] = faq_match["link_id"]

            if faq_match.get("link_url"):
                response_data["link_url"] = faq_match["link_url"]

            return response_data

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
