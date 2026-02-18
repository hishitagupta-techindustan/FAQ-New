

"""
Structured FAQ Ingestion Script
PDF → Full Text → LLM → Structured JSON → MongoDB → Vector DB
"""

"""
Structured FAQ Ingestion Script
PDF → Full Text → LLM (Pydantic Structured Output) →
MongoDB → Vector Store (Chroma)
"""

import sys
from pathlib import Path
from loguru import logger
from pymongo import MongoClient
import fitz
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from retrieval.vectorstore import VectorStore


# =====================================================
# ---------------- Pydantic Schema --------------------
# =====================================================

class FAQItem(BaseModel):
    question: str = Field(..., min_length=5, max_length=120)
    answer_blocks: List[str] = Field(..., min_items=2, max_items=3)
    related: List[str] = Field(..., min_items=1, max_items=3)
    next_prompt: Optional[str] = None


class Topic(BaseModel):
    topic_id: str
    title: str
    summary: str
    faqs: List[FAQItem]


class StructuredFAQ(BaseModel):
    topics: List[Topic]


# =====================================================
# ---------------- PDF Extraction ---------------------
# =====================================================

def extract_full_text(pdf_path: Path) -> str:
    """Extract full cleaned text from PDF."""
    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        full_text += page.get_text() + "\n"

    doc.close()

    # Basic cleaning
    full_text = " ".join(full_text.split())

    return full_text


# =====================================================
# --------- LLM Structured FAQ Generation ------------
# =====================================================

def generate_structured_faq(text: str) -> dict:
    """Generate structured FAQ JSON using Pydantic schema."""

    logger.info("Initializing LLM...")

    llm = ChatOpenAI(
        model=settings.openai_model_large,
        temperature=settings.llm_temperature,
        api_key=settings.openai_api_key
    )

    structured_llm = llm.with_structured_output(StructuredFAQ)

    prompt = """You are building a professional, structured insurance FAQ system similar to ACKO.

Your task is to convert the provided document into a clean, structured, UI-ready FAQ JSON.

OBJECTIVE:
Create a professional, customer-centric FAQ knowledge structure suitable for a modern insurance app.

-----------------------------------------
STRUCTURE REQUIREMENTS
-----------------------------------------

1. Extract 5–10 main topics from the document.

2. Each topic MUST include:

   - topic_id: snake_case, short and stable identifier
   - title: 2–4 words, professional and category-level
            (e.g., "Policy Coverage Details", "Claims & Settlement Process")
   - summary: One concise line explaining what this section covers
   - faqs: Exactly 5 FAQ entries per topic

3. Each FAQ MUST include:

   - question:
       * User-centric as well as professional phrasing
       * Start with What / How / When / Can I / Does my / Why
       * Natural customer language
       * Short and mobile-friendly (max 12–14 words)

   - answer_blocks:
       * 3 blocks exactly
       * Each block max between 10-20 words.
       * Clear, simple language
       * Slightly descriptive but not verbose
       * Avoid legal jargon
       * No marketing exaggeration
       * Directly address the customer ("you", "your")

   - related:
       * 2 related questions from the same topic
       * Must be actual questions from that topic

   - next_prompt:
       * A short clarification prompt ONLY if a follow-up may help
       * Example: "Are you asking about a new policy or renewal?"
       * Otherwise return null

-----------------------------------------
CONTENT RULES
-----------------------------------------

- Use ONLY information explicitly available in the document.
- Do NOT hallucinate features, benefits, limits, numbers, or coverage.
- Do NOT invent regulatory details.
- If something is unclear, simplify — do not assume.
- Maintain factual accuracy strictly.
- Avoid internal or company-centric phrasing.
- Do not include disclaimers unless present in the document.

-----------------------------------------
STYLE GUIDELINES
-----------------------------------------

TOPIC TITLES:
- Professional and structured
- Avoid casual wording like "About Insurance"
- Think category-level clarity

QUESTIONS:
- Must reflect real customer concerns
- Practical and scenario-driven where possible
- Avoid generic phrasing like "Explain policy"

ANSWERS:
- Organized into 3 UI blocks
- Clear and reassuring tone
- Informative but crisp
- No long paragraphs
- No bullet points
- Each block should feel scannable on mobile

-----------------------------------------
OUTPUT FORMAT
-----------------------------------------

Return STRICTLY valid JSON matching this structure:

{
  "topics": [
    {
      "topic_id": "string",
      "title": "string",
      "summary": "string",
      "faqs": [
        {
          "question": "string",
          "answer_blocks": [
            "string",
            "string",
            "string"
          ],
          "related": [
            "string",
            "string"
          ],
          "next_prompt": "string or null"
        }
      ]
    }
  ]
}

Do not include explanations.
Do not wrap in markdown.
Return JSON only.

"""

    logger.info("Calling LLM for structured FAQ generation...")

    response = structured_llm.invoke(prompt + "\n\nDOCUMENT:\n" + text)

    logger.info("LLM structured output received.")

    return response.model_dump()


# =====================================================
# ---------------- MongoDB Storage --------------------
# =====================================================

def store_in_mongodb(product_name: str, structured_data: dict):
    logger.info("Connecting to MongoDB...")

    client = MongoClient(settings.mongodb_uri)
    db = client[settings.mongodb_db]
    collection = db["structured_faqs"]

    for topic in structured_data["topics"]:
        topic["product"] = product_name

        collection.update_one(
            {
                "product": product_name,
                "topic_id": topic["topic_id"]
            },
            {"$set": topic},
            upsert=True
        )

    logger.info("Structured FAQ stored successfully in MongoDB.")


# =====================================================
# ---------------- Vector Store Embed -----------------
# =====================================================

def embed_and_store(structured_data: dict,
                    product_name: str,
                    vector_store: VectorStore):

    logger.info("Preparing documents for embedding...")

    documents = []

    for topic in structured_data["topics"]:
        topic_id = topic["topic_id"]

        for idx, faq in enumerate(topic["faqs"]):

            combined_text = f"""
            Product: {product_name}
            Topic: {topic['title']}
            Question: {faq['question']}
            Question:{faq['question']}
            Answer: {' '.join(faq['answer_blocks'])}
            """

            documents.append({
                "id": f"{product_name}_{topic_id}_{idx}",
                "text": combined_text.strip(),
                "metadata": {
                    "product": product_name,
                    "topic_id": topic_id,
                    "question": faq["question"]
                }
            })

    vector_store.add_structured_documents(documents)

    logger.info("Structured FAQ embedded into vector store.")


# =====================================================
# ---------------------- MAIN -------------------------
# =====================================================

def main():
    """
    Simple structured ingestion runner.
    Configure PDF path and product inside this function.
    """

    # ===== CONFIGURATION =====
    PDF_PATH = settings.pdf_dir
    PRODUCT_NAME = settings.default_product_name
    RESET_VECTOR_STORE = settings.reset_vector_store
    # =========================

    logger.add("logs/structured_ingestion.log", rotation="1 day")

    logger.info("Starting structured ingestion...")
    logger.info(f"PDF: {PDF_PATH}")
    logger.info(f"Product: {PRODUCT_NAME}")

    pdf_path = Path(PDF_PATH)

    if not pdf_path.exists():
        logger.error("PDF file not found.")
        print("❌ PDF file does not exist.")
        return

    vector_store = VectorStore(
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_directory,
        embedding_model=settings.embedding_model
    )

    if RESET_VECTOR_STORE:
        logger.warning("Resetting vector store...")
        vector_store.reset()

    try:
        # Step 1: Extract text
        logger.info("Extracting text from PDF...")
        text = extract_full_text(pdf_path)

        # Step 2: Generate structured FAQ
        structured_faq = generate_structured_faq(text)

        # Step 3: Store JSON
        store_in_mongodb(PRODUCT_NAME, structured_faq)

        # Step 4: Embed into vector DB
        embed_and_store(
            structured_faq,
            PRODUCT_NAME,
            vector_store
        )

        logger.info("Structured ingestion completed successfully.")
        print("\n✅ Structured FAQ ingestion complete!")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()

