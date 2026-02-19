

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
from typing import Dict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from retrieval.vectorstore import VectorStore


# =====================================================
# ---------------- Pydantic Schema --------------------
# =====================================================

class FAQItem(BaseModel):
    question: str = Field(..., min_length=5, max_length=120)
    question_variations: List[str] = Field(..., min_items=4, max_items=5)
    answer_blocks: List[str] = Field(..., min_items=2, max_items=3)
    related: List[str] = Field(..., min_items=1, max_items=3)
    link_id: Optional[str] = None
    link_url: Optional[str] = None
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

def generate_structured_faq(text: str, extra_prompt: Optional[str] = None) -> dict:
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

1. Extract AT LEAST 10 main topics from the document (10–14 is ideal).
   - Topics must be distinct, non-overlapping, and cover the full document breadth
   - Prefer structured, insurance-domain categories over generic phrasing

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

   - question_variations:
       * 4–5 professional rephrasings of the same question
       * Keep meaning identical; vary phrasing and syntax
       * Each should be short and customer-friendly
       * Do not add new information

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
          "question_variations": [
            "string",
            "string",
            "string",
            "string"
          ],
          "link_id": "string or null",
          "link_url": "string or null",
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

    final_prompt = prompt
    if extra_prompt:
        final_prompt += "\n\n" + extra_prompt.strip() + "\n"

    response = structured_llm.invoke(final_prompt + "\n\nDOCUMENT:\n" + text)

    logger.info("LLM structured output received.")

    return response.model_dump()


def load_links_from_xlsx(xlsx_path: Path) -> List[Dict[str, str]]:
    """
    Load link_id/link_url pairs from an XLSX file.
    Expects first two columns to be: link_id, link_url (header optional).
    """
    try:
        from openpyxl import load_workbook
    except Exception as e:
        raise RuntimeError("openpyxl is required to read .xlsx files") from e

    wb = load_workbook(filename=str(xlsx_path), read_only=True, data_only=True)
    ws = wb.active

    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []

    # Detect header row
    start_idx = 0
    header = [str(c).strip().lower() if c is not None else "" for c in rows[0]]
    if "link_id" in header or "linkid" in header or "link" in header:
        start_idx = 1

    links = []
    for row in rows[start_idx:]:
        if not row or len(row) < 2:
            continue
        link_id = str(row[0]).strip() if row[0] is not None else ""
        link_url = str(row[1]).strip() if row[1] is not None else ""
        if not link_id or not link_url:
            continue
        links.append({"link_id": link_id, "link_url": link_url})

    return links


def store_links_in_mongodb(product_name: str, links: List[Dict[str, str]]) -> None:
    if not links:
        return
    client = MongoClient(settings.mongodb_uri)
    db = client[settings.mongodb_db]
    collection = db["link_metadata"]

    for item in links:
        collection.update_one(
            {"product": product_name, "link_id": item["link_id"]},
            {"$set": {"product": product_name, "link_id": item["link_id"], "link_url": item["link_url"]}},
            upsert=True
        )


def add_links_to_faqs(structured_data: dict, product_name: str) -> dict:
    client = MongoClient(settings.mongodb_uri)
    db = client[settings.mongodb_db]
    collection = db["link_metadata"]

    link_map: Dict[str, str] = {}
    for doc in collection.find({"product": product_name}, {"_id": 0, "link_id": 1, "link_url": 1}):
        link_id = doc.get("link_id")
        link_url = doc.get("link_url")
        if link_id and link_url:
            link_map[link_id] = link_url

    for topic in structured_data.get("topics", []):
        for faq in topic.get("faqs", []):
            link_id = faq.get("link_id")
            if link_id:
                faq["link_url"] = link_map.get(link_id)
            else:
                faq["link_url"] = None

    return structured_data


def enrich_structured_faq(structured_data: dict, product_name: str) -> dict:
    """Add stable IDs and normalize question variations."""
    for topic in structured_data.get("topics", []):
        topic_id = topic.get("topic_id", "").strip()
        for idx, faq in enumerate(topic.get("faqs", [])):
            faq_id = f"{product_name}_{topic_id}_{idx}"
            faq["faq_id"] = faq_id

            variations = [v.strip() for v in faq.get("question_variations", []) if v and v.strip()]
            # Ensure unique, preserve order
            seen = set()
            normalized = []
            for v in variations:
                if v not in seen:
                    normalized.append(v)
                    seen.add(v)
            faq["question_variations"] = normalized

    return structured_data


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
                    question_vector_store: VectorStore,
                    rag_vector_store: VectorStore):

    logger.info("Preparing documents for embedding...")

    question_documents = []
    rag_documents = []

    for topic in structured_data["topics"]:
        topic_id = topic["topic_id"]

        for idx, faq in enumerate(topic["faqs"]):
            faq_id = faq.get("faq_id", f"{product_name}_{topic_id}_{idx}")
            question_variations = faq.get("question_variations", [])
            link_id = faq.get("link_id")

            # Question suggestion store: one entry per question/variation
            all_questions = [faq["question"]] + question_variations
            for q_idx, question_text in enumerate(all_questions):
                question_documents.append({
                    "id": f"{faq_id}_q{q_idx}",
                    "text": question_text.strip(),
                    "metadata": {
                        "product": product_name,
                        "topic_id": topic_id,
                        "faq_id": faq_id,
                        "question": question_text.strip(),
                        "link_id": link_id,
                        "question_type": "original" if q_idx == 0 else "variation",
                        "question_index": q_idx
                    }
                })

            combined_text = f"""
            Product: {product_name}
            Topic: {topic['title']}
            Question: {faq['question']}
            Question Variations: {' | '.join(question_variations)}
            Answer: {' '.join(faq['answer_blocks'])}
            """

            rag_documents.append({
                "id": f"{faq_id}_rag",
                "text": combined_text.strip(),
                "metadata": {
                    "product": product_name,
                    "topic_id": topic_id,
                    "faq_id": faq_id,
                    "link_id": link_id,
                    "question": faq["question"]
                }
            })

    question_vector_store.add_structured_documents(question_documents)
    rag_vector_store.add_structured_documents(rag_documents)

    logger.info("Structured FAQ embedded into question and RAG vector stores.")


# =====================================================
# ---------------------- MAIN -------------------------
# =====================================================

def run_ingestion(pdf_path: Path,
                  product_name: str,
                  reset_vector_store: bool = False) -> None:
    """
    Structured ingestion runner for a specific PDF.
    """

    logger.add("logs/structured_ingestion.log", rotation="1 day")

    logger.info("Starting structured ingestion...")
    logger.info(f"PDF: {pdf_path}")
    logger.info(f"Product: {product_name}")

    if not pdf_path.exists():
        logger.error("PDF file not found.")
        print("❌ PDF file does not exist.")
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    vector_store = VectorStore(
        collection_name=settings.chroma_collection_name_questions,
        persist_directory=settings.chroma_persist_directory,
        embedding_model=settings.embedding_model
    )

    rag_vector_store = VectorStore(
        collection_name=settings.chroma_collection_name_rag,
        persist_directory=settings.chroma_persist_directory,
        embedding_model=settings.embedding_model
    )

    if reset_vector_store:
        logger.warning("Resetting vector stores...")
        vector_store.reset()
        rag_vector_store.reset()

    try:
        # Step 1: Extract text
        logger.info("Extracting text from PDF...")
        text = extract_full_text(pdf_path)

        # Step 1.5: Load optional link metadata
        xlsx_candidates = list(Path(settings.data_dir).glob("*.xlsx"))
        link_items: List[Dict[str, str]] = []
        if xlsx_candidates:
            link_items = load_links_from_xlsx(xlsx_candidates[0])
            store_links_in_mongodb(product_name, link_items)

        # Step 2: Generate structured FAQ
        if link_items:
            link_ids = ", ".join([li["link_id"] for li in link_items])
            link_prompt = f"""
LINK METADATA:
You may attach a relevant link_id to an FAQ ONLY if it directly helps the answer.
Use ONLY these link_ids: {link_ids}
If no link applies, set link_id to null.
Do not invent ids or urls. Only output link_id; link_url must be null.
"""
            structured_faq = generate_structured_faq(text, extra_prompt=link_prompt)
        else:
            structured_faq = generate_structured_faq(text)
        structured_faq = enrich_structured_faq(structured_faq, product_name)
        structured_faq = add_links_to_faqs(structured_faq, product_name)

        if len(structured_faq.get("topics", [])) < 10:
            logger.warning("LLM returned fewer than 10 topics; consider regenerating.")

        # Step 3: Store JSON
        store_in_mongodb(product_name, structured_faq)

        # Step 4: Embed into vector DB
        embed_and_store(
            structured_faq,
            product_name,
            vector_store,
            rag_vector_store
        )

        logger.info("Structured ingestion completed successfully.")
        print("\n✅ Structured FAQ ingestion complete!")

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    run_ingestion(
        pdf_path=Path(settings.pdf_dir),
        product_name=settings.default_product_name,
        reset_vector_store=settings.reset_vector_store
    )
