"""
SuggestionEngine — live "search-as-you-type" suggestions
Combines:
  • Keyword match  → computed from retrieved vector candidates (no MongoDB scan)
  • Semantic match → ChromaDB vector similarity (meaning-aware)
Results are merged, deduplicated, and ranked.
"""

import re
from typing import List, Dict, Any
from loguru import logger

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from script.vectorstore import VectorStore
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class LLMSuggestionSet(BaseModel):
    questions: List[str] = Field(default_factory=list, min_length=0, max_length=3)


class SuggestionEngine:

    # Minimum characters before we bother searching
    MIN_QUERY_LENGTH = 5

    # How many raw hits to pull from each source before merging
    KEYWORD_FETCH_LIMIT = 3
    SEMANTIC_FETCH_LIMIT = 3

    # Final number of suggestions returned to the frontend
    MAX_SUGGESTIONS = 5

    # Minimum semantic similarity score to include a result
    SEMANTIC_THRESHOLD = 0.65
    # Minimum score required to return merged suggestions
    MIN_SUGGESTION_SCORE = 0.60

    def __init__(self):
        self.question_store = VectorStore(
            collection_name=settings.chroma_collection_name_questions,
            persist_directory=settings.chroma_persist_directory,
            embedding_model=settings.embedding_model
        )
        self.rag_store = VectorStore(
            collection_name=settings.chroma_collection_name_rag,
            persist_directory=settings.chroma_persist_directory,
            embedding_model=settings.embedding_model
        )
        self.llm = ChatOpenAI(
            model=settings.openai_model_large,
            temperature=0.2,
            api_key=settings.openai_api_key,
            max_tokens=256
        )
        self.structured_llm = self.llm.with_structured_output(LLMSuggestionSet)

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def suggest(self, partial_query: str, product: str) -> List[Dict[str, Any]]:
        """
        Main entry point.
        Returns up to MAX_SUGGESTIONS question suggestions for the partial query.

        Each suggestion dict:
        {
            "question":  str,          # full FAQ question text
            "topic_id":  str,          # topic it belongs to
            "topic_name": str,         # human-readable topic label
            "match_type": str,         # "keyword" | "semantic" | "both"
            "score":     float         # 0.0–1.0, higher = better match
        }
        """
        query = partial_query.strip()

        if len(query) < self.MIN_QUERY_LENGTH:
            return []

        # 1) Semantic candidates (fast retrieval), similarity search
        candidates = self._semantic_candidates(query, product)

        # 2) Keyword matches over retrieved candidates
        keyword_hits = self._keyword_search(query, candidates)

        # If no keyword hits, try a wider keyword scan over stored questions
        if not keyword_hits:
            keyword_hits = self._keyword_search_all(query, product)

        # 3) Semantic suggestions
        semantic_hits = self._semantic_hits(candidates)

        merged = self._merge_and_rank(keyword_hits, semantic_hits)
        filtered = [m for m in merged if m.get("score", 0.0) >= self.MIN_SUGGESTION_SCORE]

        # Build a final list
        final: List[Dict[str, Any]] = []
        seen = set()

        query_norm = self._normalize_question(query)

        def add_items(items: List[Dict[str, Any]]):
            for it in items:
                q = it.get("question", "").strip()
                if not q:
                    continue
                if self._normalize_question(q) == query_norm:
                    continue
                key = q.lower()
                if key in seen:
                    continue
                seen.add(key)
                final.append(it)
                if len(final) >= self.MAX_SUGGESTIONS:
                    break

        add_items(filtered)

        if len(final) < self.MAX_SUGGESTIONS and self._is_valid_query(query):
            llm_suggestions = self._llm_context_fallback(query, product)
            add_items(llm_suggestions)

        return final[: self.MAX_SUGGESTIONS]

    def _normalize_question(self, text: str) -> str:
        t = text.strip().lower()
        t = re.sub(r"[^a-z0-9\\s]", "", t)
        t = re.sub(r"\\s+", " ", t)
        return t

    def _filter_items(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        query_norm = self._normalize_question(query)
        filtered = []
        seen = set()
        for it in items:
            q = it.get("question", "").strip()
            if not q:
                continue
            norm = self._normalize_question(q)
            if not norm or norm == query_norm:
                continue
            if norm in seen:
                continue
            seen.add(norm)
            filtered.append(it)
        return filtered

    def _is_valid_query(self, query: str) -> bool:
        # Require at least 2 alphabetic characters to avoid nonsense input
        return len(re.findall(r"[A-Za-z]", query)) >= 2


    def _llm_context_fallback(
        self,
        query: str,
        product: str
    ) -> List[Dict[str, Any]]:
        """
        Generate 3 questions using LLM, grounded strictly in retrieved context,
        so answers exist in the knowledge base.
        """
        try:
            docs = self.question_store.similarity_search(
                query,
                k=2,
                filter_metadata={"product": product}
            )
        except Exception as e:
            logger.error(f"RAG retrieval for LLM fallback failed: {e}")
            return []

        if not docs:
            return []

        # Build compact, answerable context from question + answer blocks
        context_parts = []
        for doc in docs:
            meta = doc.get("metadata", {}) or {}
            question = meta.get("question")
            answer_blocks = meta.get("answer_blocks")
            if not question or not answer_blocks:
                continue
            if isinstance(answer_blocks, str):
                # Best-effort parse if stored as string
                try:
                    import ast
                    parsed = ast.literal_eval(answer_blocks)
                    if isinstance(parsed, list):
                        answer_blocks = parsed
                except Exception:
                    answer_blocks = []
            if not answer_blocks:
                continue
            context_parts.append(
                f"Q: {question}\nA: {' '.join(answer_blocks)}"
            )

        if not context_parts:
            return []

        context_text = "\n\n".join(context_parts)
        if not context_text.strip():
            return []

        try:
            # Only force inclusion if the user phrase looks sensible and appears in context
            phrase = query.strip()
            # Heuristic: only force the user phrase if it looks domain-relevant
            domain_terms = {
                "coverage", "claim", "repair", "warranty", "policy", "plan", "appliance",
                "service", "replacement", "damage", "accidental", "premium", "deductible",
                "eligible", "eligibility", "purchase", "buy", "price", "cost", "refund",
                "cancellation", "renewal", "inspection", "doorstep", "parts", "spare"
            }
            phrase_tokens = re.findall(r"[A-Za-z]{3,}", phrase.lower())
            phrase_is_sensible = bool(phrase_tokens) and any(t in domain_terms for t in phrase_tokens)

            include_clause = (
                "Include the exact user phrase in each question. "
                if phrase_is_sensible
                else "Do not include the user phrase if it is unrelated or nonsensical. "
            )

            # Keep context short to reduce latency
            max_chars = 600
            short_context = context_text[:max_chars]

            prompt = (
                "Generate 3 concise FAQ-style questions for an insurance assistant. "
                + include_clause +
                "Prioritize the user's query; use context only to ground facts. "
                "You MUST only use information present in the context. "
                "Ensure each question is answerable from the context. "
                "Do not repeat the user's query verbatim. "
                "Keep each question under 12 words. "
                "Return only the questions, no extra text.\n\n"
                f"Product: {product}\n"
                f"User query (highest priority): {query}\n\n"
                f"Context (supporting only):\n{short_context}"
            )
            resp = self.structured_llm.invoke(prompt)
            questions = resp.questions if resp else []
            results = []
            for q in questions:
                if not isinstance(q, str) or not q.strip():
                    continue
                results.append({
                    "question": q.strip(),
                    "topic_id": "",
                    "topic_name": "",
                    "match_type": "llm_context",
                    "score": 0.2
                })
            return results
        except Exception as e:
            logger.error(f"LLM fallback suggestions error: {e}")
            return []

    # ------------------------------------------------------------------
    # KEYWORD SEARCH  (Candidates)
    # ------------------------------------------------------------------

    def _keyword_search(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Case-insensitive keyword scoring over retrieved candidates.
        Splits the query into individual words and scores by coverage.
        """
        words = [w for w in re.split(r"\W+", query) if len(w) > 1]

        if not words:
            return []

        results = []
        for doc in candidates:
            q_text = doc.get("question") or ""
            q_lower = q_text.lower()
            q_lower_query = query.lower()

            # Simple relevance: boost if query appears as a contiguous substring
            if q_lower_query in q_lower:
                score = 0.80
            else:
                # Score by fraction of query words matched
                matched = sum(1 for w in words if w.lower() in q_lower)
                if matched == 0:
                    continue
                score = 0.50 + 0.30 * (matched / len(words))

            results.append({
                "question":   q_text,
                "topic_id":   doc.get("topic_id", ""),
                "topic_name": doc.get("topic_name", doc.get("topic_id", "")),
                "match_type": "keyword",
                "score":      round(score, 3),
            })

        return results[: self.KEYWORD_FETCH_LIMIT]

    def _keyword_search_all(self, query: str, product: str) -> List[Dict[str, Any]]:
        """
        Wider keyword scan over stored questions (no cache).
        Uses get_documents with a capped limit to avoid heavy scans.
        """
        try:
            docs = self.question_store.get_documents(
                filter_metadata={"product": product, "question_type": "original"},
                limit=300
            )
        except Exception as e:
            logger.error(f"Keyword scan get_documents failed: {e}")
            return []

        candidates = []
        for doc in docs:
            meta = doc.get("metadata", {}) or {}
            question = meta.get("question")
            if not question:
                continue
            candidates.append({
                "question": question,
                "topic_id": meta.get("topic_id", ""),
                "topic_name": meta.get("topic_name", meta.get("topic_id", "")),
                "score": 0.0
            })

        return self._keyword_search(query, candidates)

    # ------------------------------------------------------------------
    # SEMANTIC SEARCH  (ChromaDB)
    # ------------------------------------------------------------------

    def _semantic_candidates(self, query: str, product: str) -> List[Dict[str, Any]]:
        """
        Retrieve semantic candidates from both question and RAG stores.
        """
        candidates: List[Dict[str, Any]] = []
        try:
            q_hits = self.question_store.similarity_search(
                query,
                k=self.SEMANTIC_FETCH_LIMIT,
                filter_metadata={"product": product}
            )
            r_hits = self.rag_store.similarity_search(
                query,
                k=self.SEMANTIC_FETCH_LIMIT,
                filter_metadata={"product": product}
            )
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

        for hit in (q_hits + r_hits):
            meta = hit.get("metadata", {})
            question = meta.get("question", hit.get("text", ""))
            if not question:
                continue
            candidates.append({
                "question": question,
                "topic_id": meta.get("topic_id", ""),
                "topic_name": meta.get("topic_id", ""),
                "score": hit.get("score")
            })

        return candidates

    def _semantic_hits(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        results = []
        for c in candidates:
            score = c.get("score") or 0.0
            if score < self.SEMANTIC_THRESHOLD:
                continue
            results.append({
                "question":   c.get("question", ""),
                "topic_id":   c.get("topic_id", ""),
                "topic_name": c.get("topic_name", ""),
                "match_type": "semantic",
                "score":      round(float(score), 3),
            })
        return results

    # ------------------------------------------------------------------
    # MERGE + RANK
    # ------------------------------------------------------------------

    def _merge_and_rank(
        self,
        keyword_hits: List[Dict],
        semantic_hits: List[Dict]
    ) -> List[Dict]:
        """
        Merges the two lists by question text (case-insensitive dedup).
        When the same question appears in both lists:
          • mark match_type = "both"
          • score = weighted average (60% semantic, 40% keyword)
        Final list is sorted descending by score.
        """
        merged: Dict[str, Dict] = {}

        for hit in keyword_hits:
            key = hit["question"].lower()
            merged[key] = hit.copy()

        for hit in semantic_hits:
            key = hit["question"].lower()
            if key in merged:
                # Combine
                kw_score = merged[key]["score"]
                sem_score = hit["score"]
                merged[key]["score"] = 0.2 * kw_score + 0.8 * sem_score
                merged[key]["match_type"] = "both"
            else:
                merged[key] = hit.copy()

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return ranked
