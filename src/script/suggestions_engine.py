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


class SuggestionEngine:

    # Minimum characters before we bother searching
    MIN_QUERY_LENGTH = 1

    # How many raw hits to pull from each source before merging
    KEYWORD_FETCH_LIMIT = 10
    SEMANTIC_FETCH_LIMIT = 10

    # Final number of suggestions returned to the frontend
    MAX_SUGGESTIONS = 5

    # Minimum semantic similarity score to include a result
    SEMANTIC_THRESHOLD = 0.25

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

        candidates = self._semantic_candidates(query, product)
        keyword_hits = self._keyword_search(query, candidates)
        semantic_hits = self._semantic_hits(candidates)

        merged = self._merge_and_rank(keyword_hits, semantic_hits)
        return merged[: self.MAX_SUGGESTIONS]

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
                merged[key]["score"] = 0.4 * kw_score + 0.6 * sem_score
                merged[key]["match_type"] = "both"
            else:
                merged[key] = hit.copy()

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return ranked
