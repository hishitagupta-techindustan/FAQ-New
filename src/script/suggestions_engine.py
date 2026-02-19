"""
SuggestionEngine — live "search-as-you-type" suggestions
Combines:
  • Keyword match  → scans MongoDB FAQ question strings (fast, exact)
  • Semantic match → ChromaDB vector similarity (slower, meaning-aware)
Results are merged, deduplicated, and ranked.
"""

import re
from typing import List, Dict, Any
from loguru import logger
from pymongo import MongoClient

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from retrieval.vectorstore import VectorStore


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
        self.vector_store = VectorStore(
            collection_name=settings.chroma_collection_name_questions,
            persist_directory=settings.chroma_persist_directory,
            embedding_model=settings.embedding_model
        )
        self.mongo = MongoClient(settings.mongodb_uri)
        self.db = self.mongo[settings.mongodb_db]
        self.collection = self.db["structured_faqs"]

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

        keyword_hits = self._keyword_search(query, product)
        semantic_hits = self._semantic_search(query, product)

        merged = self._merge_and_rank(keyword_hits, semantic_hits)
        return merged[: self.MAX_SUGGESTIONS]

    # ------------------------------------------------------------------
    # KEYWORD SEARCH  (MongoDB)
    # ------------------------------------------------------------------

    def _keyword_search(self, query: str, product: str) -> List[Dict[str, Any]]:
        """
        Case-insensitive regex match against every FAQ question string
        stored in MongoDB.  Splits the query into individual words and
        requires ALL of them to appear somewhere in the question.
        """
        words = [w for w in re.split(r"\W+", query) if len(w) > 1]

        if not words:
            return []

        # Build an AND of per-word regex filters on the nested faqs.question field
        word_filters = [
            {"faqs.question": {"$regex": word, "$options": "i"}}
            for word in words
        ]

        pipeline = [
            # Filter to correct product and topics that have at least one matching word
            {"$match": {"product": product, "$and": word_filters}},
            {"$limit": self.KEYWORD_FETCH_LIMIT},
            # Unwind so we can score individual FAQ entries
            {"$unwind": "$faqs"},
            {
                "$match": {
                    "$and": [
                        {"faqs.question": {"$regex": word, "$options": "i"}}
                        for word in words
                    ]
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "topic_id": 1,
                    "topic_name": 1,
                    "question": "$faqs.question",
                }
            },
        ]

        try:
            docs = list(self.collection.aggregate(pipeline))
        except Exception as e:
            logger.error(f"Keyword search error: {e}")
            return []

        results = []
        for doc in docs:
            q_lower = doc["question"].lower()
            q_lower_query = query.lower()

            # Simple relevance: boost if query appears as a contiguous substring
            if q_lower_query in q_lower:
                score = 0.80
            else:
                # Score by fraction of query words matched
                matched = sum(1 for w in words if w.lower() in q_lower)
                score = 0.50 + 0.30 * (matched / len(words))

            results.append({
                "question":   doc["question"],
                "topic_id":   doc.get("topic_id", ""),
                "topic_name": doc.get("topic_name", doc.get("topic_id", "")),
                "match_type": "keyword",
                "score":      round(score, 3),
            })

        return results

    # ------------------------------------------------------------------
    # SEMANTIC SEARCH  (ChromaDB)
    # ------------------------------------------------------------------

    def _semantic_search(self, query: str, product: str) -> List[Dict[str, Any]]:
        """
        Embeds the partial query and runs vector similarity search.
        Filters out low-confidence hits via SEMANTIC_THRESHOLD.
        """
        try:
            raw = self.vector_store.similarity_search(query, k=self.SEMANTIC_FETCH_LIMIT)
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []

        results = []
        for hit in raw:
            score = hit.get("score") or 0.0

            if score < self.SEMANTIC_THRESHOLD:
                continue

            meta = hit.get("metadata", {})
            topic_id = meta.get("topic_id", "")
            question = meta.get("question", hit.get("text", ""))

            if not question:
                continue

            # Fetch topic_name from MongoDB (cached lookup would be an optimization)
            topic_doc = self.collection.find_one(
                {"product": product, "topic_id": topic_id},
                {"_id": 0, "topic_name": 1}
            )
            topic_name = topic_doc.get("topic_name", topic_id) if topic_doc else topic_id

            results.append({
                "question":   question,
                "topic_id":   topic_id,
                "topic_name": topic_name,
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
                merged[key]["score"] = round(0.4 * kw_score + 0.6 * sem_score, 3)
                merged[key]["match_type"] = "both"
            else:
                merged[key] = hit.copy()

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return ranked
