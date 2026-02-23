# """
# SuggestionEngine — live "search-as-you-type" suggestions
# Combines:
#   • Keyword match  → computed from retrieved vector candidates (no MongoDB scan)
#   • Semantic match → ChromaDB vector similarity (meaning-aware)
# Results are merged, deduplicated, and ranked.
# """

# import re
# from typing import List, Dict, Any
# from loguru import logger

# import sys
# from pathlib import Path

# sys.path.insert(0, str(Path(__file__).parent.parent))

# from config import settings
# from script.vectorstore import VectorStore
# from langchain_openai import ChatOpenAI
# from pydantic import BaseModel, Field


# class LLMSuggestionSet(BaseModel):
#     questions: List[str] = Field(default_factory=list, min_length=0, max_length=3)


# class SuggestionEngine:

#     # Minimum characters before we bother searching
#     MIN_QUERY_LENGTH = 5

#     # How many raw hits to pull from each source before merging
#     KEYWORD_FETCH_LIMIT = 3
#     SEMANTIC_FETCH_LIMIT = 3

#     # Final number of suggestions returned to the frontend
#     MAX_SUGGESTIONS = 5

#     # Minimum semantic similarity score to include a result
#     SEMANTIC_THRESHOLD = 0.65
#     # Minimum score required to return merged suggestions
#     MIN_SUGGESTION_SCORE = 0.60

#     def __init__(self):
#         self.question_store = VectorStore(
#             collection_name=settings.chroma_collection_name_questions,
#             persist_directory=settings.chroma_persist_directory,
#             embedding_model=settings.embedding_model
#         )
#         self.rag_store = VectorStore(
#             collection_name=settings.chroma_collection_name_rag,
#             persist_directory=settings.chroma_persist_directory,
#             embedding_model=settings.embedding_model
#         )
#         self.llm = ChatOpenAI(
#             model=settings.openai_model_large,
#             temperature=0.2,
#             api_key=settings.openai_api_key,
#             max_tokens=256
#         )
#         self.structured_llm = self.llm.with_structured_output(LLMSuggestionSet)

#     # ------------------------------------------------------------------
#     # PUBLIC
#     # ------------------------------------------------------------------

#     def suggest(self, partial_query: str, product: str) -> List[Dict[str, Any]]:
#         """
#         Main entry point.
#         Returns up to MAX_SUGGESTIONS question suggestions for the partial query.

#         Each suggestion dict:
#         {
#             "question":  str,          # full FAQ question text
#             "topic_id":  str,          # topic it belongs to
#             "topic_name": str,         # human-readable topic label
#             "match_type": str,         # "keyword" | "semantic" | "both"
#             "score":     float         # 0.0–1.0, higher = better match
#         }
#         """
#         query = partial_query.strip()

#         if len(query) < self.MIN_QUERY_LENGTH:
#             return []

#         # 1) Semantic candidates (fast retrieval), similarity search
#         candidates = self._semantic_candidates(query, product)

#         # 2) Keyword matches over retrieved candidates
#         keyword_hits = self._keyword_search(query, candidates)

#         # If no keyword hits, try a wider keyword scan over stored questions
#         if not keyword_hits:
#             keyword_hits = self._keyword_search_all(query, product)

#         # 3) Semantic suggestions
#         semantic_hits = self._semantic_hits(candidates)

#         merged = self._merge_and_rank(keyword_hits, semantic_hits)
#         filtered = [m for m in merged if m.get("score", 0.0) >= self.MIN_SUGGESTION_SCORE]

#         # Build a final list
#         final: List[Dict[str, Any]] = []
#         seen = set()

#         query_norm = self._normalize_question(query)

#         def add_items(items: List[Dict[str, Any]]):
#             for it in items:
#                 q = it.get("question", "").strip()
#                 if not q:
#                     continue
#                 if self._normalize_question(q) == query_norm:
#                     continue
#                 key = q.lower()
#                 if key in seen:
#                     continue
#                 seen.add(key)
#                 final.append(it)
#                 if len(final) >= self.MAX_SUGGESTIONS:
#                     break

#         add_items(filtered)

#         if len(final) < self.MAX_SUGGESTIONS and self._is_valid_query(query):
#             llm_suggestions = self._llm_context_fallback(query, product)
#             add_items(llm_suggestions)

#         return final[: self.MAX_SUGGESTIONS]

#     def _normalize_question(self, text: str) -> str:
#         t = text.strip().lower()
#         t = re.sub(r"[^a-z0-9\\s]", "", t)
#         t = re.sub(r"\\s+", " ", t)
#         return t

#     def _filter_items(self, query: str, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         query_norm = self._normalize_question(query)
#         filtered = []
#         seen = set()
#         for it in items:
#             q = it.get("question", "").strip()
#             if not q:
#                 continue
#             norm = self._normalize_question(q)
#             if not norm or norm == query_norm:
#                 continue
#             if norm in seen:
#                 continue
#             seen.add(norm)
#             filtered.append(it)
#         return filtered

#     def _is_valid_query(self, query: str) -> bool:
#         # Require at least 2 alphabetic characters to avoid nonsense input
#         return len(re.findall(r"[A-Za-z]", query)) >= 2


#     def _llm_context_fallback(
#         self,
#         query: str,
#         product: str
#     ) -> List[Dict[str, Any]]:
#         """
#         Generate 3 questions using LLM, grounded strictly in retrieved context,
#         so answers exist in the knowledge base.
#         """
#         try:
#             docs = self.question_store.similarity_search(
#                 query,
#                 k=2,
#                 filter_metadata={"product": product}
#             )
#         except Exception as e:
#             logger.error(f"RAG retrieval for LLM fallback failed: {e}")
#             return []

#         if not docs:
#             return []

#         # Build compact, answerable context from question + answer blocks
#         context_parts = []
#         for doc in docs:
#             meta = doc.get("metadata", {}) or {}
#             question = meta.get("question")
#             answer_blocks = meta.get("answer_blocks")
#             if not question or not answer_blocks:
#                 continue
#             if isinstance(answer_blocks, str):
#                 # Best-effort parse if stored as string
#                 try:
#                     import ast
#                     parsed = ast.literal_eval(answer_blocks)
#                     if isinstance(parsed, list):
#                         answer_blocks = parsed
#                 except Exception:
#                     answer_blocks = []
#             if not answer_blocks:
#                 continue
#             context_parts.append(
#                 f"Q: {question}\nA: {' '.join(answer_blocks)}"
#             )

#         if not context_parts:
#             return []

#         context_text = "\n\n".join(context_parts)
#         if not context_text.strip():
#             return []

#         try:
#             # Only force inclusion if the user phrase looks sensible and appears in context
#             phrase = query.strip()
#             # Heuristic: only force the user phrase if it looks domain-relevant
#             domain_terms = {
#                 "coverage", "claim", "repair", "warranty", "policy", "plan", "appliance",
#                 "service", "replacement", "damage", "accidental", "premium", "deductible",
#                 "eligible", "eligibility", "purchase", "buy", "price", "cost", "refund",
#                 "cancellation", "renewal", "inspection", "doorstep", "parts", "spare"
#             }
#             phrase_tokens = re.findall(r"[A-Za-z]{3,}", phrase.lower())
#             phrase_is_sensible = bool(phrase_tokens) and any(t in domain_terms for t in phrase_tokens)

#             include_clause = (
#                 "Include the exact user phrase in each question. "
#                 if phrase_is_sensible
#                 else "Do not include the user phrase if it is unrelated or nonsensical. "
#             )

#             # Keep context short to reduce latency
#             max_chars = 600
#             short_context = context_text[:max_chars]

#             prompt = (
#                 "Generate 3 concise FAQ-style questions for an insurance assistant. "
#                 + include_clause +
#                 "Prioritize the user's query; use context only to ground facts. "
#                 "You MUST only use information present in the context. "
#                 "Ensure each question is answerable from the context. "
#                 "Do not repeat the user's query verbatim. "
#                 "Keep each question under 12 words. "
#                 "Return only the questions, no extra text.\n\n"
#                 f"Product: {product}\n"
#                 f"User query (highest priority): {query}\n\n"
#                 f"Context (supporting only):\n{short_context}"
#             )
#             resp = self.structured_llm.invoke(prompt)
#             questions = resp.questions if resp else []
#             results = []
#             for q in questions:
#                 if not isinstance(q, str) or not q.strip():
#                     continue
#                 results.append({
#                     "question": q.strip(),
#                     "topic_id": "",
#                     "topic_name": "",
#                     "match_type": "llm_context",
#                     "score": 0.2
#                 })
#             return results
#         except Exception as e:
#             logger.error(f"LLM fallback suggestions error: {e}")
#             return []

#     # ------------------------------------------------------------------
#     # KEYWORD SEARCH  (Candidates)
#     # ------------------------------------------------------------------

#     def _keyword_search(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         """
#         Case-insensitive keyword scoring over retrieved candidates.
#         Splits the query into individual words and scores by coverage.
#         """
#         words = [w for w in re.split(r"\W+", query) if len(w) > 1]

#         if not words:
#             return []

#         results = []
#         for doc in candidates:
#             q_text = doc.get("question") or ""
#             q_lower = q_text.lower()
#             q_lower_query = query.lower()

#             # Simple relevance: boost if query appears as a contiguous substring
#             if q_lower_query in q_lower:
#                 score = 0.80
#             else:
#                 # Score by fraction of query words matched
#                 matched = sum(1 for w in words if w.lower() in q_lower)
#                 if matched == 0:
#                     continue
#                 score = 0.50 + 0.30 * (matched / len(words))

#             results.append({
#                 "question":   q_text,
#                 "topic_id":   doc.get("topic_id", ""),
#                 "topic_name": doc.get("topic_name", doc.get("topic_id", "")),
#                 "match_type": "keyword",
#                 "score":      round(score, 3),
#             })

#         return results[: self.KEYWORD_FETCH_LIMIT]

#     def _keyword_search_all(self, query: str, product: str) -> List[Dict[str, Any]]:
#         """
#         Wider keyword scan over stored questions (no cache).
#         Uses get_documents with a capped limit to avoid heavy scans.
#         """
#         try:
#             docs = self.question_store.get_documents(
#                 filter_metadata={"product": product, "question_type": "original"},
#                 limit=300
#             )
#         except Exception as e:
#             logger.error(f"Keyword scan get_documents failed: {e}")
#             return []

#         candidates = []
#         for doc in docs:
#             meta = doc.get("metadata", {}) or {}
#             question = meta.get("question")
#             if not question:
#                 continue
#             candidates.append({
#                 "question": question,
#                 "topic_id": meta.get("topic_id", ""),
#                 "topic_name": meta.get("topic_name", meta.get("topic_id", "")),
#                 "score": 0.0
#             })

#         return self._keyword_search(query, candidates)

#     # ------------------------------------------------------------------
#     # SEMANTIC SEARCH  (ChromaDB)
#     # ------------------------------------------------------------------

#     def _semantic_candidates(self, query: str, product: str) -> List[Dict[str, Any]]:
#         """
#         Retrieve semantic candidates from both question and RAG stores.
#         """
#         candidates: List[Dict[str, Any]] = []
#         try:
#             q_hits = self.question_store.similarity_search(
#                 query,
#                 k=self.SEMANTIC_FETCH_LIMIT,
#                 filter_metadata={"product": product}
#             )
#             r_hits = self.rag_store.similarity_search(
#                 query,
#                 k=self.SEMANTIC_FETCH_LIMIT,
#                 filter_metadata={"product": product}
#             )
#         except Exception as e:
#             logger.error(f"Semantic search error: {e}")
#             return []

#         for hit in (q_hits + r_hits):
#             meta = hit.get("metadata", {})
#             question = meta.get("question", hit.get("text", ""))
#             if not question:
#                 continue
#             candidates.append({
#                 "question": question,
#                 "topic_id": meta.get("topic_id", ""),
#                 "topic_name": meta.get("topic_id", ""),
#                 "score": hit.get("score")
#             })

#         return candidates

#     def _semantic_hits(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#         results = []
#         for c in candidates:
#             score = c.get("score") or 0.0
#             if score < self.SEMANTIC_THRESHOLD:
#                 continue
#             results.append({
#                 "question":   c.get("question", ""),
#                 "topic_id":   c.get("topic_id", ""),
#                 "topic_name": c.get("topic_name", ""),
#                 "match_type": "semantic",
#                 "score":      round(float(score), 3),
#             })
#         return results

#     # ------------------------------------------------------------------
#     # MERGE + RANK
#     # ------------------------------------------------------------------

#     def _merge_and_rank(
#         self,
#         keyword_hits: List[Dict],
#         semantic_hits: List[Dict]
#     ) -> List[Dict]:
#         """
#         Merges the two lists by question text (case-insensitive dedup).
#         When the same question appears in both lists:
#           • mark match_type = "both"
#           • score = weighted average (60% semantic, 40% keyword)
#         Final list is sorted descending by score.
#         """
#         merged: Dict[str, Dict] = {}

#         for hit in keyword_hits:
#             key = hit["question"].lower()
#             merged[key] = hit.copy()

#         for hit in semantic_hits:
#             key = hit["question"].lower()
#             if key in merged:
#                 # Combine
#                 kw_score = merged[key]["score"]
#                 sem_score = hit["score"]
#                 merged[key]["score"] = 0.2 * kw_score + 0.8 * sem_score
#                 merged[key]["match_type"] = "both"
#             else:
#                 merged[key] = hit.copy()

#         ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
#         return ranked


"""
SuggestionEngine v3
===================
Core fix: query is embedded ONCE, the vector is reused for every downstream call.

Flow:
  t=0   embed(query)                          ← single embedding call
  t=?   parallel:
          ├─ chroma question_store(vector)     ← no re-embedding
          ├─ chroma rag_store(vector)          ← no re-embedding
          └─ keyword scan (in-memory, CPU)     ← no embedding at all
  t=?   merge + rank (pure CPU)
  t=?   LLM fallback if needed
          └─ context = candidates already in hand, no extra chroma call
"""

import re
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional

from cachetools import TTLCache
from loguru import logger
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from script.vectorstore import VectorStore

# ---------------------------------------------------------------------------
# Compiled regexes (module-level, compiled once)
# ---------------------------------------------------------------------------
_RE_NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
_RE_MULTI_SPACE  = re.compile(r"\s+")
_RE_WORD_SPLIT   = re.compile(r"\W+")
_RE_ALPHA_3      = re.compile(r"[A-Za-z]{3,}")
_RE_ALPHA        = re.compile(r"[A-Za-z]")
_RE_JSON_FENCE   = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)

DOMAIN_TERMS = frozenset({
    "coverage", "claim", "repair", "warranty", "policy", "plan", "appliance",
    "service", "replacement", "damage", "accidental", "premium", "deductible",
    "eligible", "eligibility", "purchase", "buy", "price", "cost", "refund",
    "cancellation", "renewal", "inspection", "doorstep", "parts", "spare",
})

# Shared pool — threads are reused across requests
_POOL = ThreadPoolExecutor(max_workers=6, thread_name_prefix="suggest")


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8192)
def _normalize(text: str) -> str:
    t = _RE_NON_ALPHANUM.sub("", text.strip().lower())
    return _RE_MULTI_SPACE.sub(" ", t)


def _is_valid(query: str) -> bool:
    return len(_RE_ALPHA.findall(query)) >= 2


# ---------------------------------------------------------------------------
# Embedding cache  — avoids re-embedding the same query string
# Keyed by (query_text, model_name) so model changes don't serve stale vectors
# ---------------------------------------------------------------------------
class _EmbeddingCache:
    """
    Thread-safe LRU+TTL cache for query embeddings.
    Embedding the same partial query twice in one typing session is common
    (user types → deletes one char → retypes), so this saves real latency.
    """
    _TTL  = 120   # seconds — partial queries are ephemeral
    _SIZE = 2048  # slots

    def __init__(self):
        self._cache: TTLCache = TTLCache(maxsize=self._SIZE, ttl=self._TTL)
        self._lock  = threading.Lock()

    def get(self, key: str) -> Optional[List[float]]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, vector: List[float]) -> None:
        with self._lock:
            self._cache[key] = vector


_EMBED_CACHE = _EmbeddingCache()


# ---------------------------------------------------------------------------
# In-process question index (loaded once per product, refreshed on TTL)
# Eliminates the 300-doc get_documents I/O on every keyword-fallback miss.
# ---------------------------------------------------------------------------
class _QuestionIndex:
    INDEX_TTL = 600  # 10 minutes

    def __init__(self):
        self._lock   = threading.Lock()
        self._data:  Dict[str, List[Dict]] = {}
        self._expiry: Dict[str, float]      = {}

    def get(self, product: str, question_store: VectorStore) -> List[Dict]:
        now = time.monotonic()
        with self._lock:
            if product in self._data and now < self._expiry.get(product, 0):
                return self._data[product]

        try:
            docs = question_store.get_documents(
                filter_metadata={"product": product, "question_type": "original"},
                limit=300,
            )
        except Exception as e:
            logger.error(f"QuestionIndex load failed: {e}")
            return self._data.get(product, [])

        index = []
        for doc in docs:
            meta = doc.get("metadata", {}) or {}
            q    = meta.get("question")
            if not q:
                continue
            index.append({
                "question":   q,
                "topic_id":   meta.get("topic_id", ""),
                "topic_name": meta.get("topic_name", meta.get("topic_id", "")),
            })

        with self._lock:
            self._data[product]   = index
            self._expiry[product] = now + self.INDEX_TTL

        return index


_QUESTION_INDEX = _QuestionIndex()


# ---------------------------------------------------------------------------
# SuggestionEngine
# ---------------------------------------------------------------------------
class SuggestionEngine:

    MIN_QUERY_LENGTH     = 5
    KEYWORD_FETCH_LIMIT  = 3
    SEMANTIC_FETCH_LIMIT = 3   # per store; raise to 8–10 if keyword fallback triggers often
    MAX_SUGGESTIONS      = 5
    SEMANTIC_THRESHOLD   = 0.65
    MIN_SUGGESTION_SCORE = 0.60

    # Result cache
    _cache:      TTLCache       = TTLCache(maxsize=1024, ttl=300)
    _cache_lock: threading.Lock = threading.Lock()

    # In-flight dedup — identical concurrent requests share one Future
    _inflight:      Dict[Tuple, Future] = {}
    _inflight_lock: threading.Lock      = threading.Lock()

    def __init__(self):
        self.question_store = VectorStore(
            collection_name=settings.chroma_collection_name_questions,
            persist_directory=settings.chroma_persist_directory,
            embedding_model=settings.embedding_model,
        )
        self.rag_store = VectorStore(
            collection_name=settings.chroma_collection_name_rag,
            persist_directory=settings.chroma_persist_directory,
            embedding_model=settings.embedding_model,
        )

        # Dedicated embedder — used directly so we control when/how often
        # the query is embedded (exactly once per unique query string).
        self._embedder = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )

        # Fast/small LLM for fallback suggestions
        small_model = getattr(settings, "openai_model_small", "gpt-4o-mini")
        self._llm = ChatOpenAI(
            model=small_model,
            temperature=0.2,
            api_key=settings.openai_api_key,
            max_tokens=64,
        )

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def suggest(self, partial_query: str, product: str) -> List[Dict[str, Any]]:
        query = partial_query.strip()
        if len(query) < self.MIN_QUERY_LENGTH:
            return []

        key = (query, product)

        # 1. Exact cache hit → return immediately
        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        # 2. Prefix cache hit → return stale result instantly, warm real one
        prefix_result = self._prefix_cache_hit(query, product)

        # 3. Coalesce duplicate in-flight requests
        with self._inflight_lock:
            if key in self._inflight:
                fut = self._inflight[key]
            else:
                fut = _POOL.submit(self._compute, query, product)
                self._inflight[key] = fut

        if prefix_result is not None:
            _POOL.submit(self._settle, fut, key)   # cache in background
            return prefix_result

        return self._settle(fut, key)

    # ------------------------------------------------------------------
    # CORE PIPELINE
    # ------------------------------------------------------------------

    def _compute(self, query: str, product: str) -> List[Dict[str, Any]]:
        """
        Single-embedding pipeline.

        Step 1: embed query (once)
        Step 2: fan out all I/O in parallel using the pre-computed vector
        Step 3: CPU-only merge + rank
        Step 4: LLM fallback if still short (uses candidates already fetched)
        """

        # ── Step 1: embed once ──────────────────────────────────────────
        vector = self._embed(query)

        # ── Step 2: parallel I/O ────────────────────────────────────────
        # All three calls submit simultaneously; total wall time = slowest one.
        f_q  = _POOL.submit(self._chroma_by_vector, self.question_store, vector, product)
        f_r  = _POOL.submit(self._chroma_by_vector, self.rag_store,      vector, product)
        f_kw = _POOL.submit(self._keyword_all,       query, product)

        q_hits  = f_q.result()
        r_hits  = f_r.result()
        kw_hits = f_kw.result()

        candidates = q_hits + r_hits  # combined, deduplicated later

        # ── Step 3: build keyword hits from semantic candidates (CPU) ───
        kw_from_candidates = self._keyword_score(query, candidates)
        keyword_hits       = kw_from_candidates or kw_hits   # prefer candidate-derived

        semantic_hits = [
            {
                "question":   c["question"],
                "topic_id":   c["topic_id"],
                "topic_name": c["topic_name"],
                "match_type": "semantic",
                "score":      round(float(c["score"]), 3),
            }
            for c in candidates
            if (c.get("score") or 0.0) >= self.SEMANTIC_THRESHOLD
        ]

        merged   = self._merge_and_rank(keyword_hits, semantic_hits)
        filtered = [m for m in merged if m["score"] >= self.MIN_SUGGESTION_SCORE]
        final    = self._dedup(filtered, query)

        # ── Step 4: LLM fallback (candidates already in hand, no extra I/O) ──
        if len(final) < self.MAX_SUGGESTIONS and _is_valid(query):
            llm_hits = self._llm_fallback(query, product, candidates)
            final    = self._dedup(final + llm_hits, query)

        return final[:self.MAX_SUGGESTIONS]

    # ------------------------------------------------------------------
    # EMBEDDING  (single call, cached)
    # ------------------------------------------------------------------

    def _embed(self, query: str) -> List[float]:
        """
        Embed query exactly once per unique string.
        Cache key includes the model name so a model change invalidates entries.
        """
        cache_key = f"{settings.embedding_model}::{query}"
        cached = _EMBED_CACHE.get(cache_key)
        if cached is not None:
            return cached

        try:
            vector = self._embedder.embed_query(query)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return []

        _EMBED_CACHE.set(cache_key, vector)
        return vector

    # ------------------------------------------------------------------
    # CHROMA QUERIES  — accept pre-computed vector, never re-embed
    # ------------------------------------------------------------------

    def _chroma_by_vector(
        self,
        store: VectorStore,
        vector: List[float],
        product: str,
    ) -> List[Dict]:
        """
        Query ChromaDB with a pre-computed embedding vector.

        Uses the underlying chroma collection directly via
        `collection.query(query_embeddings=[vector], ...)` so the
        embedding model is never called again.

        Falls back to store.similarity_search (re-embeds) only if the
        collection attribute is unavailable — preserves compatibility.
        """
        if not vector:
            return []

        # Prefer direct vector query (no re-embedding)
        collection = getattr(store, "collection", None) or getattr(store, "_collection", None)

        if collection is not None:
            try:
                result = collection.query(
                    query_embeddings=[vector],
                    n_results=self.SEMANTIC_FETCH_LIMIT,
                    where={"product": product},
                    include=["metadatas", "distances"],
                )
                return self._parse_chroma_result(result)
            except Exception as e:
                logger.warning(f"Direct vector query failed, falling back: {e}")

        # Fallback: let the store embed the query itself
        # (happens at most once per store if direct access is unavailable)
        try:
            hits = store.similarity_search(
                query="",                          # unused when vector provided
                k=self.SEMANTIC_FETCH_LIMIT,
                filter_metadata={"product": product},
                embedding=vector,                  # pass pre-computed if store supports it
            )
            return self._parse_hits(hits)
        except Exception as e:
            logger.error(f"Fallback similarity_search failed: {e}")
            return []

    @staticmethod
    def _parse_chroma_result(result: Dict) -> List[Dict]:
        """Parse the raw dict returned by collection.query()."""
        out = []
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        for meta, dist in zip(metadatas, distances):
            if not meta:
                continue
            question = meta.get("question", "")
            if not question:
                continue
            # ChromaDB returns L2 distance; convert to a 0–1 similarity score.
            # For cosine distance (if configured): score = 1 - dist
            # For L2: score = 1 / (1 + dist)  — monotonically equivalent
            score = 1.0 / (1.0 + dist) if dist is not None else 0.0
            out.append({
                "question":   question,
                "topic_id":   meta.get("topic_id", ""),
                "topic_name": meta.get("topic_id", ""),
                "score":      score,
            })
        return out

    @staticmethod
    def _parse_hits(hits: List[Dict]) -> List[Dict]:
        """Parse hits returned by VectorStore.similarity_search()."""
        out = []
        for hit in hits:
            meta     = hit.get("metadata", {})
            question = meta.get("question") or hit.get("text", "")
            if not question:
                continue
            out.append({
                "question":   question,
                "topic_id":   meta.get("topic_id", ""),
                "topic_name": meta.get("topic_id", ""),
                "score":      hit.get("score", 0.0),
            })
        return out

    # ------------------------------------------------------------------
    # KEYWORD SEARCH
    # ------------------------------------------------------------------

    def _keyword_all(self, query: str, product: str) -> List[Dict]:
        index = _QUESTION_INDEX.get(product, self.question_store)
        return self._keyword_score(query, index)

    def _keyword_score(self, query: str, candidates: List[Dict]) -> List[Dict]:
        words = [w for w in _RE_WORD_SPLIT.split(query) if len(w) > 1]
        if not words:
            return []

        query_lower = query.lower()
        words_lower = [w.lower() for w in words]
        n_words     = len(words_lower)
        results     = []

        for doc in candidates:
            q_text  = doc.get("question") or ""
            q_lower = q_text.lower()

            if query_lower in q_lower:
                score = 0.80
            else:
                matched = sum(1 for w in words_lower if w in q_lower)
                if not matched:
                    continue
                score = 0.50 + 0.30 * (matched / n_words)

            results.append({
                "question":   q_text,
                "topic_id":   doc.get("topic_id", ""),
                "topic_name": doc.get("topic_name", doc.get("topic_id", "")),
                "match_type": "keyword",
                "score":      round(score, 3),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:self.KEYWORD_FETCH_LIMIT]

    # ------------------------------------------------------------------
    # LLM FALLBACK
    # ------------------------------------------------------------------

    def _llm_fallback(
        self, query: str, product: str, candidates: List[Dict]
    ) -> List[Dict]:
        """
        Uses candidates already retrieved — zero extra ChromaDB calls.
        Fast model + minimal tokens + plain JSON output.
        """
        context_lines = [
            f"- {c['question']}"
            for c in candidates[:3]
            if c.get("question")
        ]
        if not context_lines:
            return []

        context = "\n".join(context_lines)

        phrase_tokens      = _RE_ALPHA_3.findall(query.lower())
        phrase_is_sensible = any(t in DOMAIN_TERMS for t in phrase_tokens)
        include_clause     = "Include the user phrase in each question. " if phrase_is_sensible else ""

        prompt = (
            f"Insurance FAQ assistant. Product: {product}.\n"
            f'User typed: "{query}"\n'
            f"Related questions:\n{context}\n\n"
            + include_clause +
            "Generate 3 short FAQ questions (under 10 words each). "
            'Reply ONLY with a JSON array: ["Q1","Q2","Q3"]'
        )

        raw = ""
        try:
            resp = self._llm.invoke(prompt)
            raw  = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            m    = _RE_JSON_FENCE.search(raw)
            raw  = m.group(1).strip() if m else raw

            questions = json.loads(raw)
            if not isinstance(questions, list):
                raise ValueError("expected list")

            return [
                {"question": q.strip(), "topic_id": "", "topic_name": "",
                 "match_type": "llm_context", "score": 0.2}
                for q in questions[:3]
                if isinstance(q, str) and q.strip()
            ]
        except Exception as e:
            logger.warning(f"LLM fallback error: {e} | raw={raw!r}")
            return []

    # ------------------------------------------------------------------
    # MERGE / RANK / DEDUP
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_and_rank(
        keyword_hits: List[Dict], semantic_hits: List[Dict]
    ) -> List[Dict]:
        merged: Dict[str, Dict] = {}

        for hit in keyword_hits:
            merged[hit["question"].lower()] = hit.copy()

        for hit in semantic_hits:
            key = hit["question"].lower()
            if key in merged:
                merged[key]["score"]      = round(0.2 * merged[key]["score"] + 0.8 * hit["score"], 3)
                merged[key]["match_type"] = "both"
            else:
                merged[key] = hit.copy()

        return sorted(merged.values(), key=lambda x: x["score"], reverse=True)

    @staticmethod
    def _dedup(items: List[Dict], query: str) -> List[Dict]:
        query_norm = _normalize(query)
        seen: set  = set()
        out:  list = []
        for it in items:
            q = it.get("question", "").strip()
            if not q:
                continue
            if _normalize(q) == query_norm:
                continue
            k = q.lower()
            if k in seen:
                continue
            seen.add(k)
            out.append(it)
        return out

    # ------------------------------------------------------------------
    # CACHE HELPERS
    # ------------------------------------------------------------------

    def _settle(self, fut: Future, key: Tuple) -> List[Dict[str, Any]]:
        try:
            result = fut.result(timeout=5.0)
        except Exception as e:
            logger.error(f"suggest compute failed: {e}")
            result = []
        finally:
            with self._inflight_lock:
                self._inflight.pop(key, None)
        with self._cache_lock:
            self._cache[key] = result
        return result

    def _prefix_cache_hit(self, query: str, product: str) -> Optional[List[Dict[str, Any]]]:
        with self._cache_lock:
            for length in range(len(query) - 1, self.MIN_QUERY_LENGTH - 1, -1):
                hit = self._cache.get((query[:length], product))
                if hit is not None:
                    return hit
        return None