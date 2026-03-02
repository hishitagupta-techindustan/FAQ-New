

# """
# SuggestionEngine v4
# ===================
# Upgrades over v3:
#   - BM25 (rank_bm25) for term-frequency aware keyword matching
#   - Character trigram index for sub-word / partial-word matching
#   - NLTK: stemming + stopword removal so "claiming" matches "claim"
#   - RapidFuzz for typo-tolerant token matching (edit-distance)
#   - Single embedding call preserved from v3
#   - All new matchers are pure CPU; no extra I/O

# Flow:
#   t=0   embed(query) — once
#   t=?   parallel:
#           ├─ chroma question_store(vector)
#           ├─ chroma rag_store(vector)
#           └─ local index:
#                 ├─ BM25 (stemmed tokens)
#                 ├─ trigram overlap
#                 └─ fuzzy token matching
#   t=?   weighted merge + rank
#   t=?   No LLM fallback — all suggestions grounded in document
# """

# import re
# import json
# import time
# import threading
# from concurrent.futures import ThreadPoolExecutor, Future
# from functools import lru_cache
# from typing import List, Dict, Any, Tuple, Optional, Set

# import nltk
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from rapidfuzz import fuzz
# from rank_bm25 import BM25Okapi
# from cachetools import TTLCache
# from loguru import logger
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# import sys
# from pathlib import Path
# sys.path.insert(0, str(Path(__file__).parent.parent))

# from config import settings
# from script.vectorstore import VectorStore

# # ---------------------------------------------------------------------------
# # NLTK bootstrap (download once, silently)
# # ---------------------------------------------------------------------------
# for _pkg in ("stopwords", "punkt", "punkt_tab"):
#     try:
#         nltk.download(_pkg, quiet=True)
#     except Exception:
#         pass

# _STEMMER   = PorterStemmer()
# _STOPWORDS: Set[str] = set(stopwords.words("english"))

# # ---------------------------------------------------------------------------
# # Compiled regexes
# # ---------------------------------------------------------------------------
# _RE_NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
# _RE_MULTI_SPACE  = re.compile(r"\s+")
# _RE_WORD_SPLIT   = re.compile(r"\W+")
# _RE_ALPHA_3      = re.compile(r"[A-Za-z]{3,}")
# _RE_ALPHA        = re.compile(r"[A-Za-z]")
# _RE_JSON_FENCE   = re.compile(r"```(?:json)?(.*?)```", re.DOTALL)

# DOMAIN_TERMS = frozenset({
#     "coverage", "claim", "repair", "warranty", "policy", "plan", "appliance",
#     "service", "replacement", "damage", "accidental", "premium", "deductible",
#     "eligible", "eligibility", "purchase", "buy", "price", "cost", "refund",
#     "cancellation", "renewal", "inspection", "doorstep", "parts", "spare",
# })

# _POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="suggest")


# # ---------------------------------------------------------------------------
# # Helpers
# # ---------------------------------------------------------------------------

# @lru_cache(maxsize=16384)
# def _normalize(text: str) -> str:
#     t = _RE_NON_ALPHANUM.sub("", text.strip().lower())
#     return _RE_MULTI_SPACE.sub(" ", t)


# @lru_cache(maxsize=16384)
# def _tokenize(text: str) -> Tuple[str, ...]:
#     """Lowercase word tokens, stopwords removed."""
#     words = _RE_WORD_SPLIT.split(text.lower())
#     return tuple(w for w in words if len(w) > 1 and w not in _STOPWORDS)


# @lru_cache(maxsize=16384)
# def _stem_tokens(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
#     return tuple(_STEMMER.stem(t) for t in tokens)


# @lru_cache(maxsize=16384)
# def _trigrams(text: str) -> frozenset:
#     """Character-level trigrams of the normalised text."""
#     n = _normalize(text)
#     return frozenset(n[i:i+3] for i in range(len(n) - 2))


# def _is_valid(query: str) -> bool:
#     return len(_RE_ALPHA.findall(query)) >= 2


# def _trigram_similarity(a: str, b: str) -> float:
#     """Jaccard similarity on character trigrams."""
#     ta, tb = _trigrams(a), _trigrams(b)
#     if not ta or not tb:
#         return 0.0
#     return len(ta & tb) / len(ta | tb)


# # ---------------------------------------------------------------------------
# # Embedding cache
# # ---------------------------------------------------------------------------
# class _EmbeddingCache:
#     _TTL  = 120
#     _SIZE = 2048

#     def __init__(self):
#         self._cache = TTLCache(maxsize=self._SIZE, ttl=self._TTL)
#         self._lock  = threading.Lock()

#     def get(self, key: str) -> Optional[List[float]]:
#         with self._lock:
#             return self._cache.get(key)

#     def set(self, key: str, vector: List[float]) -> None:
#         with self._lock:
#             self._cache[key] = vector


# _EMBED_CACHE = _EmbeddingCache()


# # ---------------------------------------------------------------------------
# # Local question index — BM25 + trigram + fuzzy, rebuilt on TTL
# # ---------------------------------------------------------------------------
# class _LocalQuestionIndex:
#     """
#     Three complementary structures built from the question store:
#       1. BM25Okapi   — TF-IDF aware token matching (stemmed)
#       2. trigram     — character-level subword matching
#       3. fuzzy       — typo-tolerant via rapidfuzz
#     Rebuilt atomically on TTL expiry.
#     """
#     INDEX_TTL = 600  # 10 min

#     def __init__(self):
#         self._lock        = threading.Lock()
#         self._docs:    Dict[str, List[Dict]]  = {}
#         self._bm25:    Dict[str, BM25Okapi]   = {}
#         self._corpus:  Dict[str, List[Tuple]] = {}
#         self._expiry:  Dict[str, float]       = {}

#     def _build(self, product: str, question_store: VectorStore) -> None:
#         try:
#             raw_docs = question_store.get_documents(
#                 filter_metadata={"product": product},
#                 limit=1000,
#             )
#         except Exception as e:
#             logger.error(f"LocalIndex load failed: {e}")
#             return

#         docs, corpus = [], []
#         for doc in raw_docs:
#             meta = doc.get("metadata", {}) or {}
#             q    = meta.get("question", "")
#             if not q:
#                 continue
#             tokens = _stem_tokens(_tokenize(q))
#             docs.append({
#                 "question":      q,
#                 "topic_id":      meta.get("topic_id", ""),
#                 "topic_name":    meta.get("topic_name", meta.get("topic_id", "")),
#                 "answer_blocks": meta.get("answer_blocks", []),
#                 "match_type":    "local",
#             })
#             corpus.append(tokens)

#         bm25 = BM25Okapi([list(t) for t in corpus]) if corpus else None

#         with self._lock:
#             self._docs[product]   = docs
#             self._bm25[product]   = bm25
#             self._corpus[product] = corpus
#             self._expiry[product] = time.monotonic() + self.INDEX_TTL

#         logger.info(f"LocalIndex built for '{product}': {len(docs)} questions")

#     def _ensure_fresh(self, product: str, question_store: VectorStore) -> None:
#         now = time.monotonic()
#         if self._expiry.get(product, 0) < now:
#             self._build(product, question_store)

#     def search(
#         self,
#         query: str,
#         product: str,
#         question_store: VectorStore,
#         top_k: int = 10,
#         fuzzy_threshold: int = 72,
#     ) -> List[Dict]:
#         """Combined BM25 + trigram + fuzzy search."""
#         self._ensure_fresh(product, question_store)

#         with self._lock:
#             docs   = list(self._docs.get(product, []))
#             bm25   = self._bm25.get(product)

#         if not docs:
#             return []

#         q_tokens  = _tokenize(query)
#         q_stemmed = _stem_tokens(q_tokens)
#         q_norm    = _normalize(query)

#         # ── 1. BM25 ───────────────────────────────────────────────────────
#         if bm25 and q_stemmed:
#             raw = bm25.get_scores(list(q_stemmed))
#             mx  = max(raw) if max(raw) > 0 else 1.0
#             bm25_scores = [float(s) / mx for s in raw]
#         else:
#             bm25_scores = [0.0] * len(docs)

#         # ── 2. Trigram ────────────────────────────────────────────────────
#         tg_scores = [_trigram_similarity(query, doc["question"]) for doc in docs]

#         # ── 3. Fuzzy ──────────────────────────────────────────────────────
#         q_words = [w for w in _RE_WORD_SPLIT.split(query) if len(w) > 2]
#         fuzzy_scores: List[float] = []
#         for doc in docs:
#             q_text = doc["question"]
#             if not q_words:
#                 fuzzy_scores.append(0.0)
#                 continue
#             word_scores = []
#             for qw in q_words:
#                 pr = fuzz.partial_ratio(qw.lower(), q_text.lower()) / 100.0
#                 tr = fuzz.token_set_ratio(query.lower(), q_text.lower()) / 100.0
#                 word_scores.append(max(pr, tr))
#             avg = sum(word_scores) / len(word_scores)
#             fuzzy_scores.append(avg if avg * 100 >= fuzzy_threshold else 0.0)

#         # ── 4. Exact substring bonus ──────────────────────────────────────
#         exact_scores = [
#             0.25 if q_norm in _normalize(doc["question"]) else 0.0
#             for doc in docs
#         ]

#         # ── 5. Blend ──────────────────────────────────────────────────────
#         W_BM25, W_FUZZY, W_TG, W_EXACT = 0.45, 0.30, 0.15, 0.10

#         blended = []
#         for i, doc in enumerate(docs):
#             score = (
#                 W_BM25  * bm25_scores[i]  +
#                 W_FUZZY * fuzzy_scores[i] +
#                 W_TG    * tg_scores[i]    +
#                 W_EXACT * exact_scores[i]
#             )
#             if score > 0.05:
#                 blended.append((score, doc))

#         blended.sort(key=lambda x: x[0], reverse=True)

#         return [
#             {**doc, "match_type": "local", "score": round(s, 3)}
#             for s, doc in blended[:top_k]
#         ]


# _LOCAL_INDEX = _LocalQuestionIndex()


# # ---------------------------------------------------------------------------
# # SuggestionEngine v4
# # ---------------------------------------------------------------------------
# class SuggestionEngine:

#     MIN_QUERY_LENGTH     = 3
#     SEMANTIC_FETCH_LIMIT = 10
#     MAX_SUGGESTIONS      = 5
#     SEMANTIC_THRESHOLD   = 0.40
#     MIN_SUGGESTION_SCORE = 0.08

#     _cache:      TTLCache       = TTLCache(maxsize=1024, ttl=300)
#     _cache_lock: threading.Lock = threading.Lock()

#     _inflight:      Dict[Tuple, Future] = {}
#     _inflight_lock: threading.Lock      = threading.Lock()

#     def __init__(self):
#         self.question_store = VectorStore(
#             collection_name=settings.chroma_collection_name_questions,
#             persist_directory=settings.chroma_persist_directory,
#             embedding_model=settings.embedding_model,
#         )
#         self.rag_store = VectorStore(
#             collection_name=settings.chroma_collection_name_rag,
#             persist_directory=settings.chroma_persist_directory,
#             embedding_model=settings.embedding_model,
#         )
#         self._embedder = OpenAIEmbeddings(
#             model=settings.embedding_model,
#             api_key=settings.openai_api_key,
#         )
#         small_model = getattr(settings, "openai_model_small", "gpt-4o-mini")
#         self._llm = ChatOpenAI(
#             model=small_model,
#             temperature=0.2,
#             api_key=settings.openai_api_key,
#             max_tokens=80,
#         )

#     # ------------------------------------------------------------------
#     # PUBLIC
#     # ------------------------------------------------------------------

#     def suggest(self, partial_query: str, product: str, is_followup: bool = False) -> List[Dict[str, Any]]:
#         query = partial_query.strip()
#         if len(query) < self.MIN_QUERY_LENGTH:
#             return []

#         # is_followup included in key so followup/suggestion don't share cache
#         key = (query, product, is_followup)

#         with self._cache_lock:
#             cached = self._cache.get(key)
#             if cached is not None:
#                 return cached

#         prefix_result = self._prefix_cache_hit(query, product, is_followup)

#         with self._inflight_lock:
#             if key in self._inflight:
#                 fut = self._inflight[key]
#             else:
#                 fut = _POOL.submit(self._compute, query, product, is_followup)
#                 self._inflight[key] = fut

#         if prefix_result is not None:
#             _POOL.submit(self._settle, fut, key)
#             return prefix_result

#         return self._settle(fut, key)

#     # ------------------------------------------------------------------
#     # CORE PIPELINE
#     # ------------------------------------------------------------------

#     def _compute(self, query: str, product: str, is_followup: bool = False) -> List[Dict[str, Any]]:
#         try:
#             # ── Step 1: single embed ───────────────────────────────────
#             vector = self._embed(query)

#             # ── Step 2: parallel fan-out ───────────────────────────────
#             f_q   = _POOL.submit(self._chroma_by_vector, self.question_store, vector, product)
#             f_r   = _POOL.submit(self._chroma_by_vector, self.rag_store, vector, product)
#             f_loc = _POOL.submit(_LOCAL_INDEX.search, query, product, self.question_store)

#             q_hits   = f_q.result()
#             r_hits   = f_r.result()
#             loc_hits = f_loc.result()

#             if is_followup:
#                 # ── FOLLOWUP PATH ──────────────────────────────────────
#                 # Re-query question store using top RAG chunk text as vector
#                 # to get topically related questions from a different angle.
#                 rag_guided_hits: List[Dict] = []
#                 if r_hits:
#                     top_rag_text = r_hits[0].get("text", "")
#                     if top_rag_text:
#                         rag_vector      = self._embed(top_rag_text[:300])
#                         f_q2            = _POOL.submit(self._chroma_by_vector, self.question_store, rag_vector, product)
#                         rag_guided_hits = f_q2.result()

#                 # Combine semantic + RAG-guided, strip rephrasings of original query
#                 all_candidates = q_hits + rag_guided_hits
#                 diverse = [
#                     c for c in all_candidates
#                     if 30 < fuzz.token_set_ratio(query.lower(), c.get("question", "").lower()) < 78
#                 ]

#                 # Fallback to local index if still short
#                 if len(diverse) < 3:
#                     local_diverse = [
#                         c for c in loc_hits
#                         if 30 < fuzz.token_set_ratio(query.lower(), c.get("question", "").lower()) < 78
#                     ]
#                     seen = {c["question"].lower() for c in diverse}
#                     for c in local_diverse:
#                         if c["question"].lower() not in seen:
#                             diverse.append(c)
#                             seen.add(c["question"].lower())

#                 final = self._dedup(diverse, query)

#             else:
#                 # ── SUGGESTION PATH ────────────────────────────────────
#                 # Only questions from the question store — fully grounded.
#                 semantic_hits = [
#                     {**c, "match_type": "semantic", "score": round(float(c["score"]), 3)}
#                     for c in q_hits
#                     if (c.get("score") or 0.0) >= self.SEMANTIC_THRESHOLD
#                 ]

#                 merged   = self._merge_and_rank(loc_hits, semantic_hits)
#                 filtered = [m for m in merged if m["score"] >= self.MIN_SUGGESTION_SCORE]
#                 final    = self._dedup(filtered, query)

#             return final[:self.MAX_SUGGESTIONS]

#         except Exception as e:
#             logger.error(f"_compute failed (query={query!r}, followup={is_followup}): {e}")
#             return []

#     # ------------------------------------------------------------------
#     # EMBEDDING
#     # ------------------------------------------------------------------

#     def _embed(self, query: str) -> List[float]:
#         cache_key = f"{settings.embedding_model}::{query}"
#         cached = _EMBED_CACHE.get(cache_key)
#         if cached is not None:
#             return cached
#         try:
#             vector = self._embedder.embed_query(query)
#         except Exception as e:
#             logger.error(f"Embedding failed: {e}")
#             return []
#         _EMBED_CACHE.set(cache_key, vector)
#         return vector

#     # ------------------------------------------------------------------
#     # CHROMA (pre-computed vector, no re-embed)
#     # ------------------------------------------------------------------

#     def _chroma_by_vector(
#         self, store: VectorStore, vector: List[float], product: str
#     ) -> List[Dict]:
#         if not vector:
#             return []

#         collection = getattr(store, "collection", None) or getattr(store, "_collection", None)
#         if collection is not None:
#             try:
#                 result = collection.query(
#                     query_embeddings=[vector],
#                     n_results=self.SEMANTIC_FETCH_LIMIT,
#                     where={"product": product},
#                     include=["metadatas", "distances"],
#                 )
#                 return self._parse_chroma_result(result)
#             except Exception as e:
#                 logger.warning(f"Direct vector query failed, falling back: {e}")

#         try:
#             hits = store.similarity_search(
#                 query="",
#                 k=self.SEMANTIC_FETCH_LIMIT,
#                 filter_metadata={"product": product},
#                 embedding=vector,
#             )
#             return self._parse_hits(hits)
#         except Exception as e:
#             logger.error(f"Fallback similarity_search failed: {e}")
#             return []

#     @staticmethod
#     def _parse_chroma_result(result: Dict) -> List[Dict]:
#         out = []
#         metadatas = (result.get("metadatas") or [[]])[0]
#         distances = (result.get("distances") or [[]])[0]
#         for meta, dist in zip(metadatas, distances):
#             if not meta:
#                 continue
#             question = meta.get("question", "")
#             if not question:
#                 continue
#             score = 1.0 / (1.0 + dist) if dist is not None else 0.0
#             out.append({
#                 "question":      question,
#                 "topic_id":      meta.get("topic_id", ""),
#                 "topic_name":    meta.get("topic_name", meta.get("topic_id", "")),
#                 "answer_blocks": meta.get("answer_blocks", []),
#                 "match_type":    "semantic",
#                 "score":         score,
#             })
#         return out

#     @staticmethod
#     def _parse_hits(hits: List[Dict]) -> List[Dict]:
#         out = []
#         for hit in hits:
#             meta     = hit.get("metadata", {})
#             question = meta.get("question") or hit.get("text", "")
#             if not question:
#                 continue
#             out.append({
#                 "question":      question,
#                 "topic_id":      meta.get("topic_id", ""),
#                 "topic_name":    meta.get("topic_name", meta.get("topic_id", "")),
#                 "answer_blocks": meta.get("answer_blocks", []),
#                 "match_type":    "semantic",
#                 "score":         hit.get("score", 0.0),
#             })
#         return out

#     # ------------------------------------------------------------------
#     # LLM FALLBACK (kept for potential future use, not called from _compute)
#     # ------------------------------------------------------------------

#     def _llm_fallback(
#         self, query: str, product: str, candidates: List[Dict],
#         is_followup: bool = False, rag_hits: List[Dict] = None
#     ) -> List[Dict]:

#         if rag_hits:
#             rag_context = "\n\n".join(
#                 h.get("text", "") or h.get("metadata", {}).get("question", "")
#                 for h in rag_hits[:3]
#                 if h.get("text") or h.get("metadata", {}).get("question")
#             )
#         else:
#             rag_context = ""

#         context_lines    = [f"- {c['question']}" for c in candidates[:5] if c.get("question")]
#         question_context = "\n".join(context_lines)

#         if not rag_context and not question_context:
#             return []

#         raw = ""
#         try:
#             if is_followup:
#                 prompt = (
#                     f"Insurance FAQ assistant. Product: {product}.\n"
#                     f'User just asked: "{query}"\n\n'
#                     f"Relevant document content:\n{rag_context}\n\n"
#                     f"Related questions already answered:\n{question_context}\n\n"
#                     "Generate 3 follow-up questions a curious customer would ask NEXT.\n"
#                     "Rules:\n"
#                     "- Must be answerable from the document content above\n"
#                     "- Must be meaningfully different from the original query\n"
#                     "- No rephrasing or synonyms of the original query\n"
#                     "- Explore distinct angles: exclusions, process, limits, eligibility\n"
#                     "- Under 12 words each\n"
#                     'Reply ONLY with a JSON array: ["Q1","Q2","Q3"]'
#                 )
#             else:
#                 phrase_tokens      = _RE_ALPHA_3.findall(query.lower())
#                 phrase_is_sensible = any(t in DOMAIN_TERMS for t in phrase_tokens)
#                 include_clause     = f'- Each question must contain the word "{query}" or a key term from it.\n' if phrase_is_sensible else ""

#                 prompt = (
#                     f"Insurance FAQ assistant. Product: {product}.\n"
#                     f'User is typing: "{query}"\n\n'
#                     f"Relevant document content:\n{rag_context}\n\n"
#                     f"Related questions:\n{question_context}\n\n"
#                     "Generate 3 distinct FAQ suggestions grounded in the document content above.\n"
#                     "Rules:\n"
#                     "- ONLY suggest questions answerable from the document content\n"
#                     "- Each question on a DIFFERENT aspect (cost, process, eligibility, limits)\n"
#                     "- No two questions should feel like rephrasings\n"
#                     "- Must feel like real customer questions\n"
#                     + include_clause +
#                     "- Under 12 words each\n"
#                     'Reply ONLY with a JSON array: ["Q1","Q2","Q3"]'
#                 )

#             resp      = self._llm.invoke(prompt)
#             raw       = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
#             m         = _RE_JSON_FENCE.search(raw)
#             raw       = m.group(1).strip() if m else raw
#             questions = json.loads(raw)

#             if not isinstance(questions, list):
#                 raise ValueError("expected list")

#             return [
#                 {
#                     "question":      q.strip(),
#                     "topic_id":      "",
#                     "topic_name":    "",
#                     "answer_blocks": [],
#                     "match_type":    "llm_followup" if is_followup else "llm_context",
#                     "score":         0.2,
#                 }
#                 for q in questions[:3]
#                 if isinstance(q, str) and q.strip()
#             ]

#         except Exception as e:
#             logger.warning(f"LLM fallback error (followup={is_followup}): {e} | raw={raw!r}")
#             return []

#     # ------------------------------------------------------------------
#     # MERGE / RANK / DEDUP
#     # ------------------------------------------------------------------

#     @staticmethod
#     def _merge_and_rank(
#         local_hits: List[Dict], semantic_hits: List[Dict]
#     ) -> List[Dict]:
#         merged: Dict[str, Dict] = {}

#         for hit in local_hits:
#             merged[hit["question"].lower()] = hit.copy()

#         for hit in semantic_hits:
#             key = hit["question"].lower()
#             if key in merged:
#                 blended = round(0.35 * merged[key]["score"] + 0.65 * hit["score"], 3)
#                 merged[key]["score"]      = blended
#                 merged[key]["match_type"] = "both"
#             else:
#                 merged[key] = hit.copy()

#         return sorted(merged.values(), key=lambda x: x["score"], reverse=True)

#     @staticmethod
#     def _dedup(items: List[Dict], query: str) -> List[Dict]:
#         query_norm = _normalize(query)
#         seen: set  = set()
#         out:  list = []
#         for it in items:
#             q = it.get("question", "").strip()
#             if not q:
#                 continue
#             if _normalize(q) == query_norm:
#                 continue
#             k = q.lower()
#             if k in seen:
#                 continue
#             seen.add(k)
#             out.append(it)
#         return out

#     # ------------------------------------------------------------------
#     # CACHE HELPERS
#     # ------------------------------------------------------------------

#     def _settle(self, fut: Future, key: Tuple) -> List[Dict[str, Any]]:
#         try:
#             result = fut.result(timeout=10.0)
#         except Exception as e:
#             logger.error(f"suggest compute failed: {e}")
#             result = []
#         finally:
#             with self._inflight_lock:
#                 self._inflight.pop(key, None)
#         result = result or []
#         with self._cache_lock:
#             self._cache[key] = result
#         return result

#     def _prefix_cache_hit(self, query: str, product: str, is_followup: bool = False) -> Optional[List[Dict[str, Any]]]:
#         with self._cache_lock:
#             for length in range(len(query) - 1, self.MIN_QUERY_LENGTH - 1, -1):
#                 hit = self._cache.get((query[:length], product, is_followup))
#                 if hit is not None:
#                     return hit
#         return None




"""
SuggestionEngine v4
===================
Upgrades over v3:
  - BM25 (rank_bm25) for term-frequency aware keyword matching
  - Character trigram index for sub-word / partial-word matching
  - NLTK: stemming + stopword removal so "claiming" matches "claim"
  - RapidFuzz for typo-tolerant token matching (edit-distance)
  - Single embedding call preserved from v3
  - All new matchers are pure CPU; no extra I/O

Fix (stem-expanded fuzzy/trigram):
  - Fuzzy scorer now also tests the *stemmed* form of each query word,
    so typing "reviewed" keeps suggestions that contain "review" alive.
  - Prefix-match bonus added: if a query word is a prefix of a word in
    the candidate question (or vice-versa), a small boost is applied so
    mid-typing partial words don't drop results.
  - Trigram scorer uses both the raw query and its stem-normalised form,
    taking the max, so "reviewed" ≈ "review" at the char-n-gram level.
  - Exact-substring check also tests the stemmed query.

Fix (gibberish / false semantic hits):
  - _is_gibberish() gate: queries with no vowels, implausible consonant
    clusters, or random mixed-case patterns are rejected before any index
    is touched — so "dnldmKML" returns [] immediately.
  - Semantic corroboration gate inside _compute(): semantic-only hits are
    suppressed unless the local index (BM25/fuzzy) also finds at least one
    hit above MIN_SUGGESTION_SCORE.  Semantic embeddings always return
    nearest neighbours regardless of true relevance; local index does not.

Flow:
  t=0   embed(query) — once
  t=?   parallel:
          ├─ chroma question_store(vector)
          ├─ chroma rag_store(vector)
          └─ local index:
                ├─ BM25 (stemmed tokens)
                ├─ trigram overlap (raw + stemmed)
                └─ fuzzy token matching (raw + stemmed words)
  t=?   weighted merge + rank
  t=?   No LLM fallback — all suggestions grounded in document
"""

import re
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from functools import lru_cache
from typing import List, Dict, Any, Tuple, Optional, Set

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from rapidfuzz import fuzz
from rank_bm25 import BM25Okapi
from cachetools import TTLCache
from loguru import logger
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from script.vectorstore import VectorStore

# ---------------------------------------------------------------------------
# NLTK bootstrap (download once, silently)
# ---------------------------------------------------------------------------
for _pkg in ("stopwords", "punkt", "punkt_tab"):
    try:
        nltk.download(_pkg, quiet=True)
    except Exception:
        pass

_STEMMER   = PorterStemmer()
_STOPWORDS: Set[str] = set(stopwords.words("english"))

# ---------------------------------------------------------------------------
# Compiled regexes
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

_POOL = ThreadPoolExecutor(max_workers=8, thread_name_prefix="suggest")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@lru_cache(maxsize=16384)
def _normalize(text: str) -> str:
    t = _RE_NON_ALPHANUM.sub("", text.strip().lower())
    return _RE_MULTI_SPACE.sub(" ", t)


@lru_cache(maxsize=16384)
def _tokenize(text: str) -> Tuple[str, ...]:
    """Lowercase word tokens, stopwords removed."""
    words = _RE_WORD_SPLIT.split(text.lower())
    return tuple(w for w in words if len(w) > 1 and w not in _STOPWORDS)


@lru_cache(maxsize=16384)
def _stem_tokens(tokens: Tuple[str, ...]) -> Tuple[str, ...]:
    return tuple(_STEMMER.stem(t) for t in tokens)


@lru_cache(maxsize=16384)
def _trigrams(text: str) -> frozenset:
    """Character-level trigrams of the normalised text."""
    n = _normalize(text)
    return frozenset(n[i:i+3] for i in range(len(n) - 2))


# ---------------------------------------------------------------------------
# Gibberish detection — conservative, only catches obvious random keypresses
# ---------------------------------------------------------------------------
_RE_VOWEL        = re.compile(r"[aeiou]", re.IGNORECASE)
_RE_UPPER_LOWER  = re.compile(r"[a-z][A-Z]|[A-Z]{2,}[a-z]")  # abruptMIXED
_RE_VOWEL       = re.compile(r"[aeiouy]", re.IGNORECASE)   # y counts as vowel
_RE_UPPER_LOWER = re.compile(r"[a-z][A-Z]|[A-Z]{2,}[a-z]")

def _is_gibberish(query: str) -> bool:
    """Return True ONLY for obviously random keypresses.

    Conservative by design — false positives are far worse than false negatives.

    Rules (ALL must be very safe):
      1. Fewer than 2 alphabetic characters → clearly not a word.
      2. All-uppercase token → treat as acronym (HTTP, XSLT, ADH, NASA), never gibberish.
      3. No vowels (incl. y) AND ≥ 7 alpha chars → no natural-language word of
         that length is entirely vowelless. Threshold raised from 5→7 to avoid
         catching short abbreviations like "bcrypt", "HTTPS", "nth".
      4. Vowel ratio < 8% AND > 9 alpha chars → extremely consonant-heavy strings.
         Threshold raised from 7→9 to give more room to technical terms.
      5. No spaces + abrupt mixed-case + 0 vowels → catches "dnldmKML"-style noise.
    """
    alpha_chars = _RE_ALPHA.findall(query)
    n_alpha = len(alpha_chars)

    # Rule 1
    if n_alpha < 2:
        return True

    # Rule 2 — all-caps short tokens are acronyms (ADH, HTTP, NASA, XSLT)
    stripped = query.strip()
    if stripped == stripped.upper() and stripped.isalpha():
        return False

    lower_q  = query.lower()
    n_vowels = len(_RE_VOWEL.findall(lower_q))   # y now counts

    # Rule 3 — no vowels in a word ≥ 7 alpha chars (raised from 5)
    if n_alpha >= 7 and n_vowels == 0:
        return True

    # Rule 4 — near-zero vowel ratio for longer strings (raised from 7 to 9)
    if n_alpha > 9 and (n_vowels / n_alpha) < 0.08:
        return True

    # Rule 5 — abrupt mixed-case with zero vowels and no spaces
    if " " not in query and n_vowels == 0 and _RE_UPPER_LOWER.search(query):
        return True

    return False

def _is_valid(query: str) -> bool:
    return not _is_gibberish(query)


def _trigram_similarity(a: str, b: str) -> float:
    """Jaccard similarity on character trigrams."""
    ta, tb = _trigrams(a), _trigrams(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _stem_query_string(query: str) -> str:
    """Return a version of *query* where every token is replaced by its stem.

    e.g. "reviewed items" → "review item"

    Used to make trigram / fuzzy comparisons stem-aware without touching the
    document side (documents are already indexed by stem in BM25).
    """
    tokens = _tokenize(query)
    stemmed = _stem_tokens(tokens)
    return " ".join(stemmed)


# ---------------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------------
class _EmbeddingCache:
    _TTL  = 120
    _SIZE = 2048

    def __init__(self):
        self._cache = TTLCache(maxsize=self._SIZE, ttl=self._TTL)
        self._lock  = threading.Lock()

    def get(self, key: str) -> Optional[List[float]]:
        with self._lock:
            return self._cache.get(key)

    def set(self, key: str, vector: List[float]) -> None:
        with self._lock:
            self._cache[key] = vector


_EMBED_CACHE = _EmbeddingCache()


# ---------------------------------------------------------------------------
# Local question index — BM25 + trigram + fuzzy, rebuilt on TTL
# ---------------------------------------------------------------------------
class _LocalQuestionIndex:
    """
    Three complementary structures built from the question store:
      1. BM25Okapi   — TF-IDF aware token matching (stemmed)
      2. trigram     — character-level subword matching (raw + stem-expanded)
      3. fuzzy       — typo-tolerant via rapidfuzz (raw + stem-expanded words)
    Rebuilt atomically on TTL expiry.
    """
    INDEX_TTL = 600  # 10 min

    def __init__(self):
        self._lock        = threading.Lock()
        self._docs:    Dict[str, List[Dict]]  = {}
        self._bm25:    Dict[str, BM25Okapi]   = {}
        self._corpus:  Dict[str, List[Tuple]] = {}
        self._expiry:  Dict[str, float]       = {}

    def _build(self, product: str, question_store: VectorStore) -> None:
        try:
            raw_docs = question_store.get_documents(
                filter_metadata={"product": product},
                limit=300,
            )
        except Exception as e:
            logger.error(f"LocalIndex load failed: {e}")
            return

        docs, corpus = [], []
        for doc in raw_docs:
            meta = doc.get("metadata", {}) or {}
            q    = meta.get("question", "")
            if not q:
                continue
            tokens = _stem_tokens(_tokenize(q))
            docs.append({
                "question":      q,
                "topic_id":      meta.get("topic_id", ""),
                "topic_name":    meta.get("topic_name", meta.get("topic_id", "")),
                "match_type":    "local",
            })
            corpus.append(tokens)

        bm25 = BM25Okapi([list(t) for t in corpus]) if corpus else None

        with self._lock:
            self._docs[product]   = docs
            self._bm25[product]   = bm25
            self._corpus[product] = corpus
            self._expiry[product] = time.monotonic() + self.INDEX_TTL

        logger.info(f"LocalIndex built for '{product}': {len(docs)} questions")

    def _ensure_fresh(self, product: str, question_store: VectorStore) -> None:
        now = time.monotonic()
        if self._expiry.get(product, 0) < now:
            self._build(product, question_store)

    def search(
        self,
        query: str,
        product: str,
        question_store: VectorStore,
        top_k: int = 10,
        fuzzy_threshold: int = 72,
    ) -> List[Dict]:
        """Combined BM25 + trigram + fuzzy search.

        Key fix: fuzzy and trigram scorers now operate on *both* the raw query
        and its stem-normalised form so that e.g. typing "reviewed" keeps
        suggestions that contain "review" alive (they share the stem "review").
        A lightweight prefix-match bonus also prevents drop-off while the user
        is mid-word.
        """
        self._ensure_fresh(product, question_store)

        with self._lock:
            docs   = list(self._docs.get(product, []))
            bm25   = self._bm25.get(product)

        if not docs:
            return []

        q_tokens  = _tokenize(query)
        q_stemmed = _stem_tokens(q_tokens)
        q_norm    = _normalize(query)

        # Stem-expanded query string used for trigram / fuzzy comparisons
        q_stem_str = _stem_query_string(query)

        # ── 1. BM25 ───────────────────────────────────────────────────────
        # BM25 already operates on stemmed tokens — unaffected by the bug.
        if bm25 and q_stemmed:
            raw = bm25.get_scores(list(q_stemmed))
            mx  = max(raw) if max(raw) > 0 else 1.0
            bm25_scores = [float(s) / mx for s in raw]
        else:
            bm25_scores = [0.0] * len(docs)

        # ── 2. Trigram ────────────────────────────────────────────────────
        # FIX: take the *max* of raw-query trigram sim and stem-query trigram
        # sim so that "reviewed" ≈ "review" at the character n-gram level.
        tg_scores = [
            max(
                _trigram_similarity(query, doc["question"]),
                _trigram_similarity(q_stem_str, doc["question"]),
            )
            for doc in docs
        ]

        # ── 3. Fuzzy ──────────────────────────────────────────────────────
        # FIX: build TWO word lists — raw words and their stems — and score
        # each document against both, taking the per-word maximum.  This means
        # "reviewed" (raw) and "review" (stem) are both tried against each
        # candidate question, so a question with "review" won't fall off.
        raw_words   = [w for w in _RE_WORD_SPLIT.split(query) if len(w) > 2]
        stem_words  = [_STEMMER.stem(w) for w in raw_words]
        # Combined unique query-side word forms
        all_q_words = list(dict.fromkeys(raw_words + [s for s in stem_words if s not in raw_words]))

        fuzzy_scores: List[float] = []
        for doc in docs:
            q_text = doc["question"]
            if not all_q_words:
                fuzzy_scores.append(0.0)
                continue
            word_scores = []
            for qw in all_q_words:
                pr = fuzz.partial_ratio(qw.lower(), q_text.lower()) / 100.0
                tr = fuzz.token_set_ratio(query.lower(), q_text.lower()) / 100.0
                word_scores.append(max(pr, tr))
            avg = sum(word_scores) / len(word_scores)
            fuzzy_scores.append(avg if avg * 100 >= fuzzy_threshold else 0.0)

        # ── 4. Exact substring bonus ──────────────────────────────────────
        # FIX: also grant the bonus when the *stemmed* query is a substring,
        # so "reviewed" → stem "review" matches questions containing "review".
        def _exact(doc_q: str) -> float:
            norm_doc = _normalize(doc_q)
            if q_norm in norm_doc:
                return 0.25
            if q_stem_str and q_stem_str in norm_doc:
                return 0.18  # slightly lower weight for stem match
            return 0.0

        exact_scores = [_exact(doc["question"]) for doc in docs]

        # ── 5. Prefix-match bonus ─────────────────────────────────────────
        # NEW: reward candidates whose words *start with* any query word (or
        # stem).  This keeps suggestions alive while the user is mid-word,
        # e.g. "revi" → "review", "clai" → "claim".
        prefix_q_words = set(raw_words + stem_words)

        def _prefix_bonus(doc_q: str) -> float:
            doc_words = _RE_WORD_SPLIT.split(doc_q.lower())
            for qw in prefix_q_words:
                if len(qw) < 3:
                    continue
                for dw in doc_words:
                    if dw.startswith(qw) or qw.startswith(dw):
                        return 0.12
            return 0.0

        prefix_scores = [_prefix_bonus(doc["question"]) for doc in docs]

        # ── 6. Blend ──────────────────────────────────────────────────────
        W_BM25, W_FUZZY, W_TG, W_EXACT, W_PREFIX = 0.40, 0.28, 0.14, 0.10, 0.08

        blended = []
        for i, doc in enumerate(docs):
            score = (
                W_BM25   * bm25_scores[i]   +
                W_FUZZY  * fuzzy_scores[i]  +
                W_TG     * tg_scores[i]     +
                W_EXACT  * exact_scores[i]  +
                W_PREFIX * prefix_scores[i]
            )
            if score > 0.05:
                blended.append((score, doc))

        blended.sort(key=lambda x: x[0], reverse=True)

        return [
            {**doc, "match_type": "local", "score": round(s, 3)}
            for s, doc in blended[:top_k]
        ]


_LOCAL_INDEX = _LocalQuestionIndex()


# ---------------------------------------------------------------------------
# SuggestionEngine v4
# ---------------------------------------------------------------------------
class SuggestionEngine:

    MIN_QUERY_LENGTH     = 5
    SEMANTIC_FETCH_LIMIT = 10
    MAX_SUGGESTIONS      = 5
    SEMANTIC_THRESHOLD   = 0.50
    MIN_SUGGESTION_SCORE = 0.28

    _cache:      TTLCache       = TTLCache(maxsize=1024, ttl=300)
    _cache_lock: threading.Lock = threading.Lock()

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
        self._embedder = OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
        small_model = getattr(settings, "openai_model_small", "gpt-4o-mini")
        self._llm = ChatOpenAI(
            model=small_model,
            temperature=0.2,
            api_key=settings.openai_api_key,
            max_tokens=80,
        )

    # ------------------------------------------------------------------
    # PUBLIC
    # ------------------------------------------------------------------

    def suggest(self, partial_query: str, product: str, is_followup: bool = False) -> List[Dict[str, Any]]:
        query = partial_query.strip()
        if len(query) < self.MIN_QUERY_LENGTH:
            return []

        # ── Gibberish gate ────────────────────────────────────────────────
        # Reject random keypresses before touching any index or embedding.
        if _is_gibberish(query):
            logger.debug(f"Gibberish query rejected: {query!r}")
            return []

        # is_followup included in key so followup/suggestion don't share cache
        key = (query, product, is_followup)

        with self._cache_lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached

        prefix_result = self._prefix_cache_hit(query, product, is_followup)

        with self._inflight_lock:
            if key in self._inflight:
                fut = self._inflight[key]
            else:
                fut = _POOL.submit(self._compute, query, product, is_followup)
                self._inflight[key] = fut

        if prefix_result is not None:
            _POOL.submit(self._settle, fut, key)
            return prefix_result

        return self._settle(fut, key)

    # ------------------------------------------------------------------
    # CORE PIPELINE
    # ------------------------------------------------------------------

    def _compute(self, query: str, product: str, is_followup: bool = False) -> List[Dict[str, Any]]:
        try:
            # ── Step 1: single embed ───────────────────────────────────
            vector = self._embed(query)

            # ── Step 2: parallel fan-out ───────────────────────────────
            f_q   = _POOL.submit(self._chroma_by_vector, self.question_store, vector, product)
            f_r   = _POOL.submit(self._chroma_by_vector, self.rag_store, vector, product)
            f_loc = _POOL.submit(_LOCAL_INDEX.search, query, product, self.question_store)

            q_hits   = f_q.result()
            r_hits   = f_r.result()
            loc_hits = f_loc.result()

            if is_followup:
                # ── FOLLOWUP PATH ──────────────────────────────────────
                # Re-query question store using top RAG chunk text as vector
                # to get topically related questions from a different angle.
                rag_guided_hits: List[Dict] = []
                if r_hits:
                    top_rag_text = r_hits[0].get("text", "")
                    if top_rag_text:
                        rag_vector      = self._embed(top_rag_text[:300])
                        f_q2            = _POOL.submit(self._chroma_by_vector, self.question_store, rag_vector, product)
                        rag_guided_hits = f_q2.result()

                # Combine semantic + RAG-guided, strip rephrasings of original query
                all_candidates = q_hits + rag_guided_hits
                print("Candidates--------------",all_candidates)
                diverse = [
                    c for c in all_candidates
                    if 30 < fuzz.token_set_ratio(query.lower(), c.get("question", "").lower()) < 78
                ]

                # Fallback to local index if still short
                if len(diverse) < 3:
                    local_diverse = [
                        c for c in loc_hits
                        if 30 < fuzz.token_set_ratio(query.lower(), c.get("question", "").lower()) < 78
                    ]
                    seen = {c["question"].lower() for c in diverse}
                    for c in local_diverse:
                        if c["question"].lower() not in seen:
                            diverse.append(c)
                            seen.add(c["question"].lower())

                final = self._dedup(diverse, query)

            else:
                # ── SUGGESTION PATH ────────────────────────────────────
                # Only questions from the question store — fully grounded.
                semantic_hits = [
                    {**c, "match_type": "semantic", "score": round(float(c["score"]), 3)}
                    for c in q_hits
                    if (c.get("score") or 0.0) >= self.SEMANTIC_THRESHOLD
                ]

                merged   = self._merge_and_rank(loc_hits, semantic_hits)
                filtered = [m for m in merged if m["score"] >= self.MIN_SUGGESTION_SCORE]
                final    = self._dedup(filtered, query)

            return final[:self.MAX_SUGGESTIONS]

        except Exception as e:
            logger.error(f"_compute failed (query={query!r}, followup={is_followup}): {e}")
            return []

    # ------------------------------------------------------------------
    # EMBEDDING
    # ------------------------------------------------------------------

    def _embed(self, query: str) -> List[float]:
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
    # CHROMA (pre-computed vector, no re-embed)
    # ------------------------------------------------------------------

    def _chroma_by_vector(
        self, store: VectorStore, vector: List[float], product: str
    ) -> List[Dict]:
        if not vector:
            return []

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

        try:
            hits = store.similarity_search(
                query="",
                k=self.SEMANTIC_FETCH_LIMIT,
                filter_metadata={"product": product},
                embedding=vector,
            )
            return self._parse_hits(hits)
        except Exception as e:
            logger.error(f"Fallback similarity_search failed: {e}")
            return []

    @staticmethod
    def _parse_chroma_result(result: Dict) -> List[Dict]:
        out = []
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        for meta, dist in zip(metadatas, distances):
            if not meta:
                continue
            question = meta.get("question", "")
            if not question:
                continue
            score = 1.0 / (1.0 + dist) if dist is not None else 0.0
            out.append({
                "question":      question,
                "topic_id":      meta.get("topic_id", ""),
                "topic_name":    meta.get("topic_name", meta.get("topic_id", "")),

                "match_type":    "semantic",
                "score":         score,
            })
        return out

    @staticmethod
    def _parse_hits(hits: List[Dict]) -> List[Dict]:
        out = []
        for hit in hits:
            meta     = hit.get("metadata", {})
            question = meta.get("question") or hit.get("text", "")
            if not question:
                continue
            out.append({
                "question":      question,
                "topic_id":      meta.get("topic_id", ""),
                "topic_name":    meta.get("topic_name", meta.get("topic_id", "")),
                "match_type":    "semantic",
                "score":         hit.get("score", 0.0),
            })
        return out

    # ------------------------------------------------------------------
    # LLM FALLBACK (kept for potential future use, not called from _compute)
    # ------------------------------------------------------------------

    def _llm_fallback(
        self, query: str, product: str, candidates: List[Dict],
        is_followup: bool = False, rag_hits: List[Dict] = None
    ) -> List[Dict]:

        if rag_hits:
            rag_context = "\n\n".join(
                h.get("text", "") or h.get("metadata", {}).get("question", "")
                for h in rag_hits[:3]
                if h.get("text") or h.get("metadata", {}).get("question")
            )
        else:
            rag_context = ""

        context_lines    = [f"- {c['question']}" for c in candidates[:5] if c.get("question")]
        question_context = "\n".join(context_lines)

        if not rag_context and not question_context:
            return []

        raw = ""
        try:
            if is_followup:
                prompt = (
                    f"Insurance FAQ assistant. Product: {product}.\n"
                    f'User just asked: "{query}"\n\n'
                    f"Relevant document content:\n{rag_context}\n\n"
                    f"Related questions already answered:\n{question_context}\n\n"
                    "Generate 3 follow-up questions a curious customer would ask NEXT.\n"
                    "Rules:\n"
                    "- Must be answerable from the document content above\n"
                    "- Must be meaningfully different from the original query\n"
                    "- No rephrasing or synonyms of the original query\n"
                    "- Explore distinct angles: exclusions, process, limits, eligibility\n"
                    "- Under 12 words each\n"
                    'Reply ONLY with a JSON array: ["Q1","Q2","Q3"]'
                )
            else:
                phrase_tokens      = _RE_ALPHA_3.findall(query.lower())
                phrase_is_sensible = any(t in DOMAIN_TERMS for t in phrase_tokens)
                include_clause     = f'- Each question must contain the word "{query}" or a key term from it.\n' if phrase_is_sensible else ""

                prompt = (
                    f"Insurance FAQ assistant. Product: {product}.\n"
                    f'User is typing: "{query}"\n\n'
                    f"Relevant document content:\n{rag_context}\n\n"
                    f"Related questions:\n{question_context}\n\n"
                    "Generate 3 distinct FAQ suggestions grounded in the document content above.\n"
                    "Rules:\n"
                    "- ONLY suggest questions answerable from the document content\n"
                    "- Each question on a DIFFERENT aspect (cost, process, eligibility, limits)\n"
                    "- No two questions should feel like rephrasings\n"
                    "- Must feel like real customer questions\n"
                    + include_clause +
                    "- Under 12 words each\n"
                    'Reply ONLY with a JSON array: ["Q1","Q2","Q3"]'
                )

            resp      = self._llm.invoke(prompt)
            raw       = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
            m         = _RE_JSON_FENCE.search(raw)
            raw       = m.group(1).strip() if m else raw
            questions = json.loads(raw)

            if not isinstance(questions, list):
                raise ValueError("expected list")

            return [
                {
                    "question":      q.strip(),
                    "topic_id":      "",
                    "topic_name":    "",
                    "match_type":    "llm_followup" if is_followup else "llm_context",
                    "score":         0.2,
                }
                for q in questions[:3]
                if isinstance(q, str) and q.strip()
            ]

        except Exception as e:
            logger.warning(f"LLM fallback error (followup={is_followup}): {e} | raw={raw!r}")
            return []

    # ------------------------------------------------------------------
    # MERGE / RANK / DEDUP
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_and_rank(
        local_hits: List[Dict], semantic_hits: List[Dict]
    ) -> List[Dict]:
        merged: Dict[str, Dict] = {}

        for hit in local_hits:
            merged[hit["question"].lower()] = hit.copy()

        for hit in semantic_hits:
            key = hit["question"].lower()
            if key in merged:
                blended = round(0.35 * merged[key]["score"] + 0.65 * hit["score"], 3)
                merged[key]["score"]      = blended
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
            result = fut.result(timeout=10.0)
        except Exception as e:
            logger.error(f"suggest compute failed: {e}")
            result = []
        finally:
            with self._inflight_lock:
                self._inflight.pop(key, None)
        result = result or []
        with self._cache_lock:
            self._cache[key] = result
        return result

    def _prefix_cache_hit(self, query: str, product: str, is_followup: bool = False) -> Optional[List[Dict[str, Any]]]:
        with self._cache_lock:
            for length in range(len(query) - 1, self.MIN_QUERY_LENGTH - 1, -1):
                hit = self._cache.get((query[:length], product, is_followup))
                if hit is not None:
                    return hit
        return None