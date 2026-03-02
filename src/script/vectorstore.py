# """
# Vector store implementation using ChromaDB + OpenAI embeddings
# """
# from typing import List, Dict, Any, Optional
# from pathlib import Path
# import chromadb
# from chromadb.config import Settings
# from loguru import logger
# from openai import OpenAI
# from config import settings

# class VectorStore:
#     """ChromaDB vector store for document retrieval"""
    
#     def __init__(
#         self,
#         collection_name: str = "zucora_insurance_documents",
#         persist_directory: str = "./data/vectordb",
#         embedding_model: str = "text-embedding-3-small"
#     ):
#         """
#         Initialize vector store
        
#         Args:
#             collection_name: Name of the ChromaDB collection
#             persist_directory: Directory to persist the database
#             embedding_model: Sentence transformer model for embeddings
#         """
#         self.collection_name = collection_name
#         self.persist_directory = Path(persist_directory)

#         # Initialize ChromaDB client (Cloud if configured, else local)
#         if settings.chroma_api_key and settings.chroma_tenant and settings.chroma_database:
#             self.client = chromadb.CloudClient(
#                 api_key=settings.chroma_api_key,
#                 tenant=settings.chroma_tenant,
#                 database=settings.chroma_database
#             )
#             logger.info("ChromaDB: using CloudClient")
#         else:
#             self.persist_directory.mkdir(parents=True, exist_ok=True)
#             self.client = chromadb.PersistentClient(
#                 path=str(self.persist_directory),
#                 settings=Settings(
#                     anonymized_telemetry=False,
#                     allow_reset=True
#                 )
#             )
#             logger.info(f"ChromaDB: using local PersistentClient at {self.persist_directory}")
        
#         # OpenAI embeddings client
#         logger.info(f"Using OpenAI embedding model: {embedding_model}")
#         self.embedding_model_name = embedding_model
#         self.openai_client = OpenAI(api_key=settings.openai_api_key)
        
#         # Get or create collection
#         try:
#             self.collection = self.client.get_collection(name=collection_name)
#             logger.info(f"Loaded existing collection: {collection_name}")
#         except:
#             self.collection = self.client.create_collection(
#                 name=collection_name,
#                 metadata={"hnsw:space": "cosine"}
#             )
#             logger.info(f"Created new collection: {collection_name}")
    
        
#     def similarity_search(
#         self,
#         query: str,
#         k: int,
#         filter_metadata: Optional[Dict[str, Any]] = None
#     ) -> List[Dict[str, Any]]:
#         """
#         Perform semantic similarity search using cosine distance.

#         Args:
#             query: User search query
#             k: Number of top results to return
#             filter_metadata: Optional metadata filters (e.g., {"product": "health_insurance"})

#         Returns:
#             List of dictionaries:
#             [
#                 {
#                     "id": str,
#                     "text": str,
#                     "metadata": dict,
#                     "score": float   # cosine similarity (0–1)
#                 }
#             ]
#         """

#         if not query.strip():
#             logger.warning("Empty query received for similarity search")
#             return []

#         # Generate embedding for query via OpenAI
#         try:
#             resp = self.openai_client.embeddings.create(
#                 model=self.embedding_model_name,
#                 input=query,
#                 encoding_format="float"
#             )
#             query_embedding = resp.data[0].embedding
#         except Exception as e:
#             logger.error(f"OpenAI embedding failed: {e}")
#             return []
        
#         try:
#             results = self.collection.query(
#                 query_embeddings=[query_embedding],
#                 n_results=k,
#                 where=filter_metadata
#             )
#         except Exception as e:
#             logger.error(f"Chroma query failed: {e}")
#             return []

#         formatted_results = []

#         if not results or not results.get("ids"):
#             return []

#         ids = results.get("ids", [[]])[0]
#         documents = results.get("documents", [[]])[0]
#         metadatas = results.get("metadatas", [[]])[0]
#         distances = results.get("distances", [[]])[0]

#         for i in range(len(ids)):
#             distance = distances[i] if distances else None

#             # Convert cosine distance → similarity score
#             similarity_score = 1 - distance if distance is not None else None

#             formatted_results.append({
#                 "id": ids[i],
#                 "text": documents[i],
#                 "metadata": metadatas[i],
#                 "score": similarity_score
#             })

#         logger.debug(f"Similarity search returned {len(formatted_results)} results")
#         print(formatted_results)
        

#         return formatted_results

#     def reset(self) -> None:
#         """
#         Delete and recreate the collection.
#         """
#         try:
#             self.client.delete_collection(name=self.collection_name)
#         except Exception as e:
#             logger.warning(f"Delete collection failed (may not exist): {e}")

#         self.collection = self.client.create_collection(
#             name=self.collection_name,
#             metadata={"hnsw:space": "cosine"}
#         )
#         logger.info(f"Reset collection: {self.collection_name}")

#     def get_documents(
#         self,
#         filter_metadata: Optional[Dict[str, Any]] = None,
#         limit: int = 100
#     ) -> List[Dict[str, Any]]:
#         """
#         Fetch documents directly from the collection using metadata filters.

#         Returns:
#             List of dictionaries:
#             [
#                 {
#                     "id": str,
#                     "text": str,
#                     "metadata": dict
#                 }
#             ]
#         """
#         where = None
#         if filter_metadata:
#             if len(filter_metadata) == 1:
#                 where = filter_metadata
#             else:
#                 where = {"$and": [{k: v} for k, v in filter_metadata.items()]}

#         try:
#             results = self.collection.get(
#                 where=where,
#                 limit=limit
#             )
#         except Exception as e:
#             logger.error(f"Chroma get failed: {e}")
#             return []

#         ids = results.get("ids", [])
#         documents = results.get("documents", [])
#         metadatas = results.get("metadatas", [])

#         formatted_results = []
#         for i in range(len(ids)):
#             formatted_results.append({
#                 "id": ids[i],
#                 "text": documents[i] if documents else "",
#                 "metadata": metadatas[i] if metadatas else {}
#             })

#         return formatted_results


#     def add_structured_documents(self, documents: list):
#         if not documents:
#             logger.warning("No structured documents to add")
#             return

#         texts = [doc["text"] for doc in documents]
#         metadatas = [self._sanitize_metadata(doc["metadata"]) for doc in documents]
#         ids = [doc["id"] for doc in documents]

#         logger.info("Generating embeddings for structured documents...")
#         embeddings = self._embed_texts(texts)
#         if not embeddings:
#             logger.error("No embeddings generated; skipping add.")
#             return

#         self.collection.add(
#             documents=texts,
#             metadatas=metadatas,
#             ids=ids,
#             embeddings=embeddings
#         )
        
        
#     def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Chroma metadata values must be str/int/float/bool. Remove None and
#         coerce unsupported types to strings.
#         """
#         clean: Dict[str, Any] = {}
#         for k, v in metadata.items():
#             if v is None:
#                 continue
#             if isinstance(v, (str, int, float, bool)):
#                 clean[k] = v
#             else:
#                 clean[k] = str(v)
#         return clean

#     def _embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
#         """
#         Embed a list of texts using OpenAI embeddings with batching.
#         """
#         all_embeddings: List[List[float]] = []
#         for i in range(0, len(texts), batch_size):
#             batch = texts[i:i + batch_size]
#             try:
#                 resp = self.openai_client.embeddings.create(
#                     model=self.embedding_model_name,
#                     input=batch,
#                     encoding_format="float"
#                 )
#                 all_embeddings.extend([d.embedding for d in resp.data])
#             except Exception as e:
#                 logger.error(f"OpenAI embeddings batch failed: {e}")
#                 return []
#         return all_embeddings
        
"""
Vector store implementation using ChromaDB + OpenAI embeddings
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from loguru import logger
from openai import OpenAI
from config import settings

class VectorStore:
    """ChromaDB vector store for document retrieval"""
    
    def __init__(
        self,
        collection_name: str = "zucora_insurance_documents",
        persist_directory: str = "./data/vectordb",
        embedding_model: str = "text-embedding-3-small"
    ):
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)

        if settings.chroma_api_key and settings.chroma_tenant and settings.chroma_database:
            self.client = chromadb.CloudClient(
                api_key=settings.chroma_api_key,
                tenant=settings.chroma_tenant,
                database=settings.chroma_database
            )
            logger.info("ChromaDB: using CloudClient")
        else:
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB: using local PersistentClient at {self.persist_directory}")

        logger.info(f"Using OpenAI embedding model: {embedding_model}")
        self.embedding_model_name = embedding_model
        self.openai_client = OpenAI(api_key=settings.openai_api_key)

        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")

    def similarity_search(
        self,
        query: str,
        k: int,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if not query.strip():
            logger.warning("Empty query received for similarity search")
            return []

        try:
            resp = self.openai_client.embeddings.create(
                model=self.embedding_model_name,
                input=query,
                encoding_format="float"
            )
            query_embedding = resp.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI embedding failed: {e}")
            return []

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata
            )
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return []

        if not results or not results.get("ids"):
            return []

        ids       = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        formatted_results = []
        for i in range(len(ids)):
            distance         = distances[i] if distances else None
            similarity_score = 1 - distance if distance is not None else None
            formatted_results.append({
                "id":       ids[i],
                "text":     documents[i],
                "metadata": metadatas[i],
                "score":    similarity_score
            })

        logger.debug(f"Similarity search returned {len(formatted_results)} results")
        return formatted_results

    def reset(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception as e:
            logger.warning(f"Delete collection failed (may not exist): {e}")

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Reset collection: {self.collection_name}")

    def get_documents(
        self,
        filter_metadata: Optional[Dict[str, Any]] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        where = None
        if filter_metadata:
            if len(filter_metadata) == 1:
                where = filter_metadata
            else:
                where = {"$and": [{k: v} for k, v in filter_metadata.items()]}

        try:
            results = self.collection.get(where=where, limit=limit)
        except Exception as e:
            logger.error(f"Chroma get failed: {e}")
            return []

        ids       = results.get("ids", [])
        documents = results.get("documents", [])
        metadatas = results.get("metadatas", [])

        return [
            {
                "id":       ids[i],
                "text":     documents[i] if documents else "",
                "metadata": metadatas[i] if metadatas else {}
            }
            for i in range(len(ids))
        ]

    def add_structured_documents(self, documents: list) -> None:
        if not documents:
            logger.warning("No structured documents to add")
            return

        texts     = [doc["text"] for doc in documents]
        metadatas = [self._sanitize_metadata(doc["metadata"]) for doc in documents]
        ids       = [doc["id"] for doc in documents]

        logger.info(f"Generating embeddings for {len(texts)} documents...")
        embeddings = self._embed_texts(texts)
        if not embeddings:
            logger.error("No embeddings generated; skipping add.")
            return

        # Chroma also has a limit on how many docs can be added in one call.
        # Add in batches of 500 to stay safely under both OpenAI and Chroma limits.
        CHROMA_BATCH = 300
        for i in range(0, len(documents), CHROMA_BATCH):
            self.collection.add(
                documents=texts[i:i + CHROMA_BATCH],
                metadatas=metadatas[i:i + CHROMA_BATCH],
                ids=ids[i:i + CHROMA_BATCH],
                embeddings=embeddings[i:i + CHROMA_BATCH],
            )
            logger.info(f"  Added batch {i // CHROMA_BATCH + 1}: docs {i}–{min(i + CHROMA_BATCH, len(documents))}")

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        for k, v in metadata.items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            else:
                clean[k] = str(v)
        return clean

    def _embed_texts(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Embed texts in batches of 100 (well under OpenAI's 1000-input limit).
        Using 100 also keeps individual request payload sizes small and avoids
        token-count limits on large text batches.
        """
        all_embeddings: List[List[float]] = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            try:
                resp = self.openai_client.embeddings.create(
                    model=self.embedding_model_name,
                    input=batch,
                    encoding_format="float"
                )
                all_embeddings.extend([d.embedding for d in resp.data])
                logger.info(f"  Embedding batch {batch_num}/{total_batches} ({len(batch)} texts) ✓")
            except Exception as e:
                logger.error(f"OpenAI embeddings batch {batch_num} failed: {e}")
                return []

        return all_embeddings