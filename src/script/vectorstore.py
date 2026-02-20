"""
Vector store implementation using ChromaDB
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger

class VectorStore:
    """ChromaDB vector store for document retrieval"""
    
    def __init__(
        self,
        collection_name: str = "zucora_insurance_documents",
        persist_directory: str = "./data/vectordb",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the database
            embedding_model: Sentence transformer model for embeddings
        """
        self.collection_name = collection_name
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Load embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
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
        """
        Perform semantic similarity search using cosine distance.

        Args:
            query: User search query
            k: Number of top results to return
            filter_metadata: Optional metadata filters (e.g., {"product": "health_insurance"})

        Returns:
            List of dictionaries:
            [
                {
                    "id": str,
                    "text": str,
                    "metadata": dict,
                    "score": float   # cosine similarity (0–1)
                }
            ]
        """

        if not query.strip():
            logger.warning("Empty query received for similarity search")
            return []

        # Generate embedding for query
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata
            )
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return []

        formatted_results = []

        if not results or not results.get("ids"):
            return []

        ids = results.get("ids", [[]])[0]
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for i in range(len(ids)):
            distance = distances[i] if distances else None

            # Convert cosine distance → similarity score
            similarity_score = 1 - distance if distance is not None else None

            formatted_results.append({
                "id": ids[i],
                "text": documents[i],
                "metadata": metadatas[i],
                "score": similarity_score
            })

        logger.debug(f"Similarity search returned {len(formatted_results)} results")
        
        

        return formatted_results
