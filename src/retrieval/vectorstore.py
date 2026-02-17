"""
Vector store implementation using ChromaDB
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from loguru import logger

from utils.pdf_processor import DocumentChunk


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
    
    def add_documents(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to the vector store
        
        Args:
            chunks: List of DocumentChunk objects
        """
        if not chunks:
            logger.warning("No chunks to add")
            return
        
        logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Prepare data
        texts = [chunk.text for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True
        ).tolist()
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_end = min(i + batch_size, len(chunks))
            
            self.collection.add(
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end],
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end]
            )
            
            logger.debug(f"Added batch {i//batch_size + 1}")
        
        logger.info(f"Successfully added {len(chunks)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filters
            
        Returns:
            List of result dictionaries
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(
            query,
            convert_to_numpy=True
        ).tolist()
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )
        
        # Format results
        formatted_results = []
        
        if results['ids'] and len(results['ids']) > 0:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'id': results['ids'][0][i],
                    'distance': results['distances'][0][i] if 'distances' in results else None
                })
        
        logger.debug(f"Retrieved {len(formatted_results)} results for query")
        return formatted_results
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        semantic_weight: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining semantic and keyword matching
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search (0-1)
            
        Returns:
            List of result dictionaries
        """
        # Semantic search
        semantic_results = self.search(query, top_k=top_k * 2)
        
        # Keyword matching (simple approach - check for query terms)
        query_terms = set(query.lower().split())
        
        # Score results
        scored_results = []
        for result in semantic_results:
            text_lower = result['text'].lower()
            
            # Keyword score (fraction of query terms present)
            keyword_score = sum(
                1 for term in query_terms if term in text_lower
            ) / len(query_terms) if query_terms else 0
            
            # Semantic score (convert distance to similarity)
            semantic_score = 1 - (result['distance'] or 0)
            
            # Combined score
            combined_score = (
                semantic_weight * semantic_score +
                (1 - semantic_weight) * keyword_score
            )
            
            result['combined_score'] = combined_score
            scored_results.append(result)
        
        # Sort by combined score
        scored_results.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return scored_results[:top_k]
    
    def count_documents(self) -> int:
        """Get total number of documents in collection"""
        return self.collection.count()
    
    def delete_collection(self) -> None:
        """Delete the collection"""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")
    
    def reset(self) -> None:
        """Reset the vector store"""
        self.delete_collection()
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Reset collection: {self.collection_name}")
        
    def add_structured_documents(self, documents: list):
        texts = [doc["text"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        ids = [doc["id"] for doc in documents]

        self.collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
        
    def similarity_search(
    self,
    query: str,
    k: int ,
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
        
        print(self.collection)

        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=filter_metadata
            )
            print(results)
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

