"""
Reranking module for improving retrieval quality
"""
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from loguru import logger


class Reranker:
    """Rerank retrieved documents using cross-encoder"""
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize reranker
        
        Args:
            model_name: Cross-encoder model to use
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker model loaded")
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents based on relevance to query
        
        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top results to return
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Prepare pairs for cross-encoder
        pairs = [[query, doc['text']] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Add scores to documents
        for doc, score in zip(documents, scores):
            doc['rerank_score'] = float(score)
        
        # Sort by rerank score
        reranked = sorted(
            documents,
            key=lambda x: x['rerank_score'],
            reverse=True
        )
        
        logger.debug(f"Reranked {len(documents)} documents, returning top {top_k}")
        return reranked[:top_k]


class SimpleReranker:
    """Lightweight reranker without cross-encoder (faster but less accurate)"""
    
    @staticmethod
    def rerank(
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Simple reranking based on term overlap and position
        
        Args:
            query: Search query
            documents: List of retrieved documents
            top_k: Number of top results to return
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        query_terms = set(query.lower().split())
        
        for doc in documents:
            text_lower = doc['text'].lower()
            
            # Term overlap score
            overlap_score = sum(
                1 for term in query_terms if term in text_lower
            ) / len(query_terms) if query_terms else 0
            
            # Position score (earlier = better)
            position_score = 1 / (documents.index(doc) + 1)
            
            # Combined score
            doc['rerank_score'] = (
                0.6 * overlap_score +
                0.2 * position_score +
                0.2 * (1 - (doc.get('distance', 0.5)))
            )
        
        # Sort by score
        reranked = sorted(
            documents,
            key=lambda x: x['rerank_score'],
            reverse=True
        )
        
        return reranked[:top_k]