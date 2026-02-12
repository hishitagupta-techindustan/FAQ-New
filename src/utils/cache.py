"""
Caching utilities for improved performance
"""
import hashlib
import json
from typing import Any, Optional
from cachetools import TTLCache
from loguru import logger


class CacheManager:
    """Manages caching for queries and responses"""
    
    def __init__(self, ttl: int = 3600, maxsize: int = 1000):
        """
        Initialize cache manager
        
        Args:
            ttl: Time to live in seconds
            maxsize: Maximum number of cache entries
        """
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
        self.stats = {"hits": 0, "misses": 0, "sets": 0}
    
    def _generate_key(self, query: str, session_id: Optional[str] = None) -> str:
        """
        Generate cache key from query and session
        
        Args:
            query: User query
            session_id: Optional session identifier
            
        Returns:
            Hash-based cache key
        """
        key_data = f"{query}:{session_id or 'global'}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def get(self, query: str, session_id: Optional[str] = None) -> Optional[Any]:
        """
        Retrieve cached response
        
        Args:
            query: User query
            session_id: Optional session identifier
            
        Returns:
            Cached response or None
        """
        key = self._generate_key(query, session_id)
        
        if key in self.cache:
            self.stats["hits"] += 1
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self.cache[key]
        
        self.stats["misses"] += 1
        logger.debug(f"Cache miss for query: {query[:50]}...")
        return None
    
    def set(self, query: str, response: Any, session_id: Optional[str] = None) -> None:
        """
        Store response in cache
        
        Args:
            query: User query
            response: Response to cache
            session_id: Optional session identifier
        """
        key = self._generate_key(query, session_id)
        self.cache[key] = response
        self.stats["sets"] += 1
        logger.debug(f"Cached response for query: {query[:50]}...")
    
    def invalidate(self, query: str, session_id: Optional[str] = None) -> None:
        """
        Remove specific entry from cache
        
        Args:
            query: User query
            session_id: Optional session identifier
        """
        key = self._generate_key(query, session_id)
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Invalidated cache for query: {query[:50]}...")
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.cache.clear()
        logger.info("Cache cleared")
    
    def get_stats(self) -> dict:
        """
        Get cache statistics
        
        Returns:
            Dictionary with cache stats
        """
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0
        
        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }


class QueryCache:
    """Specialized cache for query enhancements"""
    
    def __init__(self, ttl: int = 7200):  # 2 hours
        self.cache = TTLCache(maxsize=500, ttl=ttl)
    
    def get_enhanced_query(self, original_query: str) -> Optional[str]:
        """Get cached enhanced query"""
        key = hashlib.sha256(original_query.encode()).hexdigest()
        return self.cache.get(key)
    
    def set_enhanced_query(self, original_query: str, enhanced_query: str) -> None:
        """Cache enhanced query"""
        key = hashlib.sha256(original_query.encode()).hexdigest()
        self.cache[key] = enhanced_query


class RetrievalCache:
    """Cache for retrieved documents"""
    
    def __init__(self, ttl: int = 3600):
        self.cache = TTLCache(maxsize=200, ttl=ttl)
    
    def get_documents(self, query: str, top_k: int) -> Optional[list]:
        """Get cached documents for query"""
        key = self._make_key(query, top_k)
        return self.cache.get(key)
    
    def set_documents(self, query: str, top_k: int, documents: list) -> None:
        """Cache retrieved documents"""
        key = self._make_key(query, top_k)
        self.cache[key] = documents
    
    def _make_key(self, query: str, top_k: int) -> str:
        """Generate cache key"""
        key_data = f"{query}:{top_k}"
        return hashlib.sha256(key_data.encode()).hexdigest()


# Global cache instances
cache_manager = CacheManager()
query_cache = QueryCache()
retrieval_cache = RetrievalCache()