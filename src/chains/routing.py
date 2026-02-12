"""
Query routing for adaptive retrieval strategies
"""
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from ..config import settings, ROUTER_PROMPT


QueryType = Literal["simple", "complex", "conversational"]


class QueryRouter:
    """Route queries to appropriate retrieval strategies"""
    
    def __init__(self):
        """Initialize query router"""
        self.llm = ChatOpenAI(
            model=settings.openai_model_large,  # e.g. gpt-4o
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key
        )
        
        self.prompt = ChatPromptTemplate.from_template(ROUTER_PROMPT)
    
    def route(self, query: str) -> QueryType:
        """
        Classify query type
        
        Args:
            query: User query
            
        Returns:
            Query type classification
        """
        try:
            messages = self.prompt.format_messages(query=query)
            response = self.llm.invoke(messages)
            
            classification = response.content.strip().lower()
            
            # Validate and normalize
            if "simple" in classification:
                query_type = "simple"
            elif "complex" in classification:
                query_type = "complex"
            elif "conversational" in classification:
                query_type = "conversational"
            else:
                # Fallback heuristic
                query_type = self._heuristic_classification(query)
            
            logger.debug(f"Routed query as: {query_type}")
            return query_type
            
        except Exception as e:
            logger.error(f"Error routing query: {e}")
            return self._heuristic_classification(query)
    
    def _heuristic_classification(self, query: str) -> QueryType:
        """
        Fallback heuristic classification
        
        Args:
            query: User query
            
        Returns:
            Query type
        """
        query_lower = query.lower()
        
        # Conversational indicators
        conversational_words = ["it", "that", "this", "they", "what about"]
        if any(word in query_lower for word in conversational_words):
            return "conversational"
        
        # Complex indicators (multiple questions, comparisons)
        complex_indicators = ["compare", "difference between", "both", "versus", "vs"]
        if any(ind in query_lower for ind in complex_indicators):
            return "complex"
        
        # Question marks suggest complexity
        if query.count('?') > 1:
            return "complex"
        
        # Default to simple
        return "simple"
    
    def get_retrieval_params(self, query_type: QueryType) -> dict:
        """
        Get retrieval parameters based on query type
        
        Args:
            query_type: Classified query type
            
        Returns:
            Dictionary of retrieval parameters
        """
        params = {
            "simple": {
                "top_k_retrieval": 5,
                "top_k_rerank": 3,
                "use_hybrid": False
            },
            "complex": {
                "top_k_retrieval": 15,
                "top_k_rerank": 7,
                "use_hybrid": True
            },
            "conversational": {
                "top_k_retrieval": 10,
                "top_k_rerank": 5,
                "use_hybrid": True,
                "use_history": True
            }
        }
        
        return params.get(query_type, params["simple"])