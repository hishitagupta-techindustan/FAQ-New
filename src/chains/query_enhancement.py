"""
Query enhancement chain for improving search quality
"""
from typing import List, Optional, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from ..config import settings, QUERY_ENHANCEMENT_PROMPT
from ..utils.cache import query_cache


class QueryEnhancer:
    """Enhance user queries for better retrieval"""
    
    def __init__(self):
        """Initialize query enhancer"""
        self.llm = ChatOpenAI(
            model=settings.openai_model_large,  # e.g. gpt-4o
            temperature=settings.llm_temperature,
            api_key=settings.openai_api_key
        )
        
        self.prompt = ChatPromptTemplate.from_template(QUERY_ENHANCEMENT_PROMPT)
    
    def enhance(
        self,
        query: str,
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Enhance query with conversation context
        
        Args:
            query: Original user query
            conversation_history: Previous conversation turns
            
        Returns:
            Enhanced query
        """
        # Check cache first
        cached = query_cache.get_enhanced_query(query)
        if cached:
            logger.debug("Using cached enhanced query")
            return cached
        
        # Format history
        history_text = self._format_history(conversation_history or [])
        
        # If no history and query is already good, return as-is
        if not history_text and len(query.split()) >= 3:
            return query
        
        try:
            # Enhance query
            messages = self.prompt.format_messages(
                query=query,
                history=history_text or "No previous conversation"
            )
            
            response = self.llm.invoke(messages)
            enhanced_query = response.content.strip()
            
            # Cache the result
            query_cache.set_enhanced_query(query, enhanced_query)
            
            logger.debug(f"Enhanced query: '{query}' -> '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query  # Fallback to original
    
    def _format_history(self, history: List[Dict]) -> str:
        """Format conversation history for prompt"""
        if not history:
            return ""
        
        formatted = []
        for turn in history[-3:]:  # Last 3 turns
            role = turn.get("role", "user")
            content = turn.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")
        
        return "\n".join(formatted)


class QueryExpander:
    """Generate multiple query variations"""
    
    def __init__(self):
        """Initialize query expander"""
        self.llm = ChatOpenAI(
            model=settings.openai_model_large,  # e.g. gpt-4o
            temperature=0.3,
            api_key=settings.openai_api_key
        )
    
    def expand(self, query: str, num_variations: int = 3) -> List[str]:
        """
        Generate query variations for comprehensive retrieval
        
        Args:
            query: Original query
            num_variations: Number of variations to generate
            
        Returns:
            List of query variations (including original)
        """
        prompt = f"""Generate {num_variations - 1} alternative phrasings of this insurance question.
Keep them semantically similar but use different words.

Original: {query}

Alternative phrasings (one per line):"""
        
        try:
            response = self.llm.invoke(prompt)
            variations = [query]  # Include original
            
            # Parse variations
            for line in response.content.strip().split('\n'):
                line = line.strip()
                # Remove numbering if present
                line = line.lstrip('0123456789.-) ')
                if line and line not in variations:
                    variations.append(line)
            
            return variations[:num_variations]
            
        except Exception as e:
            logger.error(f"Error expanding query: {e}")
            return [query]