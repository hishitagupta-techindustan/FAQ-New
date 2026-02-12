"""
LangGraph agent nodes for orchestrating the RAG pipeline
"""
from typing import TypedDict, List, Dict, Any, Optional
from loguru import logger

from src.chains.query_enhancement import QueryEnhancer
from src.chains.routing import QueryRouter
from src.chains.generation import AnswerGenerator
from src.retrieval.vectorstore import VectorStore
from src.retrieval.reranker import Reranker
from src.utils.cache import cache_manager, retrieval_cache
from src.config import settings


# State definition
class AgentState(TypedDict):
    """State for the RAG agent"""
    query: str
    enhanced_query: Optional[str]
    query_type: Optional[str]
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    answer: Optional[str]
    sources: List[Dict[str, Any]]
    followup_questions: List[str]
    conversation_history: List[Dict[str, Any]]
    error: Optional[str]


class RAGNodes:
    """Node functions for the RAG graph"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        use_reranker: bool = True
    ):
        """
        Initialize RAG nodes
        
        Args:
            vector_store: Vector store instance
            use_reranker: Whether to use cross-encoder reranker
        """
        self.vector_store = vector_store
        self.query_enhancer = QueryEnhancer()
        self.router = QueryRouter()
        self.answer_generator = AnswerGenerator()
        
        # Initialize reranker if needed
        self.use_reranker = use_reranker
        if use_reranker:
            try:
                self.reranker = Reranker()
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}. Using simple reranker.")
                self.use_reranker = False
        
        logger.info("RAG nodes initialized")
    
    def enhance_query_node(self, state: AgentState) -> AgentState:
        """
        Enhance the user query with context
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        logger.info("Node: Query Enhancement")
        
        try:
            enhanced = self.query_enhancer.enhance(
                query=state["query"],
                conversation_history=state.get("conversation_history", [])
            )
            
            state["enhanced_query"] = enhanced
            logger.debug(f"Enhanced query: {enhanced}")
            
        except Exception as e:
            logger.error(f"Error in query enhancement: {e}")
            state["enhanced_query"] = state["query"]
            state["error"] = f"Query enhancement failed: {str(e)}"
        
        return state
    
    def route_query_node(self, state: AgentState) -> AgentState:
        """
        Route query to determine retrieval strategy
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        logger.info("Node: Query Routing")
        
        try:
            query_type = self.router.route(state["query"])
            state["query_type"] = query_type
            logger.debug(f"Query type: {query_type}")
            
        except Exception as e:
            logger.error(f"Error in routing: {e}")
            state["query_type"] = "simple"
            state["error"] = f"Routing failed: {str(e)}"
        
        return state
    
    def retrieve_documents_node(self, state: AgentState) -> AgentState:
        """
        Retrieve relevant documents
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        logger.info("Node: Document Retrieval")
        
        try:
            query = state.get("enhanced_query") or state["query"]
            query_type = state.get("query_type", "simple")
            
            # Get retrieval params based on query type
            params = self.router.get_retrieval_params(query_type)
            
            # Check cache
            cached_docs = retrieval_cache.get_documents(
                query,
                params["top_k_retrieval"]
            )
            
            if cached_docs:
                logger.debug("Using cached retrieval results")
                state["retrieved_docs"] = cached_docs
            else:
                # Retrieve documents
                if params.get("use_hybrid", False):
                    docs = self.vector_store.hybrid_search(
                        query=query,
                        top_k=params["top_k_retrieval"]
                    )
                else:
                    docs = self.vector_store.search(
                        query=query,
                        top_k=params["top_k_retrieval"]
                    )
                
                state["retrieved_docs"] = docs
                
                # Cache results
                retrieval_cache.set_documents(
                    query,
                    params["top_k_retrieval"],
                    docs
                )
                
                logger.debug(f"Retrieved {len(docs)} documents")
        
        except Exception as e:
            logger.error(f"Error in retrieval: {e}")
            state["retrieved_docs"] = []
            state["error"] = f"Retrieval failed: {str(e)}"
        
        return state
    
    def rerank_documents_node(self, state: AgentState) -> AgentState:
        """
        Rerank retrieved documents
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        logger.info("Node: Document Reranking")
        
        try:
            docs = state.get("retrieved_docs", [])
            
            if not docs:
                logger.warning("No documents to rerank")
                state["reranked_docs"] = []
                return state
            
            query = state.get("enhanced_query") or state["query"]
            query_type = state.get("query_type", "simple")
            params = self.router.get_retrieval_params(query_type)
            
            # Rerank
            if self.use_reranker:
                reranked = self.reranker.rerank(
                    query=query,
                    documents=docs,
                    top_k=params["top_k_rerank"]
                )
            else:
                from ..retrieval.reranker import SimpleReranker
                reranked = SimpleReranker.rerank(
                    query=query,
                    documents=docs,
                    top_k=params["top_k_rerank"]
                )
            
            state["reranked_docs"] = reranked
            logger.debug(f"Reranked to {len(reranked)} documents")
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Fallback to top retrieved docs
            state["reranked_docs"] = state.get("retrieved_docs", [])[:5]
            state["error"] = f"Reranking failed: {str(e)}"
        
        return state
    
    def generate_answer_node(self, state: AgentState) -> AgentState:
        """
        Generate answer from context
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        logger.info("Node: Answer Generation")
        
        reranked_docs = state.get("reranked_docs", [])
        logger.debug(f"Reranked docs ({len(reranked_docs)} items): {reranked_docs}")
        
        try:
            result = self.answer_generator.generate(
                query=state["query"],
                context_docs=state.get("reranked_docs", []),
                conversation_history=state.get("conversation_history", [])
            )
            
            state["answer"] = result["answer"]
            state["sources"] = result["sources"]
            
            logger.debug("Answer generated successfully")
            
        except Exception as e:
            logger.error(f"Error in answer generation: {e}")
            state["answer"] = "I apologize, but I encountered an error generating the answer."
            state["sources"] = []
            state["error"] = f"Answer generation failed: {str(e)}"
        
        return state
    
    def generate_followups_node(self, state: AgentState) -> AgentState:
        """
        Generate follow-up questions
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        logger.info("Node: Follow-up Generation")
        
        try:
            followups = self.answer_generator.generate_followup_questions(
                query=state["query"],
                answer=state.get("answer", ""),
                context_docs=state.get("reranked_docs", []),
                
            )
            
            state["followup_questions"] = followups
            logger.debug(f"Generated {len(followups)} follow-up questions")
            
        except Exception as e:
            logger.error(f"Error generating follow-ups: {e}")
            state["followup_questions"] = []
        
        return state


def should_enhance_query(state: AgentState) -> bool:
    """Decide if query needs enhancement"""
    # Always enhance if there's conversation history
    if state.get("conversation_history"):
        return True
    
    # Enhance short queries
    query_words = len(state["query"].split())
    return query_words < 5


def should_rerank(state: AgentState) -> bool:
    """Decide if reranking is needed"""
    # Rerank for complex queries or when we have many results
    return len(state.get("retrieved_docs", [])) > 5