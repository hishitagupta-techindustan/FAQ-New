"""
LangGraph orchestrator for the RAG pipeline
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from loguru import logger

from src.agents.nodes import AgentState, RAGNodes
from src.retrieval.vectorstore import VectorStore
from src.utils.cache import cache_manager


class RAGOrchestrator:
    """Orchestrates the complete RAG pipeline using LangGraph"""
    
    def __init__(
        self,
        vector_store: VectorStore,
        use_reranker: bool = True
    ):
        """
        Initialize the orchestrator
        
        Args:
            vector_store: Vector store instance
            use_reranker: Whether to use cross-encoder reranker
        """
        self.vector_store = vector_store
        self.nodes = RAGNodes(vector_store, use_reranker)
        self.graph = self._build_graph()
        
        logger.info("RAG Orchestrator initialized")
    
    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow
        
        Returns:
            Compiled graph
        """
        # Create graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("enhance_query", self.nodes.enhance_query_node)
        workflow.add_node("route_query", self.nodes.route_query_node)
        workflow.add_node("retrieve_documents", self.nodes.retrieve_documents_node)
        workflow.add_node("rerank_documents", self.nodes.rerank_documents_node)
        workflow.add_node("generate_answer", self.nodes.generate_answer_node)
        workflow.add_node("generate_followups", self.nodes.generate_followups_node)
        
        # Define edges
        workflow.set_entry_point("enhance_query")
        
        workflow.add_edge("enhance_query", "route_query")
        workflow.add_edge("route_query", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "rerank_documents")
        workflow.add_edge("rerank_documents", "generate_answer")
        workflow.add_edge("generate_answer", "generate_followups")
        workflow.add_edge("generate_followups", END)
        
        # Compile graph
        app = workflow.compile()
        
        logger.info("LangGraph workflow built")
        return app
    
    def query(
        self,
        query: str,
        conversation_history: list = None,
        session_id: str = None
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline
        
        Args:
            query: User query
            conversation_history: Previous conversation turns
            session_id: Session identifier for caching
            
        Returns:
            Dictionary with answer and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # Check cache first
        cached_response = cache_manager.get(query, session_id)
        if cached_response:
            logger.info("Returning cached response")
            return cached_response
        
        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "enhanced_query": None,
            "query_type": None,
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": None,
            "sources": [],
            "followup_questions": [],
            "conversation_history": conversation_history or [],
            "error": None
        }
        
        try:
            # Run graph
            final_state = self.graph.invoke(initial_state)
            
            # Prepare response
            response = {
                "answer": final_state.get("answer", ""),
                "sources": final_state.get("sources", []),
                "followup_questions": final_state.get("followup_questions", []),
                "metadata": {
                    "query_type": final_state.get("query_type"),
                    "num_sources": len(final_state.get("sources", [])),
                    "error": final_state.get("error")
                }
            }
            print(response)
            
            # Cache response
            if not final_state.get("error"):
                cache_manager.set(query, response, session_id)
            
            logger.info("Query processed successfully")
            return response
            
        except Exception as e:
            logger.error(f"Error in RAG pipeline: {e}")
            return {
                "answer": "I apologize, but I encountered an error processing your query. Please try again.",
                "sources": [],
                "followup_questions": [],
                "metadata": {
                    "error": str(e)
                }
            }
    
    def stream_query(
        self,
        query: str,
        conversation_history: list = None
    ):
        """
        Process query with streaming response
        
        Args:
            query: User query
            conversation_history: Previous turns
            
        Yields:
            Response chunks
        """
        # For streaming, we need to use the streaming generator
        from ..chains.generation import StreamingAnswerGenerator
        
        streaming_generator = StreamingAnswerGenerator()
        
        # Run retrieval pipeline
        initial_state: AgentState = {
            "query": query,
            "enhanced_query": None,
            "query_type": None,
            "retrieved_docs": [],
            "reranked_docs": [],
            "answer": None,
            "sources": [],
            "followup_questions": [],
            "conversation_history": conversation_history or [],
            "error": None
        }
        
        try:
            # Run up to generation
            state = initial_state
            state = self.nodes.enhance_query_node(state)
            state = self.nodes.route_query_node(state)
            state = self.nodes.retrieve_documents_node(state)
            state = self.nodes.rerank_documents_node(state)
            
            # Stream answer
            for chunk in streaming_generator.generate_stream(
                query=state["query"],
                context_docs=state.get("reranked_docs", []),
                conversation_history=conversation_history
            ):
                yield {
                    "type": "answer_chunk",
                    "content": chunk
                }
            
            # Send sources
            state = self.nodes.generate_answer_node(state)
            yield {
                "type": "sources",
                "content": state.get("sources", [])
            }
            
            # Send follow-ups
            state = self.nodes.generate_followups_node(state)
            yield {
                "type": "followups",
                "content": state.get("followup_questions", [])
            }
            
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            yield {
                "type": "error",
                "content": str(e)
            }