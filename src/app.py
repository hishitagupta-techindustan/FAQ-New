
import streamlit as st
from typing import List, Dict
from loguru import logger
from config import settings
from src.retrieval.vectorstore import VectorStore
from src.agents.graph import RAGOrchestrator
from src.utils.security import SecurityFilter, RateLimiter, validate_session_id
from src.utils.cache import cache_manager

# Configure logger
logger.add("logs/app.log", rotation="1 day", retention="7 days")

# Page config
st.set_page_config(
    page_title="Insurance FAQ Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .followup-button {
        margin: 5px;
    }
    .metric-box {
        background-color: #e8eaf6;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize the RAG system"""
    try:
        logger.info("Initializing RAG system...")
        
        # Initialize vector store
        vector_store = VectorStore(
            collection_name=settings.chroma_collection_name_rag,
            persist_directory=settings.chroma_persist_directory,
            embedding_model=settings.embedding_model
        )
        
        # Check if vector store has documents
        doc_count = vector_store.count_documents()
        if doc_count == 0:
            logger.warning("Vector store is empty! Please ingest documents first.")
        else:
            logger.info(f"Vector store loaded with {doc_count} documents")
        
        # Initialize orchestrator
        orchestrator =  RAGOrchestrator(
            vector_store=vector_store,
            use_reranker=True
        )
        
        logger.info("RAG system initialized successfully")
        return orchestrator, vector_store
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        st.error(f"Failed to initialize system: {e}")
        return None, None


@st.cache_resource
def initialize_security():
    """Initialize security components"""
    return SecurityFilter(), RateLimiter(
        max_requests=settings.max_requests_per_minute,
        window_seconds=60
    )



def render_message(message: Dict):
    """Render a chat message"""
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        st.markdown(content)
        
        # Render sources if available
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources", expanded=False):
                for i, source in enumerate(message["sources"], 1):
                    if isinstance(source, str):
                        st.markdown(f"**{i}.** {source}")
                    elif isinstance(source, dict):
                        source_name = source.get('source', 'Unknown')
                        page = source.get('page')
                        source_display = f"**{i}.** {source_name}"
                        if page:
                            source_display += f" (Page {page})"
                        st.markdown(source_display)
        
        # Render follow-up questions
        if "followups" in message and message["followups"]:
            st.markdown("**💭 Related questions:**")
            msg_idx = st.session_state.messages.index(message)
            for idx, question in enumerate(message["followups"]):
                if st.button(
                    question,
                    key=f"followup_{msg_idx}_{idx}",
                    use_container_width=True
                ):
                    st.session_state.pending_query = question
                    st.rerun()


def process_query(user_query: str, orchestrator, security_filter, rate_limiter):
    """Process a user query with streaming response"""
    
    # Validate session
    if not validate_session_id(st.session_state.session_id):
        st.error("Invalid session. Please refresh the page.")
        return None
    
    # Rate limiting
    is_allowed, rate_msg = rate_limiter.is_allowed(st.session_state.session_id)
    if not is_allowed:
        st.warning(rate_msg)
        return None
    
    # Security check
    sanitized_query, is_safe, rejection_reason = security_filter.validate_and_sanitize(
        user_query,
        strict=True
    )
    
    if not is_safe:
        st.warning(f"⚠️ Your query was rejected: {rejection_reason}")
        logger.warning(f"Rejected query: {user_query}")
        return None
    
    # Add user message to session state
    st.session_state.messages.append({
        "role": "user",
        "content": sanitized_query
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(sanitized_query)
    
    # Generate streaming response
    with st.chat_message("assistant"):
        # Create placeholder for streaming content
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Check if orchestrator has streaming method
            if hasattr(orchestrator, 'query_stream'):
                # Stream the response
                for chunk in orchestrator.query_stream(
                    query=sanitized_query,
                    conversation_history=st.session_state.conversation_history,
                    session_id=st.session_state.session_id
                ):
                    if isinstance(chunk, dict):
                        # Handle structured chunk
                        if "token" in chunk:
                            full_response += chunk["token"]
                            message_placeholder.markdown(full_response + "▌")
                        elif "answer" in chunk:
                            full_response = chunk["answer"]
                            message_placeholder.markdown(full_response)
                    else:
                        # Handle string chunk
                        full_response += chunk
                        message_placeholder.markdown(full_response + "▌")
                
                # Remove cursor
                message_placeholder.markdown(full_response)
                
                # Get final response with sources and follow-ups
                response = orchestrator.get_last_response()
                
            else:
                # Fallback to non-streaming with spinner
                with st.spinner("Thinking..."):
                    response = orchestrator.query(
                        query=sanitized_query,
                        conversation_history=st.session_state.conversation_history,
                        session_id=st.session_state.session_id
                    )
                    full_response = response["answer"]
                    message_placeholder.markdown(full_response)
            
            # Display sources
            if response.get("sources"):
                with st.expander("📚 Sources", expanded=False):
                    for i, source in enumerate(response["sources"], 1):
                        if isinstance(source, str):
                            st.markdown(f"**{i}.** {source}")
                        elif isinstance(source, dict):
                            source_name = source.get('source', 'Unknown')
                            page = source.get('page')
                            source_display = f"**{i}.** {source_name}"
                            if page:
                                source_display += f" (Page {page})"
                            st.markdown(source_display)
            
            # Display follow-up questions
            if response.get("followup_questions"):
                st.markdown("**💭 Related questions:**")
                # Use temporary index for new messages
                temp_idx = len(st.session_state.messages)
                for idx, question in enumerate(response["followup_questions"]):
                    if st.button(
                        question,
                        key=f"followup_new_{temp_idx}_{idx}",
                        use_container_width=True
                    ):
                        st.session_state.pending_query = question
                        st.rerun()
            
            # Add assistant message to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": full_response,
                "sources": response.get("sources", []),
                "followups": response.get("followup_questions", [])
            })

            # Update conversation history
            st.session_state.conversation_history.append({
                "role": "user",
                "content": sanitized_query
            })
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            st.error(f"Error: {str(e)}")
            # Remove incomplete message from session state if error occurs
            if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.messages.pop()
            return None


def main():
    """Main application"""
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "session_id" not in st.session_state:
        import uuid
        st.session_state.session_id = str(uuid.uuid4())
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    
    # Initialize system
    orchestrator, vector_store = initialize_system()
    security_filter, rate_limiter = initialize_security()
    
    if not orchestrator:
        st.error("System initialization failed. Please check logs.")
        return
    
    # Render UI
    st.title("🏥 Insurance FAQ Assistant")
    st.caption("Ask me anything about your insurance policies")
    
    # render_sidebar()
    
    # Process pending query first (from follow-up button)
    if st.session_state.pending_query:
        user_query = st.session_state.pending_query
        st.session_state.pending_query = None
        process_query(user_query, orchestrator, security_filter, rate_limiter)
    
    # Display chat history
    for message in st.session_state.messages:
        render_message(message)
    
    # Chat input
    user_query = st.chat_input("Type your question here...")
    
    if user_query:
        process_query(user_query, orchestrator, security_filter, rate_limiter)


if __name__ == "__main__":
    main()
