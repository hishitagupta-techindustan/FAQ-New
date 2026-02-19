"""
Configuration settings for the Insurance FAQ Chatbot
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: str = "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"
    openai_model_large : str = "gpt-4o"
    langchain_api_key: Optional[str] = None
    
    # LangSmith Monitoring
    langchain_tracing_v2: bool = True
    langchain_project: str = "insurance-faq-chatbot"
    
    # Application
    app_env: str = "development"
    debug: bool = True
    
    # LLM Settings
   
    llm_temperature: float = 0.0
    llm_max_tokens: int = 4000
    
    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    
    # Retrieval Settings
    top_k_retrieval: int = 10
    top_k_rerank: int = 5
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    # ChromaDB
    chroma_persist_directory: str = "/home/hishita/Desktop/Projects/FAQ New/data/vectordb"
    chroma_collection_name_questions: str = "zucora_insurance_questions"
    chroma_collection_name_rag: str = "zucora_insurance_faqs_rag"
    reset_vector_store : bool = True
    
    # MongoDb 
    mongodb_uri : str ="mongodb+srv://hishitagupta_db_user:UGtdfvm7BJV5xAjh@cluster0.t7gcm7s.mongodb.net/?appName=Cluster0"
    mongodb_db : str ="zucora"
    
    # Caching
    cache_ttl: int = 3600  # 1 hour
    redis_url: Optional[str] = None
    
    # Rate Limiting
    max_requests_per_minute: int = 20
    max_tokens_per_request: int = 4000
    
    # Security
    allowed_origins: str = "http://localhost:8501,http://localhost:8000"
    
    # Paths
    data_dir: Path = Path("/home/hishita/Desktop/Projects/FAQ New/data")
    pdf_dir: Path = Path("/home/hishita/Desktop/Projects/FAQ New/data/pdfs/zucora.pdf")
    
    # Product name
    default_product_name : str ="zucora"
    
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        # self.data_dir.mkdir(exist_ok=True)
        # self.pdf_dir.mkdir(exist_ok=True)
        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


# Prompt Templates
QUERY_ENHANCEMENT_PROMPT = """You are an expert at understanding insurance-related questions.

Given a user query and conversation history, enhance the query to make it more specific and searchable. Give more significance to user query than conversation history and return the enhanced query accordingly.

Conversation History:
{history}

Current Query: {query}

Enhanced Query (keep it concise and focused):"""


ROUTER_PROMPT = """You are a query router for an insurance FAQ system.

Analyze the user query and determine the best retrieval strategy:
- "simple": For straightforward questions about specific terms or definitions
- "complex": For multi-part questions requiring comprehensive search
- "conversational": For follow-up questions that need conversation context

Query: {query}

Classification (respond with only: simple, complex, or conversational):"""


ANSWER_GENERATION_PROMPT = """You are a helpful insurance assistant. Answer the user's question based on the provided context.

Context from insurance documents:
{context}

Conversation History:
{history}

User Question: {query}

Instructions:
1. If the question is not related to insurance, answer the question in short along with prompting the user to explore Zucora in a witty manner.
2. Answer based ONLY on the provided context
3. If you cannot find the answer in the context, say so clearly
4. Be clear, concise, and accurate
5. If the question is ambiguous, ask for clarification
6. Answer in 3 small blocks of under 50 words without using the em dash.
7. Answers should be user centric yet professional.

Answer:"""


FOLLOWUP_QUESTIONS_PROMPT = """Based on the user's question, the answer provided and the context, suggest 3 relevant follow-up questions they might ask. If the query is unrelated to zucora, add followup questions based on retrieved docs.

User Question: {query}
Answer: {answer}
Context: {context}


Generate 3 follow-up questions (one per line, without numbering):"""
