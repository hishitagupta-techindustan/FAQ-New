"""
Configuration settings for the Insurance FAQ Chatbot
"""
import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # API Keys
    openai_api_key: str = "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"
    openai_model_large : str = "gpt-4o"
    langchain_api_key: Optional[str] = None
    
    
    # LLM Settings
   
    llm_temperature: float = 0.0
    llm_max_tokens: int = 8000
    
    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    
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
    chroma_api_key: Optional[str] = None
    chroma_tenant: Optional[str] = None
    chroma_database: Optional[str] = None
    
    # MongoDb 
    mongodb_uri : str ="mongodb+srv://hishitagupta_db_user:UGtdfvm7BJV5xAjh@cluster0.t7gcm7s.mongodb.net/?appName=Cluster0"
    mongodb_db : str ="zucora"
    
    
    # Rate Limiting
    max_requests_per_minute: int = 20
    max_tokens_per_request: int = 4000
    
    # Security
    allowed_origins: str = "http://localhost:8501,http://localhost:8000"
    
    # Paths
    data_dir: Path = Path("/home/hishita/Desktop/Projects/FAQ New/data")
    pdf_dir: Path = Path("/home/hishita/Hishita/FAQ New/src/data/uploads/fff7d5dc-de5c-4c47-828c-f1f40acf1e1c_source.pdf")
    
    # Product name
    default_product_name : str ="zucora"
    
    
    class Config:
        # Resolve .env relative to project root (one level above this file)
        env_file = str(Path(__file__).resolve().parent.parent / ".env")
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        # self.data_dir.mkdir(exist_ok=True)
        # self.pdf_dir.mkdir(exist_ok=True)
        Path(self.chroma_persist_directory).mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
