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



