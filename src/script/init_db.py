"""
Initialize the insurance FAQ chatbot system
"""
import sys
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings


def main():
    """Initialize the system"""
    logger.add("logs/init.log", rotation="1 day")
    
    print("=" * 60)
    print("Insurance FAQ Chatbot - System Initialization")
    print("=" * 60)
    
    # Create directories
    print("\n1. Creating directories...")
    
    directories = [
        settings.data_dir,
        settings.pdf_dir,
        Path(settings.chroma_persist_directory),
        Path("logs"),
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"   ✓ {directory}")
    
    # Check environment
    print("\n2. Checking environment...")
    
    if not settings.openai_api_key or settings.openai_api_key == "your_anthropic_api_key_here":
        print("   ⚠ Warning: ANTHROPIC_API_KEY not set!")
        print("   Please set it in your .env file")
    else:
        print("   ✓ Anthropic API key configured")
    
    if settings.langchain_api_key:
        print("   ✓ LangSmith monitoring enabled")
    else:
        print("   ℹ LangSmith monitoring disabled (optional)")
    
    # System info
    print("\n3. System Configuration:")
    print(f"   • LLM Model: {settings.llm_model}")
    print(f"   • Embedding Model: {settings.embedding_model}")
    print(f"   • Chunk Size: {settings.chunk_size}")
    print(f"   • Top-K Retrieval: {settings.top_k_retrieval}")
    print(f"   • Top-K Rerank: {settings.top_k_rerank}")
    
    # Next steps
    print("\n" + "=" * 60)
    print("✅ Initialization complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Place PDF documents in:", settings.pdf_dir)
    print("2. Run ingestion:")
    print("   python src/scripts/ingest_documents.py")
    print("3. Start the chatbot:")
    print("   streamlit run src/app.py")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()