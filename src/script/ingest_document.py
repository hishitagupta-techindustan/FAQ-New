"""
Script to ingest PDF documents into the vector database
"""
import sys
from pathlib import Path
import argparse
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import settings
from src.utils.pdf_processor import PDFProcessor
from src.retrieval.vectorstore import VectorStore


def main():
    """Main ingestion function"""
    parser = argparse.ArgumentParser(
        description="Ingest PDF documents into the vector database"
    )
    parser.add_argument(
        "--pdf-dir",
        type=str,
        default=str(settings.pdf_dir),
        help="Directory containing PDF files"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset the vector store before ingesting"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=settings.chunk_size,
        help="Chunk size for text splitting"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=settings.chunk_overlap,
        help="Overlap between chunks"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logger.add("logs/ingestion.log", rotation="1 day")
    
    logger.info("Starting document ingestion")
    logger.info(f"PDF directory: {args.pdf_dir}")
    logger.info(f"Chunk size: {args.chunk_size}, Overlap: {args.chunk_overlap}")
    
    # Initialize components
    pdf_processor = PDFProcessor(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    vector_store = VectorStore(
        collection_name=settings.chroma_collection_name,
        persist_directory=settings.chroma_persist_directory,
        embedding_model=settings.embedding_model
    )
    
    # Reset if requested
    if args.reset:
        logger.warning("Resetting vector store...")
        vector_store.reset()
    
    # Check PDF directory
    pdf_dir = Path(args.pdf_dir)
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        print(f"Error: Directory {pdf_dir} does not exist")
        return 
    
    # Process PDFs
    logger.info("Processing PDF files...")
    try:
        chunks = pdf_processor.process_directory(pdf_dir)
        
        if not chunks:
            logger.warning("No chunks extracted from PDFs")
            print("Warning: No content extracted from PDFs")
            return
        
        logger.info(f"Extracted {len(chunks)} chunks from PDFs")
        print(f"\nExtracted {len(chunks)} chunks from PDFs")
        
        # Add to vector store
        logger.info("Adding chunks to vector store...")
        print("Adding chunks to vector store...")
        
        vector_store.add_documents(chunks)
        
        
        # Verify
        doc_count = vector_store.count_documents()
        logger.info(f"Ingestion complete. Total documents: {doc_count}")
        print(f"\n✅ Ingestion complete!")
        print(f"Total documents in vector store: {doc_count}")
        
    except Exception as e:
        logger.error(f"Error during ingestion: {e}")
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()