"""
PDF document processing utilities
"""
import re
from pathlib import Path
from typing import List, Dict, Any
from loguru import logger

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    logger.warning("PyMuPDF not available, falling back to pypdf")

from pypdf import PdfReader


class DocumentChunk:
    """Represents a chunk of document text with metadata"""
    
    def __init__(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_id: str
    ):
        self.text = text
        self.metadata = metadata
        self.chunk_id = chunk_id
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "text": self.text,
            "metadata": self.metadata,
            "chunk_id": self.chunk_id
        }


class PDFProcessor:
    """Process PDF documents for RAG"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_pymupdf: bool = True
    ):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Target size for text chunks
            chunk_overlap: Overlap between consecutive chunks
            use_pymupdf: Use PyMuPDF if available (better quality)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_pymupdf = use_pymupdf and PYMUPDF_AVAILABLE
    
    def extract_text_from_pdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """
        Extract text from PDF with page metadata
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of dictionaries with text and metadata per page
        """
        if self.use_pymupdf:
            return self._extract_with_pymupdf(pdf_path)
        else:
            return self._extract_with_pypdf(pdf_path)
    
    def _extract_with_pymupdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract using PyMuPDF (better quality)"""
        pages = []
        
        try:
            doc = fitz.open(pdf_path)
            
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text")
                
                # Clean text
                text = self._clean_text(text)
                
                if text.strip():
                    pages.append({
                        "text": text,
                        "page": page_num,
                        "source": pdf_path.name
                    })
            
            doc.close()
            logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error extracting PDF with PyMuPDF: {e}")
            raise
        
        return pages
    
    def _extract_with_pypdf(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract using pypdf (fallback)"""
        pages = []
        
        try:
            reader = PdfReader(pdf_path)
            
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                
                # Clean text
                text = self._clean_text(text)
                
                if text.strip():
                    pages.append({
                        "text": text,
                        "page": page_num,
                        "source": pdf_path.name
                    })
            
            logger.info(f"Extracted {len(pages)} pages from {pdf_path.name}")
            
        except Exception as e:
            logger.error(f"Error extracting PDF with pypdf: {e}")
            raise
        
        return pages
    
    def _clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove page numbers and headers/footers (common patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', '', text)
        
        return text.strip()
    
    def chunk_text(
        self,
        pages: List[Dict[str, Any]]
    ) -> List[DocumentChunk]:
        """
        Split text into overlapping chunks
        
        Args:
            pages: List of page dictionaries
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        chunk_counter = 0
        
        for page_data in pages:
            text = page_data["text"]
            page_num = page_data["page"]
            source = page_data["source"]
            
            # Split into sentences (simple approach)
            sentences = re.split(r'[.!?]+\s+', text)
            
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                sentence_length = len(sentence)
                
                # If adding this sentence exceeds chunk size, save current chunk
                if current_length + sentence_length > self.chunk_size and current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(
                        DocumentChunk(
                            text=chunk_text,
                            metadata={
                                "page": page_num,
                                "source": source,
                                "chunk_index": chunk_counter
                            },
                            chunk_id=f"{source}_page{page_num}_chunk{chunk_counter}"
                        )
                    )
                    chunk_counter += 1
                    
                    # Keep overlap
                    overlap_text = ' '.join(current_chunk)
                    if len(overlap_text) > self.chunk_overlap:
                        # Keep last few sentences for overlap
                        current_chunk = current_chunk[-(len(current_chunk)//2):]
                        current_length = len(' '.join(current_chunk))
                    else:
                        current_chunk = []
                        current_length = 0
                
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            
            # Add remaining text as final chunk
            if current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        metadata={
                            "page": page_num,
                            "source": source,
                            "chunk_index": chunk_counter
                        },
                        chunk_id=f"{source}_page{page_num}_chunk{chunk_counter}"
                    )
                )
                chunk_counter += 1
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def process_pdf(self, pdf_path: Path) -> List[DocumentChunk]:
        """
        Complete pipeline: extract and chunk PDF
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of DocumentChunk objects
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        # Extract pages
        pages = self.extract_text_from_pdf(pdf_path)
        
        # Chunk text
        chunks = self.chunk_text(pages)
        
        return chunks
    
    def process_directory(self, pdf_dir: Path) -> List[DocumentChunk]:
        """
        Process all PDFs in a directory
        
        Args:
            pdf_dir: Directory containing PDFs
            
        Returns:
            List of all DocumentChunk objects
        """
        print("Exists:", pdf_dir.exists())
        print("Is dir:", pdf_dir.is_dir())
        print("Absolute path:", pdf_dir.resolve())
        print("Files inside:", list(pdf_dir.iterdir()) if pdf_dir.exists() else "Folder not found")

        pdf_dir = Path(pdf_dir)
        
        all_chunks = []
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
        
        for pdf_path in pdf_files:
            try:
                chunks = self.process_pdf(pdf_path)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {e}")
                continue
        
        logger.info(f"Processed {len(pdf_files)} files into {len(all_chunks)} chunks")
        return all_chunks