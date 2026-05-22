"""
PDF utilities for the Emotion Regulation Assistant.

This module provides utilities for extracting text from PDF files.
"""
import os  # <--- ADD THIS LINE
import io
import logging
from typing import List, Dict, Any, Optional
import pypdf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractor:
    """Extract text from PDF files."""
    
    @staticmethod
    def extract_text(pdf_content: io.BytesIO) -> str:
        """
        Extracts text from a PDF file provided as an in-memory BytesIO object.
        """
        # ... (this method is unchanged)
        logger.info("Extracting text from in-memory PDF content...")
        try:
            pdf_reader = pypdf.PdfReader(pdf_content)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n\n"
            logger.info(f"Extracted {len(text)} characters from PDF stream.")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF stream: {e}")
            raise

    @staticmethod
    def extract_text_from_path(file_path: str) -> str:
        """
        Extracts text from a PDF file located at a given file path.
        """
        logger.info(f"Extracting text from PDF file at: {file_path}")
        try:
            with open(file_path, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            
            # This line was causing the error because 'os' was not imported
            logger.info(f"Extracted {len(text)} characters from {os.path.basename(file_path)}.")
            return text
        except FileNotFoundError:
            logger.error(f"File not found at path: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error extracting text from PDF file {file_path}: {e}")
            raise