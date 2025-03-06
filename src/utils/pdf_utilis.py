"""
PDF utilities for the Emotion Regulation Assistant.

This module provides utilities for extracting text from PDF files.
"""

import io
import logging
from typing import List, Dict, Any, Optional
import pypdf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFExtractor:
    """
    Extract text from PDF files.
    
    This class provides methods for extracting text from PDF files.
    """
    
    @staticmethod
    def extract_text(pdf_content: io.BytesIO) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_content (io.BytesIO): The PDF file content as a BytesIO object.
            
        Returns:
            str: The extracted text.
        """
        logger.info("Extracting text from PDF...")
        
        try:
            # Create a PDF reader object
            pdf_reader = pypdf.PdfReader(pdf_content)
            
            # Extract text from each page
            text = ""
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
            
            logger.info(f"Extracted {len(text)} characters from PDF")
            return text
        
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            raise