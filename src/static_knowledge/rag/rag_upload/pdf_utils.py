"""
PDF Extraction Utilities for the Emotion Regulation Assistant.

This module provides the necessary tooling to extract raw text from 
scientific and therapeutic PDF documents. The extracted text serves as the 
foundational domain knowledge for the agent's "Actions Catalog" (DB2).
Without raw text extraction, the Dynamic Planner cannot access the 
concrete tactical plans necessary for the Executor (Pico et al., 2024).
"""

import os
import io
import logging
import pypdf

logger = logging.getLogger(__name__)

class PDFExtractor:
    """
    A utility class dedicated to parsing and extracting raw text from PDF files.
    It encapsulates the PyPDF library, providing clean interfaces for both 
    in-memory files (streams) and local disk files.
    """
    
    @staticmethod
    def extract_text(pdf_content: io.BytesIO) -> str:
        """
        Extracts text from a PDF file provided as an in-memory BytesIO object.
        
        Useful if documents are uploaded via a web interface (Streamlit file uploader) 
        rather than read from a local hard drive.
        
        Args:
            pdf_content (io.BytesIO): The binary stream of the PDF file.
            
        Returns:
            str: The concatenated raw text extracted from all pages of the PDF.
            
        Raises:
            Exception: If the PDF stream is corrupted or cannot be parsed.
        """
        logger.info("Extracting text from in-memory PDF content...")
        try:
            # Initialize the PDF reader object with the byte stream
            pdf_reader = pypdf.PdfReader(pdf_content)
            text = ""
            
            # Iterate sequentially through all pages to maintain document flow
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                # Only append if text was actually found on the page
                if extracted:
                    text += extracted + "\n\n" # Double newline preserves paragraph breaks
                    
            logger.info(f"Extracted {len(text)} characters from PDF stream.")
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF stream: {e}")
            raise

    @staticmethod
    def extract_text_from_path(file_path: str) -> str:
        """
        Extracts text from a locally stored PDF file.
        
        This is the primary method used by the automated local_to_supabase pipeline.
        
        Args:
            file_path (str): The absolute or relative path to the PDF file on disk.
            
        Returns:
            str: The concatenated raw text extracted from the document.
            
        Raises:
            FileNotFoundError: If the specified file path does not exist.
            Exception: If parsing fails due to encryption or corruption.
        """
        logger.info(f"Extracting text from PDF file at: {file_path}")
        try:
            # Open the file in binary read mode ("rb"), required for PDFs
            with open(file_path, "rb") as f:
                pdf_reader = pypdf.PdfReader(f)
                text = ""
                
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            
            logger.info(f"Extracted {len(text)} characters from {os.path.basename(file_path)}.")
            return text
        except FileNotFoundError:
            logger.error(f"File not found at path: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error extracting text from PDF file {file_path}: {e}")
            raise