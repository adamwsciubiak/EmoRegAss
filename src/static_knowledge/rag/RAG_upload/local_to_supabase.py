"""
Data Ingestion Pipeline for the Static Knowledge Base.

This script acts as the automated orchestration pipeline for uploading local therapeutic 
documents (PDFs, TXTs) to the Supabase vector store. It ensures that the 
Emotion Regulation Agent has a fully populated "Actions Catalog" (DB2) 
to pull tactical plans from during the interaction cycle (Pico et al., 2024).

It reads files from a designated folder, extracts text, chunks it, embeds it, 
and logs successful uploads to prevent duplicate processing in the future.

Usage:
    Run this script from the root of the project:
    python -m src.static_knowledge.rag.rag_upload.local_to_supabase
"""

import os
import sys
import argparse
import logging
import time
from dotenv import load_dotenv

# --- SYSTEM PATH RESOLUTION ---
# To allow this script to run seamlessly from the terminal while importing 
# modules from the 'src' package, we must explicitly add the project root 
# (four levels up from this script's location) to the Python system path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import extraction and processing tools
from src.static_knowledge.rag.rag_upload.pdf_utils import PDFExtractor
from src.static_knowledge.rag.rag_upload.document_processor import DocumentProcessor
import src.static_knowledge.rag.rag_upload.rag_config as rag_cfg

# Load environment variables (API keys, Supabase URLs) needed by the VectorStoreConnector
load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_processed_files() -> set:
    """
    Loads the registry of previously processed files.
    
    This function reads 'processed_files.log'. It prevents the pipeline from 
    re-uploading and re-embedding the same PDF multiple times, which would 
    bloat the database and cause duplicate search results during retrieval.
    
    Returns:
        set: A collection of filenames that have already been uploaded.
    """
    if not os.path.exists(rag_cfg.PROCESSED_FILES_LOG_PATH):
        return set() # Return an empty set if the log doesn't exist yet
    try:
        with open(rag_cfg.PROCESSED_FILES_LOG_PATH, 'r', encoding='utf-8') as f:
            # Read lines and strip newline characters, storing them in an O(1) lookup set
            return set(f.read().splitlines())
    except Exception as e:
        logger.error(f"Could not load processed files log: {e}")
        return set()

def mark_file_as_processed(filename: str) -> None:
    """
    Registers a filename as successfully processed by appending it to the log file.
    
    Args:
        filename (str): The name of the processed file (e.g., 'CBT_manual.pdf').
    """
    try:
        # Open in append mode ('a') to add to the bottom of the log safely
        with open(rag_cfg.PROCESSED_FILES_LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(f"{filename}\n")
    except Exception as e:
        logger.error(f"Could not write to processed files log: {e}")

def main():
    """
    Orchestrates the full data ingestion lifecycle.
    
    Workflow:
    1. Parse command-line arguments to find the target directory.
    2. Scan the directory for valid file types (.pdf, .txt) excluding already processed ones.
    3. Iterate through valid files, extracting raw text via PDFExtractor.
    4. Pass the raw text and metadata to DocumentProcessor for chunking and embedding.
    5. Mark the file as processed.
    """
    # Define command line arguments. Defaults to the folder specified in rag_config.py
    parser = argparse.ArgumentParser(description="Upload therapeutic documents to the DB2 Vector Store.")
    parser.add_argument(
        "--source-dir", 
        default=rag_cfg.DEFAULT_SOURCE_DIR,
        help="Path to the directory containing documents to upload."
    )
    args = parser.parse_args()

    # Verify the target directory exists
    if not os.path.isdir(args.source_dir):
        logger.error(f"Source directory not found: {args.source_dir}")
        return

    # Initialize processing agents
    doc_processor = DocumentProcessor()
    pdf_extractor = PDFExtractor()
    processed_files = load_processed_files()

    logger.info(f"Starting local ingestion pipeline for directory: {args.source_dir}")
    logger.info(f"Found {len(processed_files)} previously processed files in the registry.")

    # Step 1: Scan the directory and filter files
    files_to_process = []
    for root, _, files in os.walk(args.source_dir):
        for filename in files:
            # Filter by extension and ensure it hasn't been uploaded before
            if filename.endswith(('.pdf', '.txt')) and filename not in processed_files:
                file_path = os.path.join(root, filename)
                files_to_process.append((filename, file_path))

    if not files_to_process:
        logger.info("No new documents to process.")
        return

    logger.info(f"Found {len(files_to_process)} new document(s) to process.")
    successful_uploads = 0
    
    # Step 2: Ingestion Loop
    for filename, file_path in files_to_process:
        logger.info(f"--- Processing file: {filename} ---")
        try:
            text = ""
            # Extract text based on file type
            if filename.endswith('.pdf'):
                text = pdf_extractor.extract_text_from_path(file_path)
            elif filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            # Safeguard against empty or unreadable files
            if not text or not text.strip():
                logger.warning(f"No text extracted from {filename}. Skipping.")
                mark_file_as_processed(filename) # Mark to avoid retrying corrupted files forever
                continue

            # Construct contextual metadata for the vector store
            metadata = {
                "file_name": filename, 
                "source": "local_upload", 
                "processed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            }
            
            # Step 3: Chunk, Embed, and Store
            chunks_stored = doc_processor.process_and_store(text, metadata)
            logger.info(f"Successfully stored '{filename}' in {chunks_stored} chunks.")
            
            # Step 4: Register success
            mark_file_as_processed(filename)
            successful_uploads += 1
            
        except Exception as e:
            logger.error(f"FAILED to process {filename}. Error: {e}", exc_info=True)
    
    logger.info(f"--- Pipeline finished. Successfully uploaded {successful_uploads} new document(s). ---")

if __name__ == "__main__":
    main()