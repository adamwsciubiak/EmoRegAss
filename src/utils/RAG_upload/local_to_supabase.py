"""
Pipeline for uploading local documents (PDFs, TXTs) to the Supabase vector store.
"""
import os
import sys
import argparse
import logging
import time
from dotenv import load_dotenv

# BEST PRACTICE: This block makes the script runnable from anywhere
# by adding the project's root directory to the Python path.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now, use absolute imports from 'src', which is robust and clear.
# This works because the sys.path was just modified.
# NOTE: Using the RENAMED folder 'RAG_upload' and your filename 'pdf_utilis'.
from src.utils.RAG_upload.pdf_utilis import PDFExtractor
from src.utils.RAG_upload.document_processor import DocumentProcessor

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# This will now create the log file inside the script's directory.
PROCESSED_FILES_LOG = os.path.join(os.path.dirname(__file__), 'processed_files.log')

def load_processed_files() -> set:
    if not os.path.exists(PROCESSED_FILES_LOG):
        return set()
    try:
        with open(PROCESSED_FILES_LOG, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
    except Exception as e:
        logger.error(f"Could not load processed files log: {e}")
        return set()

def mark_file_as_processed(filename: str):
    try:
        with open(PROCESSED_FILES_LOG, 'a', encoding='utf-8') as f:
            f.write(f"{filename}\n")
    except Exception as e:
        logger.error(f"Could not write to processed files log: {e}")

def main():
    parser = argparse.ArgumentParser(description="Upload local documents to Supabase vector store.")
    parser.add_argument("--source-dir", required=True, help="Path to the directory containing documents to upload.")
    args = parser.parse_args()

    if not os.path.isdir(args.source_dir):
        logger.error(f"Source directory not found: {args.source_dir}")
        return

    doc_processor = DocumentProcessor()
    pdf_extractor = PDFExtractor()
    processed_files = load_processed_files()

    logger.info(f"Starting local ingestion pipeline for directory: {args.source_dir}")
    logger.info(f"Found {len(processed_files)} previously processed files.")

    files_to_process = []
    for root, _, files in os.walk(args.source_dir):
        for filename in files:
            if filename.endswith(('.pdf', '.txt')) and filename not in processed_files:
                file_path = os.path.join(root, filename)
                files_to_process.append((filename, file_path))

    if not files_to_process:
        logger.info("No new documents to process.")
        return

    logger.info(f"Found {len(files_to_process)} new document(s) to process.")
    successful_uploads = 0
    for filename, file_path in files_to_process:
        logger.info(f"--- Processing file: {filename} ---")
        try:
            text = ""
            # Assuming you have the updated PDFExtractor with extract_text_from_path
            if filename.endswith('.pdf'):
                text = pdf_extractor.extract_text_from_path(file_path)
            elif filename.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            
            if not text or not text.strip():
                logger.warning(f"No text extracted from {filename}. Skipping.")
                mark_file_as_processed(filename)
                continue

            metadata = {"file_name": filename, "source": "local_upload", "processed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            chunks_stored = doc_processor.process_and_store(text, metadata)
            logger.info(f"Successfully stored '{filename}' in {chunks_stored} chunks.")
            mark_file_as_processed(filename)
            successful_uploads += 1
        except Exception as e:
            logger.error(f"FAILED to process {filename}. Error: {e}", exc_info=True)
    
    logger.info(f"--- Pipeline finished. Successfully uploaded {successful_uploads} new document(s). ---")

if __name__ == "__main__":
    main()