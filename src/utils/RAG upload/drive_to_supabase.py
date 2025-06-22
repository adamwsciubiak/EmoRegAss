"""
Pipeline for monitoring Google Drive and updating the Supabase vector store.

This module provides a pipeline that monitors Google Drive for new files,
extracts text from PDFs, and updates the Supabase vector store.
"""

import os
import logging
import time
from typing import List, Dict, Any, Optional
import argparse
from dotenv import load_dotenv

from src.utils.google_drive_utils import GoogleDriveMonitor
from src.utils.pdf_utils import PDFExtractor
from src.utils.vector_store import VectorStoreManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DriveToSupabasePipeline:
    """
    Pipeline for monitoring Google Drive and updating the Supabase vector store.
    
    This class provides methods for running a pipeline that monitors Google Drive
    for new files, extracts text from PDFs, and updates the Supabase vector store.
    """
    
    def __init__(self, folder_id: str, credentials_path: str, token_path: str):
        """
        Initialize the Drive to Supabase Pipeline.
        
        Args:
            folder_id (str): The ID of the Google Drive folder to monitor.
            credentials_path (str): Path to the Google Drive credentials.json file.
            token_path (str): Path to save the Google Drive token.pickle file.
        """
        self.drive_monitor = GoogleDriveMonitor(folder_id, credentials_path, token_path)
        self.pdf_extractor = PDFExtractor()
        self.vector_store_manager = VectorStoreManager()
        
        logger.info("Drive to Supabase Pipeline initialized")
    
    def run_once(self) -> int:
        """
        Run the pipeline once.
        
        Returns:
            int: The number of files processed.
        """
        logger.info("Running Drive to Supabase Pipeline...")
        
        # Check for new files
        new_files = self.drive_monitor.check_for_new_files()
        
        if not new_files:
            logger.info("No new files found")
            return 0
        
        logger.info(f"Found {len(new_files)} new files")
        
        # Process each file
        processed_count = 0
        for file in new_files:
            file_id = file['id']
            file_name = file['name']
            
            try:
                # Download the file
                file_content = self.drive_monitor.download_file(file_id)
                
                # Extract text from the PDF
                text = self.pdf_extractor.extract_text(file_content)
                
                # Create metadata
                metadata = {
                    "file_id": file_id,
                    "file_name": file_name,
                    "source": "google_drive",
                    "processed_date": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Process and store the document
                chunks_stored = self.vector_store_manager.process_and_store_document(text, metadata)
                
                # Mark the file as processed
                self.drive_monitor.mark_as_processed(file_id)
                
                logger.info(f"Successfully processed file: {file_name} ({chunks_stored} chunks stored)")
                processed_count += 1
            
            except Exception as e:
                logger.error(f"Error processing file {file_name} ({file_id}): {e}")
        
        logger.info(f"Processed {processed_count} out of {len(new_files)} files")
        return processed_count
    
    def run_continuously(self, interval_seconds: int = 3600):
        """
        Run the pipeline continuously at specified intervals.
        
        Args:
            interval_seconds (int, optional): The interval between runs in seconds.
                Defaults to 3600 (1 hour).
        """
        logger.info(f"Starting continuous pipeline with interval: {interval_seconds} seconds")
        
        try:
            while True:
                self.run_once()
                logger.info(f"Sleeping for {interval_seconds} seconds...")
                time.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")
        
        except Exception as e:
            logger.error(f"Pipeline stopped due to error: {e}")
            raise


def main():
    """Run the Drive to Supabase Pipeline."""
    parser = argparse.ArgumentParser(description="Monitor Google Drive and update Supabase vector store")
    parser.add_argument("--folder-id", required=True, help="Google Drive folder ID to monitor")
    parser.add_argument("--credentials", required=True, help="Path to Google Drive credentials.json file")
    parser.add_argument("--token", required=True, help="Path to save Google Drive token.pickle file")
    parser.add_argument("--interval", type=int, default=3600, help="Interval between runs in seconds (default: 3600)")
    parser.add_argument("--once", action="store_true", help="Run the pipeline once and exit")
    
    args = parser.parse_args()
    
    pipeline = DriveToSupabasePipeline(args.folder_id, args.credentials, args.token)
    
    if args.once:
        pipeline.run_once()
    else:
        pipeline.run_continuously(args.interval)


if __name__ == "__main__":
    main()