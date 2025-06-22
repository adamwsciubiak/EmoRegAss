"""
Google Drive utilities for the Emotion Regulation Assistant.

This module provides utilities for monitoring Google Drive for new files
and downloading them for processing.
"""

import os
import io
import logging
from typing import List, Dict, Any, Optional
import pickle
import datetime
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GoogleDriveMonitor:
    """
    Monitor Google Drive for new files.
    
    This class provides methods for authenticating with Google Drive,
    monitoring a specific folder for new files, and downloading them.
    """
    
    def __init__(self, folder_id: str, credentials_path: str, token_path: str):
        """
        Initialize the Google Drive Monitor.
        
        Args:
            folder_id (str): The ID of the folder to monitor.
            credentials_path (str): Path to the credentials.json file.
            token_path (str): Path to save the token.pickle file.
        """
        self.folder_id = folder_id
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.scopes = ['https://www.googleapis.com/auth/drive.readonly']
        self.service = self._authenticate()
        self.processed_files = self._load_processed_files()
        
        logger.info(f"Google Drive Monitor initialized for folder: {folder_id}")
    
    def _authenticate(self):
        """
        Authenticate with Google Drive.
        
        Returns:
            service: A Google Drive service object.
        """
        creds = None
        
        # Check if token.pickle exists
        if os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        
        # If credentials don't exist or are invalid, get new ones
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, self.scopes)
                creds = flow.run_local_server(port=0)
            
            # Save the credentials for the next run
            with open(self.token_path, 'wb') as token:
                pickle.dump(creds, token)
        
        # Build the service
        service = build('drive', 'v3', credentials=creds)
        return service
    
    def _load_processed_files(self) -> Dict[str, datetime.datetime]:
        """
        Load the list of processed files.
        
        Returns:
            Dict[str, datetime.datetime]: A dictionary of file IDs and their processing timestamps.
        """
        processed_files_path = os.path.join(os.path.dirname(self.token_path), 'processed_files.pickle')
        
        if os.path.exists(processed_files_path):
            with open(processed_files_path, 'rb') as f:
                return pickle.load(f)
        
        return {}
    
    def _save_processed_files(self):
        """Save the list of processed files."""
        processed_files_path = os.path.join(os.path.dirname(self.token_path), 'processed_files.pickle')
        
        with open(processed_files_path, 'wb') as f:
            pickle.dump(self.processed_files, f)
    
    def check_for_new_files(self, mime_type: str = 'application/pdf') -> List[Dict[str, Any]]:
        """
        Check for new files in the monitored folder.
        
        Args:
            mime_type (str, optional): The MIME type of files to check for.
                Defaults to 'application/pdf'.
                
        Returns:
            List[Dict[str, Any]]: A list of new file metadata.
        """
        logger.info(f"Checking for new files in folder: {self.folder_id}")
        
        # Query for files in the folder
        query = f"'{self.folder_id}' in parents and mimeType='{mime_type}' and trashed=false"
        results = self.service.files().list(
            q=query,
            fields="files(id, name, mimeType, createdTime)"
        ).execute()
        
        files = results.get('files', [])
        
        # Filter for new files
        new_files = []
        for file in files:
            file_id = file['id']
            if file_id not in self.processed_files:
                logger.info(f"Found new file: {file['name']} ({file_id})")
                new_files.append(file)
        
        return new_files
    
    def download_file(self, file_id: str) -> io.BytesIO:
        """
        Download a file from Google Drive.
        
        Args:
            file_id (str): The ID of the file to download.
            
        Returns:
            io.BytesIO: The file content as a BytesIO object.
        """
        logger.info(f"Downloading file: {file_id}")
        
        request = self.service.files().get_media(fileId=file_id)
        file_content = io.BytesIO()
        downloader = MediaIoBaseDownload(file_content, request)
        
        done = False
        while not done:
            status, done = downloader.next_chunk()
            logger.info(f"Download progress: {int(status.progress() * 100)}%")
        
        file_content.seek(0)
        return file_content
    
    def mark_as_processed(self, file_id: str):
        """
        Mark a file as processed.
        
        Args:
            file_id (str): The ID of the file to mark as processed.
        """
        self.processed_files[file_id] = datetime.datetime.now()
        self._save_processed_files()
        logger.info(f"Marked file as processed: {file_id}")