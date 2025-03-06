"""
Vector store management using Supabase.

This module provides utilities for managing a vector store in Supabase
for storing and retrieving emotion regulation techniques.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import json
import uuid

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from supabase.client import Client, create_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStoreManager:
    """
    Manager for the Supabase vector store.
    
    This class provides methods for creating, loading, and managing
    a vector store in Supabase for emotion regulation techniques.
    """
    
    def __init__(self):
        """Initialize the Vector Store Manager."""
        # Get Supabase credentials from environment variables
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        # Initialize Supabase client
        self.supabase_client = create_client(supabase_url, supabase_key)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()
        
        # Supabase table and column names
        self.table_name = os.getenv("SUPABASE_TABLE", "documents")
        self.content_column = "content"
        self.embedding_column = "embedding"
        self.metadata_column = "metadata"
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        logger.info(f"Vector Store Manager initialized with table: {self.table_name}")
    
    def load_or_create(self) -> SupabaseVectorStore:
        """
        Load the existing vector store or create a new one if it doesn't exist.
        
        Returns:
            SupabaseVectorStore: The loaded or created vector store.
        """
        logger.info("Loading vector store...")
        
        try:
            # Create the vector store with the correct parameters
            # Note: The API has changed, so we need to check the current parameters
            vector_store = SupabaseVectorStore(
                client=self.supabase_client,
                embedding=self.embeddings,
                table_name=self.table_name,
                query_name="match_documents"  # Default query name
            )
            
            logger.info("Vector store loaded successfully")
            return vector_store
        
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise
    
    def process_and_store_document(self, text: str, metadata: Dict[str, Any] = None) -> int:
        """
        Process a document and store it in the vector store.
        
        Args:
            text (str): The document text.
            metadata (Dict[str, Any], optional): Metadata for the document.
                Defaults to None.
                
        Returns:
            int: The number of chunks stored.
        """
        logger.info("Processing document...")
        
        if metadata is None:
            metadata = {}
        
        # Add a unique ID to the metadata
        metadata["doc_id"] = str(uuid.uuid4())
        
        # Split the text into chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split document into {len(chunks)} chunks")
        
        # Create documents
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        # Get the vector store
        vector_store = self.load_or_create()
        
        # Add documents to the vector store
        vector_store.add_documents(documents)
        logger.info(f"Added {len(documents)} chunks to the vector store")
        
        return len(documents)