# src/pipeline/document_processor.py

"""
Handles processing and storing of documents into the vector store.
This is used exclusively by the data ingestion pipeline.
"""

import logging
import uuid
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# We reuse the connector to get access to the store
from src.utils.vector_store_connector import VectorStoreConnector

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """A dedicated class for processing text and adding it to the vector store."""

    def __init__(self):
        """Initializes the processor and its text splitter."""
        # It uses the connector to get a writable instance of the vector store
        self.connector = VectorStoreConnector()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        logger.info("DocumentProcessor initialized.")

    def process_and_store(self, text: str, metadata: Dict[str, Any] = None) -> int:
        """
        Splits a document, creates chunks, and stores them in the vector store.

        Args:
            text: The document text to process.
            metadata: A dictionary of metadata to attach to the document.

        Returns:
            The number of chunks that were successfully stored.
        """
        logger.info(f"Processing document for storage...")
        
        if metadata is None:
            metadata = {}
        
        # Add a unique ID for the entire document
        metadata["doc_id"] = str(uuid.uuid4())
        
        # Split the text into chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split document into {len(chunks)} chunks.")
        
        # Create LangChain Document objects for each chunk
        documents = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = i
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        # Load a vector store instance that we can write to
        vector_store = self.connector.load_vector_store()
        
        # Add the new documents to Supabase
        vector_store.add_documents(documents)
        logger.info(f"Successfully added {len(documents)} chunks to the vector store.")
        
        return len(documents)