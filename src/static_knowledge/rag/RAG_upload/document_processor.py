"""
Document Processor for RAG Data Ingestion.

This module is responsible for the semantic chunking and storing of textual data 
into the Vector Database (Supabase). In the BDI cognitive architecture proposed 
by Pico et al. (2024), this process populates the static "Actions Catalog" (DB2).
It transforms monolithic raw therapeutic manuals into retrievable, discrete 
tactical chunks that the Dynamic Planner can query via vector similarity search.
"""

import logging
import uuid
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.utils.vector_store_connector import VectorStoreConnector
import src.static_knowledge.rag.rag_upload.rag_config as rag_cfg

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Orchestrates the transformation of raw text strings into embedded vectors.
    It handles the splitting (chunking) of text to maintain semantic value and 
    delegates the actual database insertion to the VectorStoreConnector.
    """

    def __init__(self):
        """
        Initializes the Document Processor.
        It sets up the database connection and configures the LangChain Text Splitter
        using the strict parameters defined in rag_config.py.
        """
        # Connector handles authentication and connection to Supabase/Postgres
        self.connector = VectorStoreConnector()
        
        # RecursiveCharacterTextSplitter is the industry standard for RAG.
        # It recursively attempts to split text using the defined hierarchy of separators,
        # ensuring that chunks respect logical boundaries (like paragraphs).
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=rag_cfg.CHUNK_SIZE,
            chunk_overlap=rag_cfg.CHUNK_OVERLAP,
            length_function=len,
            separators=rag_cfg.SEPARATORS
        )
        logger.info("DocumentProcessor initialized with parameters from rag_config.")

    def process_and_store(self, text: str, metadata: Dict[str, Any] = None) -> int:
        """
        Splits a document, enriches chunks with metadata, and stores them in the vector database.

        Args:
            text (str): The raw, unbroken document text.
            metadata (Dict[str, Any], optional): Contextual metadata (e.g., source filename, 
                                                 upload date) to attach to the document. 
                                                 Crucial for future metadata filtering during retrieval.

        Returns:
            int: The total number of chunks successfully embedded and stored in the database.
        """
        logger.info("Processing document for vector storage...")
        
        if metadata is None:
            metadata = {}
        
        # Generate a universally unique identifier (UUID) for the parent document.
        # This allows all chunks originating from the same file to be linked together.
        metadata["doc_id"] = str(uuid.uuid4())
        
        # Split the monolithic text string into a list of smaller string chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Document split into {len(chunks)} semantic chunks.")
        
        documents = []
        # Iterate through string chunks to wrap them in LangChain Document objects
        for i, chunk in enumerate(chunks):
            # Create a localized copy of the metadata for this specific chunk
            chunk_metadata = metadata.copy()
            # Assign a sequential chunk ID to preserve the original reading order
            chunk_metadata["chunk_id"] = i
            
            # A LangChain Document binds the text payload with its metadata dictionary
            documents.append(Document(page_content=chunk, metadata=chunk_metadata))
        
        # Load the Supabase vector store interface
        vector_store = self.connector.load_vector_store()
        
        # add_documents automatically handles:
        # 1. Sending the text to the Embedding Model (e.g., text-embedding-3-small)
        # 2. Receiving the vector arrays
        # 3. Inserting the text, metadata, and vectors into the Postgres pgvector table
        vector_store.add_documents(documents)
        logger.info(f"Successfully added {len(documents)} chunks to the vector store.")
        
        return len(documents)