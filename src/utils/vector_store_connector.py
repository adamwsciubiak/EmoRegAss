
"""
Provides a dedicated connector for loading the Supabase vector store.
This is used by the runtime application for retrieving documents.
"""
import os
import logging
from supabase.client import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class VectorStoreConnector:
    """A dedicated class to connect to and load the Supabase vector store."""

    def __init__(self):
        """Initializes the connector with credentials and embeddings."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")
        
        self.supabase_client = create_client(supabase_url, supabase_key)
        self.embeddings = OpenAIEmbeddings()
        self.table_name = os.getenv("SUPABASE_TABLE", "documents")
        
        logger.info(f"VectorStoreConnector initialized for table: {self.table_name}")

    def load_vector_store(self) -> SupabaseVectorStore:
        """
        Loads the existing vector store for querying.

        Returns:
            An instance of SupabaseVectorStore ready for similarity searches.
        """
        logger.info("Loading vector store for querying...")
        try:
            vector_store = SupabaseVectorStore(
                client=self.supabase_client,
                embedding=self.embeddings,
                table_name=self.table_name,
                query_name="match_documents"
            )
            logger.info("Vector store loaded successfully.")
            return vector_store
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            raise