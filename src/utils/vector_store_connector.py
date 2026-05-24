"""
Provides a dedicated connector for loading the Supabase vector store.
This is used by the runtime application and ingestion pipeline for retrieving documents.
"""

import os
import re
import logging
from dotenv import load_dotenv, find_dotenv

# Force load the .env file from the absolute project root
# This prevents Path issues when running scripts from different directories
load_dotenv(find_dotenv(), override=True)

import supabase._sync.client
from supabase.client import create_client
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings

logger = logging.getLogger(__name__)

# ==============================================================================
# MONKEYPATCH: SUPPORT FOR MODERN SUPABASE OPAQUE KEYS (2025/2026)
# The supabase-py library contains a hardcoded regex that assumes all API keys 
# are JWT tokens. Since Supabase introduced 'sb_secret_' and 'sb_publishable_' 
# keys, the library throws an immediate Exception. We patch the regex engine 
# safely to bypass validation exclusively for modern Supabase keys.
# ==============================================================================

_original_match = re.match
_LEGACY_JWT_REGEX = r"^[A-Za-z0-9-_=]+\.[A-Za-z0-9-_=]+\.?[A-Za-z0-9-_.+/=]*$"

def _patched_match(pattern, string, flags=0):
    # If the library is specifically running its JWT validation regex, 
    # and the key is a modern Supabase key (starts with sb_), we force-pass it.
    if pattern == _LEGACY_JWT_REGEX and isinstance(string, str) and string.startswith("sb_"):
        return True
    # Otherwise, behave normally
    return _original_match(pattern, string, flags)

# Apply the patch globally for the current runtime
re.match = _patched_match
# ==============================================================================

class VectorStoreConnector:
    """A dedicated class to connect to and load the Supabase vector store."""

    def __init__(self):
        """Initializes the connector with credentials and embeddings."""
        supabase_url = os.getenv("SUPABASE_URL")
        # Ensure you are using the modern sb_secret_... key here
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in the .env file")
            
        if not supabase_key.startswith("sb_secret_"):
            logger.warning("Your SUPABASE_KEY does not start with 'sb_secret_'. Make sure you are using the modern Secret Key, not the Publishable one.")
        
        # Thanks to the monkeypatch, create_client will now accept the new key!
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