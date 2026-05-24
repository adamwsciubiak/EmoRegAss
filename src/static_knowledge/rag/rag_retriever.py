"""
RAG Retriever for Emotion Regulation.

This module acts purely as a semantic search engine (Static Knowledge Base / DB2).
It represents the "Actions Catalog" of specific sub-strategies needed by the Dynamic 
Planner to convert abstract strategies (e.g., Cognitive Change) into real actions 
(Pico et al., 2024; Section 4). It retrieves the most relevant therapeutic guidelines 
from the vector store based on the strategy name.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Fallback texts (Action definitions) in case the vector store is empty or fails.
FALLBACK_MANUALS = {
    "Distraction": "Focus your attention on a neutral or highly engaging external stimulus. For example, count backwards from 100 by 7s, or list 5 things you can see around you.",
    "Reappraisal": "Try to view the situation from a third-person perspective. Ask yourself: 'What is another, more neutral way to interpret this event? What would I tell a friend in this situation?'",
    "Deep Breathing": "Inhale slowly and deeply through your nose for a count of 4. Hold for 4. Exhale fully through your mouth for 6. Repeat 5 times.",
    "Default": "Follow general therapeutic guidelines: acknowledge the emotion, breathe deeply, and allow yourself time before reacting."
}

class RAGRetriever:
    """
    A pure retrieval agent that queries the static knowledge base (Vector Store).
    Does not employ LLM generation to avoid latency bottlenecks.
    """
    
    def __init__(self, vector_store):
        """
        Initialize the RAG retriever.
        
        Args:
            vector_store: The LangChain vector store connected to the therapeutic database.
        """
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        self.last_warning: Optional[str] = None
        logger.info("RAGRetriever initialized (Pure Retrieval mode, no LLM).")

    def retrieve_technique_manual(self, strategy_name: str) -> List[str]:
        """
        Queries the vector database for a specific regulation strategy.
        
        This enables the "Tactical" planning phase, translating an abstract regulation 
        action (α) into specific steps (φ) (Pico et al., 2024; Section 4). 
        Notice: User message is NOT passed here, keeping the semantic query clean.
        
        Args:
            strategy_name (str): The name of the technique (e.g., 'Distraction').
            
        Returns:
            List[str]: A list of retrieved text chunks containing instructions.
                       Returns static fallbacks on failure.
        """
        logger.info(f"Retrieving manual for strategy: {strategy_name}")
        self.last_warning = None # Reset warning state
        
        try:
            query = f"Psychological emotion regulation technique: {strategy_name}. Step-by-step instructions and reasoning."
            docs = self.retriever.invoke(query)
            
            if not docs:
                logger.warning(f"No documents found for {strategy_name}. Using fallback.")
                self.last_warning = f"The knowledge base contained no documents for '{strategy_name}'. Reverting to a basic static manual."
                fallback_text = FALLBACK_MANUALS.get(strategy_name, FALLBACK_MANUALS["Default"])
                return [fallback_text]
                
            # Logowanie odzyskanych snippetów z bazy danych (RAG Chunks) do pliku/terminala
            snippets = "\n".join([f"Chunk {i+1}: {doc.page_content[:150]}..." for i, doc in enumerate(docs)])
            logger.info(f"Successfully retrieved {len(docs)} chunks from database. Previews:\n{snippets}")
                
            return [doc.page_content for doc in docs]
            
        except Exception as e:
            logger.error(f"Error during vector store retrieval: {e}", exc_info=True)
            self.last_warning = f"Connection to the Vector Database failed while fetching '{strategy_name}'. A basic static manual was used instead."
            fallback_text = FALLBACK_MANUALS.get(strategy_name, FALLBACK_MANUALS["Default"])
            return [fallback_text]