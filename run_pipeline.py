"""
RAG Agent for Emotion Regulation.

This module provides a Retrieval-Augmented Generation agent that retrieves
relevant emotion regulation techniques based on the user's emotional state
and personality traits.
"""

import json
import os
from typing import Dict, Any, List, Optional
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.utils.openai_utils import get_openai_chat_model
from src.utils.vector_store import VectorStoreManager
from src.config import RAG_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAgent:
    """
    A Retrieval-Augmented Generation agent for emotion regulation.
    
    This class retrieves relevant emotion regulation techniques based on
    the user's emotional state and personality traits.
    """
    
    def __init__(self, temperature: float = 0.2):
        """
        Initialize the RAG agent.
        
        Args:
            temperature (float, optional): The temperature setting for the model.
                Defaults to 0.2.
        """
        # Initialize the vector store manager
        vector_store_manager = VectorStoreManager()
        self.vector_store = vector_store_manager.load_or_create()
        
        # Initialize the language model
        self.llm = get_openai_chat_model(
            temperature=temperature, 
            model_name=os.getenv("RAG_MODEL", "gpt-4o")
        )
        
        self.format_instructions = """
        You must respond with a JSON object with the following structure:
        {
            "techniques": [
                {
                    "name": "Technique Name",
                    "description": "Brief description of the technique",
                    "steps": ["Step 1", "Step 2", ...],
                    "effectiveness": <float 0-1>,
                    "personality_match": <float 0-1>,
                    "reasoning": "Why this technique is appropriate"
                },
                ...
            ]
        }
        
        Include 3-5 techniques that are most appropriate for the user's emotional state and personality.
        Respond ONLY with the JSON object, no other text.
        """
        
        self.retrieval_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert emotion regulation assistant.
            Based on the user's emotional state, personality traits, and message,
            retrieve and rank the most appropriate emotion regulation techniques.
            
            Consider:
            1. The specific emotions detected (e.g., anger, sadness, anxiety)
            2. The intensity of the emotions (arousal)
            3. The user's personality traits (OCEAN model)
            4. The context of the user's message
            
            {format_instructions}
            """),
            ("user", """
            User message: {user_message}
            
            Emotion analysis: {emotion_analysis}
            
            Personality traits (OCEAN model, scale 0-10):
            - Openness: {openness}
            - Conscientiousness: {conscientiousness}
            - Extraversion: {extraversion}
            - Agreeableness: {agreeableness}
            - Neuroticism: {neuroticism}
            
            Retrieved knowledge:
            {retrieved_documents}
            """)
        ])
        
        self.output_parser = StrOutputParser()
        
        logger.info("RAG Agent initialized")
    
    def retrieve_relevant_documents(self, query: str, k: int = 5) -> List[str]:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            query (str): The query to search for.
            k (int, optional): The number of documents to retrieve. Defaults to 5.
            
        Returns:
            List[str]: A list of retrieved document contents.
        """
        logger.info(f"Retrieving documents for query: {query[:50]}...")
        
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def get_regulation_techniques(
        self, 
        user_message: str, 
        emotion_analysis: Dict[str, Any],
        personality_traits: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Get appropriate emotion regulation techniques for the user.
        
        Args:
            user_message (str): The user's message.
            emotion_analysis (Dict[str, Any]): The emotion analysis results.
            personality_traits (Dict[str, int]): The user's personality traits
                on a scale of 0-10.
                
        Returns:
            Dict[str, Any]: A dictionary containing recommended techniques.
                
        Raises:
            Exception: If there's an error parsing the model response
        """
        logger.info("Retrieving emotion regulation techniques...")
        
        # Create a query from the user message and emotion analysis
        emotions_str = ", ".join([f"{e}: {s:.2f}" for e, s in emotion_analysis["emotions"].items() if s > 0.3])
        query = f"User feeling {emotions_str}. Message: {user_message}"
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_relevant_documents(query)
        retrieved_text = "\n\n".join(retrieved_docs)
        
        # Create the chain
        chain = self.retrieval_prompt | self.llm | self.output_parser
        
        # Run the chain
        response = chain.invoke({
            "user_message": user_message,
            "emotion_analysis": json.dumps(emotion_analysis, indent=2),
            "openness": personality_traits.get("openness", 5),
            "conscientiousness": personality_traits.get("conscientiousness", 5),
            "extraversion": personality_traits.get("extraversion", 5),
            "agreeableness": personality_traits.get("agreeableness", 5),
            "neuroticism": personality_traits.get("neuroticism", 5),
            "retrieved_documents": retrieved_text,
            "format_instructions": self.format_instructions
        })
        
        # Parse the JSON response
        try:
            result = json.loads(response)
            logger.debug(f"Regulation techniques result: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing regulation techniques response: {e}")
            logger.error(f"Raw response: {response}")
            raise Exception(f"Failed to parse regulation techniques response: {e}")