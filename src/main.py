"""
Main application for the Emotion Regulation Assistant.

This module provides the main application class that integrates all components
of the Emotion Regulation Assistant.
"""

import json
import os
from typing import Dict, Any, List, Optional
import logging

from src.components.emotion_recognition import EmotionRecognitionModel
from src.components.rag_agent import RAGAgent
from src.components.planner_verifier import PlannerVerifierAgent
from src.components.empathetic_response import EmpatheticResponseAgent
from src.utils.memory import ChatMemory
from src.utils.vector_store import VectorStoreManager

# Configure logging
log_file = "assistant_logs.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),              # Console logs (keep existing behavior)
        logging.FileHandler(log_file)         # File logs (for Streamlit)
    ]
)
logger = logging.getLogger(__name__)

class EmotionRegulationAssistant:
    """
    The main Emotion Regulation Assistant application.
    
    This class integrates all components of the Emotion Regulation Assistant
    to provide a complete emotion regulation support system.
    """
    
    def __init__(self):
        """Initialize the Emotion Regulation Assistant."""
        logger.info("Initializing Emotion Regulation Assistant...")
        
        # Initialize components
        self.emotion_recognition = EmotionRecognitionModel()
        self.vector_store_manager = VectorStoreManager()
        self.vector_store = self.vector_store_manager.load_or_create()
        self.rag_agent = RAGAgent(vector_store=self.vector_store)
        self.planner_verifier = PlannerVerifierAgent()
        self.response_agent = EmpatheticResponseAgent()
        self.memory = ChatMemory()
        
        # Store the last emotion analysis
        self.last_emotion_analysis = None
        
        logger.info("Emotion Regulation Assistant initialized successfully")
    
    def process_message(self, user_message: str, personality_traits: Dict[str, int]) -> str:
        """
        Process a user message and generate a response.
        
        Args:
            user_message (str): The user's message.
            personality_traits (Dict[str, int]): The user's personality traits
                on a scale of 1-10.
                
        Returns:
            str: The assistant's response.
        """
        logger.info(f"Processing message: {user_message[:200]}...")
        
        # Add user message to memory
        self.memory.add_message("user", user_message)
        
        # Step 1: Analyze emotions in the user's message
        emotion_analysis = self.emotion_recognition.analyze_emotion(user_message)
        self.last_emotion_analysis = emotion_analysis
        logger.info(f"Emotion analysis: {emotion_analysis}")
        
        # Step 2: Create a response plan
        chat_history = self.memory.get_messages()
        response_plan = self.planner_verifier.create_plan(
            user_message=user_message,
            emotion_analysis=emotion_analysis,
            personality_traits=personality_traits,
            chat_history=chat_history
        )
        logger.info(f"Response plan created")
        
        # Step 3: Retrieve relevant emotion regulation techniques
        regulation_techniques = self.rag_agent.get_regulation_techniques(
            user_message=user_message,
            emotion_analysis=emotion_analysis,
            personality_traits=personality_traits
        )
        logger.info(f"Retrieved {len(regulation_techniques.get('techniques', []))} regulation techniques")
        
        # Step 4: Generate an empathetic response
        response = self.response_agent.generate_response(
            user_message=user_message,
            emotion_analysis=emotion_analysis,
            personality_traits=personality_traits,
            response_plan=response_plan,
            regulation_techniques=regulation_techniques,
            chat_history=chat_history
        )
        logger.info(f"Generated response: {response[:200]}...")
        
        # Step 5: Verify the response quality
        verification = self.planner_verifier.verify_response(
            user_message=user_message,
            emotion_analysis=emotion_analysis,
            personality_traits=personality_traits,
            response_to_verify=response
        )
        
        # If the response quality is too low, regenerate it
        quality_score = verification.get("verification", {}).get("quality_score", 0)
        logger.info(f"Response quality score: {quality_score}")
        
        if quality_score < 0.7:
            logger.warning(f"Response quality too low ({quality_score}), regenerating...")
            # Regenerate with more specific instructions based on verification feedback
            weaknesses = verification.get("verification", {}).get("weaknesses", [])
            suggestions = verification.get("verification", {}).get("improvement_suggestions", [])
            
            # Add verification feedback to the response plan
            response_plan["plan"]["considerations"].extend([
                f"Improve: {weakness}" for weakness in weaknesses
            ])
            response_plan["plan"]["considerations"].extend([
                f"Suggestion: {suggestion}" for suggestion in suggestions
            ])
            
            # Regenerate the response
            response = self.response_agent.generate_response(
                user_message=user_message,
                emotion_analysis=emotion_analysis,
                personality_traits=personality_traits,
                response_plan=response_plan,
                regulation_techniques=regulation_techniques,
                chat_history=chat_history
            )
            logger.info(f"Regenerated response: {response[:50]}...")
        
        # Add assistant response to memory
        self.memory.add_message("assistant", response)
        
        return response