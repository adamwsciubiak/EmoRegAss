# src/main.py

"""
Main application logic for the Emotion Regulation Assistant.

This module provides the core orchestration pipeline that integrates all components.
"""

import json
from typing import Dict, Any, List, Optional
import logging

from src.components.emotion_recognition import EmotionRecognitionModel
from src.components.rag_agent import RAGAgent
from src.components.planner_verifier import PlannerVerifierAgent
from src.components.empathetic_response import EmpatheticResponseAgent
from src.utils.memory import ChatMemory
from src.utils.vector_store import VectorStoreManager

# Configure logging (can be simplified, but let's keep it for now)
logger = logging.getLogger(__name__)

# --- We are replacing the class with this function ---

def run_emotion_regulation_pipeline(
    user_message: str,
    personality_traits: Dict[str, int],
    chat_memory: ChatMemory,
    emotion_recognition_model: EmotionRecognitionModel,
    rag_agent: RAGAgent,
    planner_verifier_agent: PlannerVerifierAgent,
    empathetic_response_agent: EmpatheticResponseAgent
    ) -> tuple[str, Dict[str, Any]]:  # Return a tuple: (response, analysis)
    """
    Orchestrates the full emotion regulation pipeline.
    This function is stateless and relies on inputs for all operations.
    """
    logger.info(f"Processing message in pipeline: {user_message[:200]}...")

    # The memory is now managed by the caller (app.py)
    # The caller will add the user message before calling this function.

    # Step 1: Analyze emotions in the user's message
    emotion_analysis = emotion_recognition_model.analyze_emotion(user_message)
    logger.info(f"Emotion analysis: {emotion_analysis}")

    # Step 2: Create a response plan
    chat_history = chat_memory.get_messages()
    response_plan = planner_verifier_agent.create_plan(
        user_message=user_message,
        emotion_analysis=emotion_analysis,
        personality_traits=personality_traits,
        chat_history=chat_history
    )
    logger.info(f"Response plan created")

    # Step 3: Retrieve relevant emotion regulation techniques
    regulation_techniques = rag_agent.get_regulation_techniques(
        user_message=user_message,
        emotion_analysis=emotion_analysis,
        personality_traits=personality_traits
    )
    logger.info(f"Retrieved {len(regulation_techniques.get('techniques', []))} regulation techniques")

    # Step 4: Generate an empathetic response
    response = empathetic_response_agent.generate_response(
        user_message=user_message,
        emotion_analysis=emotion_analysis,
        personality_traits=personality_traits,
        response_plan=response_plan,
        regulation_techniques=regulation_techniques,
        chat_history=chat_history
    )
    logger.info(f"Generated response: {response[:200]}...")

    # Step 5: Verify the response quality
    verification = planner_verifier_agent.verify_response(
        user_message=user_message,
        emotion_analysis=emotion_analysis,
        personality_traits=personality_traits,
        response_to_verify=response
    )

    quality_score = verification.get("verification", {}).get("quality_score", 0)
    logger.info(f"Response quality score: {quality_score}")

    if quality_score < 0.7:
        logger.warning(f"Response quality too low ({quality_score}), regenerating...")
        # (This part remains the same, just using the agent passed as an argument)
        weaknesses = verification.get("verification", {}).get("weaknesses", [])
        suggestions = verification.get("verification", {}).get("improvement_suggestions", [])
        
        response_plan["plan"]["considerations"].extend([
            f"Improve: {weakness}" for weakness in weaknesses
        ])
        response_plan["plan"]["considerations"].extend([
            f"Suggestion: {suggestion}" for suggestion in suggestions
        ])
        
        response = empathetic_response_agent.generate_response(
            user_message=user_message,
            emotion_analysis=emotion_analysis,
            personality_traits=personality_traits,
            response_plan=response_plan,
            regulation_techniques=regulation_techniques,
            chat_history=chat_history
        )
        logger.info(f"Regenerated response: {response[:50]}...")

    # The caller (app.py) will be responsible for adding the assistant's response to memory.
    
    # Return both results explicitly
    return response, emotion_analysis