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

from src.components.pico_planner import PicoPlanner
from src.components.action_executor import ActionExecutor
from src.utils.action_catalog import EmotionalState

from src.utils.memory import ChatMemory
from src.utils.vector_store import VectorStoreManager

# Configure logging (can be simplified, but let's keep it for now)
logger = logging.getLogger(__name__)

# --- We are replacing the class with this function ---

def run_pico_pipeline(
    user_message: str,
    personality_traits: Dict[str, int],
    goal_state: EmotionalState,
    use_rag: bool,
    chat_memory: ChatMemory,
    emotion_model: EmotionRecognitionModel, # This is the generic name, will work with your classifier
    pico_planner: PicoPlanner,
    action_executor: ActionExecutor
) -> (str, Dict[str, Any]):
    
    logger.info(f"Running Pico pipeline... (RAG: {use_rag})")

    # --- Step 1: Analyze Emotion ---
    # NOTE: For now, we use the LLM-based model. You will swap this with your classifier.
    # The classifier should also output the arousal/valence dict.
    emotion_analysis = emotion_model.analyze_emotion(user_message)
    current_state = (emotion_analysis['arousal'], emotion_analysis['valence'])
    logger.info(f"Current emotional state (Sa): {current_state}")

    # --- Step 2: Normalize Personality ---
    # The paper's planner expects personality scores in the [0, 1] range.
    normalized_personality = {k: (v - 1) / 9.0 for k, v in personality_traits.items()}

    # --- Step 3: Select Action with PicoPlanner ---
    chosen_action = pico_planner.select_action(
        current_state=current_state,
        goal_state=goal_state,
        personality=normalized_personality
    )
    
    if not chosen_action:
        logger.warning("PicoPlanner did not return an action. Using a default response.")
        return "I'm not quite sure how to help with that right now, but I'm here to listen.", emotion_analysis

    logger.info(f"PicoPlanner selected action: '{chosen_action.name}'")

    # --- Step 4: Execute Action and Generate Response ---
    # The Executor uses the other agents to formulate the final text.
    final_response = action_executor.execute_action(
        chosen_action=chosen_action,
        user_message=user_message,
        emotion_analysis=emotion_analysis,
        personality_traits=personality_traits,
        chat_memory=chat_memory,
        use_rag=use_rag
    )

    return final_response, emotion_analysis