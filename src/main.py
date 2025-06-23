# src/main.py

"""
Main application logic for the Emotion Regulation Assistant.

This module provides the core orchestration pipeline that integrates all components,
now featuring the Q-Learning dynamic planner.
"""

import json
from typing import Dict, Any, List, Optional, Tuple
import logging
import numpy as np

# --- NEW IMPORTS ---
from src.components.q_learning_planner import QLearningPlanner
from src.utils.action_catalog import RegulationAction

# --- EXISTING IMPORTS ---
from src.components.emotion_recognition import EmotionRecognitionModel
from src.components.action_executor import ActionExecutor
from src.utils.action_catalog import EmotionalState
from src.utils.memory import ChatMemory

logger = logging.getLogger(__name__)

# --- NEW: Reward Calculation Function ---
def calculate_reward(
    prev_state: EmotionalState,
    next_state: EmotionalState,
    goal_state: EmotionalState
) -> float:
    """
    Calculates the reward based on the change in distance to the goal state.

    Reference: Pico et al. (2024), Section 4.3. The reward is based on the
    "success or failure" of an action, which we define as moving closer to
    the emotional equilibrium Sε.
    """
    dist_before = np.sqrt((prev_state[0] - goal_state[0])**2 + (prev_state[1] - goal_state[1])**2)
    dist_after = np.sqrt((next_state[0] - goal_state[0])**2 + (next_state[1] - goal_state[1])**2)
    
    # Reward is the reduction in distance. A positive value means we got closer.
    reward = dist_before - dist_after
    return reward

# --- RENAMED and RESTRUCTURED Pipeline Function ---

def run_q_learning_pipeline(
    q_planner: QLearningPlanner,
    action_executor: ActionExecutor,
    emotion_model: EmotionRecognitionModel,
    user_message: str,
    prev_emotional_state: EmotionalState,
    goal_state: EmotionalState,
    last_action: Optional[RegulationAction],
    use_rag: bool,
    chat_memory: ChatMemory,
    personality_traits: Dict[str, int]
) -> Tuple[str, Dict[str, Any], RegulationAction]:
    """
    Orchestrates the full Q-Learning interaction loop.
    
    Returns:
        A tuple containing (final_response, emotion_analysis, chosen_action).
    """
    
    logger.info(f"Running Q-Learning pipeline... (RAG: {use_rag})")

    # --- Step 1: Analyze Emotion (Observation) ---
    emotion_analysis = emotion_model.analyze_emotion(user_message)
    current_emotional_state = (emotion_analysis['arousal'], emotion_analysis['valence'])
    logger.info(f"Current emotional state (s'): {current_emotional_state}")
    
    # --- Step 2: Update Q-Table (Learning) ---
    # This happens *before* selecting the next action. We learn from the outcome
    # of the *previous* turn's action.
    if last_action is not None:
        # Calculate the reward for the transition from prev_state -> current_state
        reward = calculate_reward(
            prev_state=prev_emotional_state,
            next_state=current_emotional_state,
            goal_state=goal_state
        )
        logger.info(f"Calculated reward: {reward:.4f}")

        # Update the Q-table with the observed transition and reward
        q_planner.update_q_table(
            prev_state=prev_emotional_state,
            action=last_action,
            reward=reward,
            next_state=current_emotional_state
        )
        logger.info(f"Q-Table updated for s={prev_emotional_state}, a='{last_action.name}'")
    else:
        logger.info("First turn, skipping Q-table update.")

    # --- Step 3: Select Next Action (Decision-Making) ---
    chosen_action = q_planner.select_action(
        current_state=current_emotional_state
    )
    
    if not chosen_action:
        logger.warning("QLearningPlanner did not return an action. Using a default response.")
        # We still return a "None" action to avoid breaking the calling function
        return "I'm not quite sure how to help with that right now, but I'm here to listen.", emotion_analysis, None

    logger.info(f"QLearningPlanner selected action: '{chosen_action.name}'")

    # --- Step 4: Execute Action and Generate Response ---
    final_response = action_executor.execute_action(
        chosen_action=chosen_action,
        user_message=user_message,
        emotion_analysis=emotion_analysis,
        personality_traits=personality_traits,
        chat_memory=chat_memory,
        use_rag=use_rag
    )

    return final_response, emotion_analysis, chosen_action