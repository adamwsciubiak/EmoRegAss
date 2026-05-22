"""
System Orchestrator for the Emotion Regulation Assistant.

This module acts as the Facade for the entire BDI cognitive architecture.
It encapsulates all memory stores (DB1), knowledge bases (DB2), and agents 
(Recognition, Planner, Executor). It enforces the strict data bottleneck, 
ensuring the raw `user_message` only touches the perception input and the final NLG output.

References:
    Pico et al. (2024). Towards an Affective Intelligent Agent Model for Extrinsic Emotion Regulation.
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd

from src.components.emotion_recognition import EmotionRecognitionModel
from src.components.planner import DynamicPlanner
from src.components.executor import Executor
from src.dynamic_knowledge.sensory_memory import SensoryMemory
from src.dynamic_knowledge.episodic_memory import EpisodicMemory
from src.static_knowledge.action_catalog import RegulationAction, EmotionalState

logger = logging.getLogger(__name__)

class EmotionRegulationSystem:
    """
    The main orchestrator uniting all BDI components of the Emotion Regulation Assistant.
    """
    def __init__(
        self,
        emotion_model: EmotionRecognitionModel,
        planner: DynamicPlanner,
        executor: Executor,
        goal_state: EmotionalState
    ):
        self.emotion_model = emotion_model
        self.planner = planner
        self.executor = executor
        
        self.sensory_memory = SensoryMemory()
        self.episodic_memory = EpisodicMemory()
        
        self.goal_state = goal_state
        logger.info("Emotion Regulation System Orchestrator initialized.")

    def calibrate_system(self, personality_traits: Dict[str, float]) -> None:
        """Calibrates the dynamic planner (Q-Table) using the user's personality."""
        self.planner.calibrate(self.goal_state, personality_traits)
        self.sensory_memory.clear() 

    def get_q_table_dataframe(self) -> pd.DataFrame:
        """Returns the Q-Table as a DataFrame for UI visualization with State labels."""
        action_names = [a.name for a in self.planner.action_catalog]
        grid_size = self.planner.grid_size
        
        state_labels = []
        for i in range(grid_size * grid_size):
            arousal_bin = i // grid_size
            valence_bin = i % grid_size
            arousal = (arousal_bin / (grid_size - 1)) * 2 - 1
            valence = (valence_bin / (grid_size - 1)) * 2 - 1
            state_labels.append(f"A: {arousal:.2f} | V: {valence:.2f}")
            
        return pd.DataFrame(self.planner.q_table, columns=action_names, index=state_labels)

    def process_interaction(
        self, 
        user_message: str, 
        personality_traits: Dict[str, float]
    ) -> Tuple[str, Dict[str, Any], Optional[RegulationAction], List[str]]:
        """
        The core cognitive pipeline. Iteratively steps through Observation, Evaluation, and Reaction.
        
        Returns:
            Tuple containing:
            - Final Text Response (str)
            - Emotion Analysis Data (Dict)
            - Chosen Regulation Action (RegulationAction)
            - List of system warnings/fallback notifications (List[str])
        """
        logger.info("--- Starting new interaction cycle ---")
        self.episodic_memory.add_message("user", user_message)
        
        system_warnings = []

        # --- PHASE 1: RECOGNITION (Input Bottleneck) ---
        emotion_analysis = self.emotion_model.analyze_emotion(user_message)
        if self.emotion_model.last_warning:
            system_warnings.append(self.emotion_model.last_warning)
            
        current_state = (emotion_analysis['arousal'], emotion_analysis['valence'])
        logger.info(f"Phase 1 Complete - Recognized State: {current_state}")

        # --- PHASE 2: PLANNING (Cognitive Core) ---
        needs_regulation = self.planner.evaluate(current_state, self.goal_state, self.sensory_memory)
        chosen_strategy = self.planner.strategize(current_state)
        response_plan = self.planner.tactics(chosen_strategy)
        
        # Check if RAG retriever used a fallback
        if self.planner.rag_retriever.last_warning:
            system_warnings.append(self.planner.rag_retriever.last_warning)
            
        self.sensory_memory.update_experience(current_state, chosen_strategy)
        logger.info("Phase 2 Complete - Tactical plan generated.")

        # --- PHASE 3: EXECUTION (Output Bottleneck) ---
        final_response = self.executor.generate_response(
            user_message=user_message,
            emotion_analysis=emotion_analysis,
            personality_traits=personality_traits,
            response_plan=response_plan,
            episodic_memory=self.episodic_memory
        )
        
        self.episodic_memory.add_message("assistant", final_response)
        logger.info("Phase 3 Complete - Empathetic response generated.")
        
        return final_response, emotion_analysis, chosen_strategy, system_warnings