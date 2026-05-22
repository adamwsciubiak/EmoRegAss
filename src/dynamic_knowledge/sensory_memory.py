"""
Sensory Memory (MDP Buffer) for the Emotion Regulation Assistant.

From a cognitive architecture perspective, this acts as an ultra-short-term memory (sensory buffer).
It stores the Markov Decision Process (MDP) tuple from the previous time-step (t-1).
Specifically, it remembers the user's previous emotional state (S_t-1) and the 
action the agent took (A_t-1) so that the Dynamic Planner's Evaluator can calculate 
the reward when observing the new state (S_t).
"""

from typing import Optional, Tuple
import logging
from src.static_knowledge.action_catalog import EmotionalState, RegulationAction

logger = logging.getLogger(__name__)

class SensoryMemory:
    """
    Manages the short-term state-action buffer for the RL Evaluator.
    """
    
    def __init__(self):
        self.previous_state: Optional[EmotionalState] = None
        self.previous_action: Optional[RegulationAction] = None
        logger.info("Initialized SensoryMemory (MDP Buffer).")
        
    def update_experience(self, state: EmotionalState, action: RegulationAction) -> None:
        """
        Stores the state and action that just occurred (t-1).
        """
        self.previous_state = state
        self.previous_action = action
        logger.debug(f"Sensory Memory updated: S_t-1={state}, A_t-1='{action.name}'")
        
    def get_last_experience(self) -> Tuple[Optional[EmotionalState], Optional[RegulationAction]]:
        """
        Retrieves the S_{t-1} and A_{t-1} needed for Reward calculation.
        """
        return self.previous_state, self.previous_action
        
    def clear(self) -> None:
        """Flushes the sensory buffer (e.g., on hard reset)."""
        self.previous_state = None
        self.previous_action = None
        logger.info("Sensory Memory cleared.")