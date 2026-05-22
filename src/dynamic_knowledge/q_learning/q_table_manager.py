"""
Q-Table Manager.

This module handles the persistence (saving and loading) of the agent's
learned Q-table. This ensures the Reinforcement Learning progress 
(User's dynamic knowledge) is not lost between sessions or server restarts.
It's sole role is Q-table input/output.
"""

import os
import numpy as np
import pandas as pd
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class QTableManager:
    """
    Manages the disk operations for the RL Q-Table, saving it as a CSV
    for easy inspection and debugging.
    """
    
    def __init__(self, filepath: str = "src/dynamic_knowledge/q_learning/q_table.csv"):
        self.filepath = filepath
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        
    def save_q_table(self, q_table: np.ndarray, action_names: List[str]) -> None:
        """
        Saves the numpy Q-table array to a CSV file.
        Uses action names as column headers for readability.
        """
        try:
            df = pd.DataFrame(q_table, columns=action_names)
            df.to_csv(self.filepath, index=False)
            logger.info(f"Q-Table successfully saved to {self.filepath}")
        except Exception as e:
            logger.error(f"Failed to save Q-Table to disk: {e}")
            
    def load_q_table(self) -> Optional[np.ndarray]:
        """Loads the numpy Q-table array from the CSV file if it exists."""
        if os.path.exists(self.filepath):
            try:
                df = pd.read_csv(self.filepath)
                q_table = df.to_numpy()
                logger.info(f"Q-Table successfully loaded from {self.filepath}")
                return q_table
            except Exception as e:
                logger.error(f"Failed to load Q-Table from disk: {e}. Returning None.")
                return None
        else:
            logger.info(f"No existing Q-Table found at {self.filepath}.")
            return None
            
    def q_table_exists(self) -> bool:
        """Checks if a saved Q-Table exists."""
        return os.path.exists(self.filepath)
        
    def clear_saved_q_table(self) -> None:
        """Deletes the saved Q-table from disk (e.g., on personality change)."""
        if self.q_table_exists():
            try:
                os.remove(self.filepath)
                logger.info(f"Deleted saved Q-Table at {self.filepath}")
            except Exception as e:
                logger.error(f"Failed to delete Q-Table: {e}")