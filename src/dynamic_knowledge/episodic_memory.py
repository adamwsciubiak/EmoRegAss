"""
Episodic Memory (current session history) for the Emotion Regulation Assistant.

This module provides functionality to store and retrieve the conversational
history (autobiography of the current session). It is used strictly by the Executor (NLG) 
to maintain context across interactions.
"""

from typing import Dict, List
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class EpisodicMemory:
    """
    A class for managing conversational history.
    It acts as the semantic context for generating empathetic responses.
    """
    
    def __init__(self, max_history: int = 20):
        self.messages = []
        self.max_history = max_history
        logger.info(f"Initialized EpisodicMemory with max_history={max_history}")
    
    def add_message(self, role: str, content: str) -> None:
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        self.messages.append(message)
        
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
            
        logger.debug(f"Added episodic message from {role}: {content[:50]}...")
    
    def get_messages(self, include_timestamps: bool = False) -> List[Dict[str, str]]:
        if include_timestamps:
            return self.messages
        return [{"role": msg["role"], "content": msg["content"]} for msg in self.messages]
    
    def clear(self) -> None:
        self.messages = []
        logger.info("Episodic memory cleared")