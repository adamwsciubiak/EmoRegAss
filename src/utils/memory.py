"""
Memory utilities for the Emotion Regulation Assistant.

This module provides functionality to store and retrieve chat history,
enabling the assistant to maintain context across interactions.
"""

from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChatMemory:
    """
    A class for managing chat history.
    
    This class provides methods to store and retrieve chat messages,
    maintaining context across interactions.
    """
    
    def __init__(self, max_history: int = 20):
        """
        Initialize the chat memory.
        
        Args:
            max_history (int, optional): Maximum number of messages to store.
                Defaults to 20.
        """
        self.messages = []
        self.max_history = max_history
        logger.info(f"Initialized ChatMemory with max_history={max_history}")
    
    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the chat history.
        
        Args:
            role (str): The role of the message sender (e.g., 'user', 'assistant').
            content (str): The content of the message.
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        }
        
        self.messages.append(message)
        
        # Trim history if it exceeds max_history
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
        
        logger.debug(f"Added message from {role}: {content[:50]}...")
    
    def get_messages(self, include_timestamps: bool = False) -> List[Dict[str, str]]:
        """
        Get all messages in the chat history.
        
        Args:
            include_timestamps (bool, optional): Whether to include timestamps
                in the returned messages. Defaults to False.
                
        Returns:
            List[Dict[str, str]]: A list of message dictionaries.
        """
        if include_timestamps:
            return self.messages
        
        # Return messages without timestamps
        return [{k: v for k, v in msg.items() if k != 'timestamp'} 
                for msg in self.messages]
    
    def get_last_n_messages(self, n: int, include_timestamps: bool = False) -> List[Dict[str, str]]:
        """
        Get the last n messages from the chat history.
        
        Args:
            n (int): The number of messages to retrieve.
            include_timestamps (bool, optional): Whether to include timestamps
                in the returned messages. Defaults to False.
                
        Returns:
            List[Dict[str, str]]: A list of the last n message dictionaries.
        """
        messages = self.messages[-n:] if n < len(self.messages) else self.messages
        
        if include_timestamps:
            return messages
        
        # Return messages without timestamps
        return [{k: v for k, v in msg.items() if k != 'timestamp'} 
                for msg in messages]
    
    def clear(self) -> None:
        """Clear all messages from the chat history."""
        self.messages = []
        logger.info("Chat history cleared")