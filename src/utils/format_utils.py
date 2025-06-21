"""
Utility functions for formatting data for prompts.
"""
from typing import Dict, List

def format_chat_history(chat_history: List[Dict[str, str]]) -> str:
    """Formats a list of chat messages into a single string for prompts."""
    if not chat_history:
        return "No chat history yet."
    return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat_history])