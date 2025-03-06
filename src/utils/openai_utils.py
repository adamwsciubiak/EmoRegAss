"""
Utility functions for interacting with OpenAI models.

This module provides helper functions to create and use OpenAI models
consistently across the application.
"""

from typing import Dict, Any, Optional
import logging
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseChatModel

from src.config import OPENAI_API_KEY, DEFAULT_TEMPERATURE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_openai_client() -> OpenAI:
    """
    Create and return an OpenAI client instance.
    
    Returns:
        OpenAI: An initialized OpenAI client
        
    Raises:
        ValueError: If the OpenAI API key is not set
    """
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    
    return OpenAI(api_key=OPENAI_API_KEY)

def get_openai_chat_model(
    temperature: float = DEFAULT_TEMPERATURE,
    model_name: str = "gpt-4o-mini"
) -> BaseChatModel:
    """
    Create and return a LangChain ChatOpenAI model instance.
    
    Args:
        temperature (float, optional): The temperature setting for the model.
            Higher values make output more random, lower values more deterministic.
            Defaults to DEFAULT_TEMPERATURE.
        model_name (str, optional): The name of the OpenAI model to use.
            Defaults to "gpt-4o-mini".
    
    Returns:
        BaseChatModel: A LangChain ChatOpenAI model instance
        
    Raises:
        ValueError: If the OpenAI API key is not set
    """
    if not OPENAI_API_KEY:
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    
    logger.debug(f"Creating ChatOpenAI model: {model_name} with temperature {temperature}")
    return ChatOpenAI(
        api_key=OPENAI_API_KEY,
        model_name=model_name,
        temperature=temperature
    )