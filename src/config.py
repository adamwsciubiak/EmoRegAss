"""
Configuration settings for the Emotion Regulation Assistant.

This module contains all configuration parameters including API keys,
model names, and other settings used throughout the application.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# OpenAI API configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMOTION_MODEL = os.getenv("EMOTION_MODEL", "gpt-4o-mini")
PERSONALITY_MODEL = os.getenv("PERSONALITY_MODEL", "gpt-4o-mini")
RAG_MODEL = os.getenv("RAG_MODEL", "gpt-4o")
PLANNER_MODEL = os.getenv("PLANNER_MODEL", "gpt-4o")
RESPONSE_MODEL = os.getenv("RESPONSE_MODEL", "gpt-4o")

# Model parameters
DEFAULT_TEMPERATURE = 0.2
RESPONSE_TEMPERATURE = 0.7

# Vector database configuration
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vector_store")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")