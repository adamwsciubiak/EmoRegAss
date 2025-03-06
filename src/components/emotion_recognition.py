"""
Emotion Recognition Model.

This module provides functionality to analyze text for emotional content,
extracting emotions, valence, and arousal values.
"""

import json
import os
from typing import Dict, Any, List, Optional
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.utils.openai_utils import get_openai_chat_model
from src.config import EMOTION_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionRecognitionModel:
    """
    A model for recognizing emotions in text.
    
    This class uses a language model to analyze text and extract emotional
    content, including specific emotions (happy, sad, angry, etc.), valence
    (positive/negative), and arousal (intensity).
    """
    
    def __init__(self, temperature: float = 0.2):
        """
        Initialize the emotion recognition model.
        
        Args:
            temperature (float, optional): The temperature setting for the model.
                Defaults to 0.2.
        """
        self.llm = get_openai_chat_model(
            temperature=temperature, 
            model_name=os.getenv("EMOTION_MODEL", "gpt-4o-mini")
        )
        
        self.format_instructions = """
        You must respond with a JSON object with the following structure:
        {
            "emotions": {
                "Happy": <float 0-1>,
                "Sad": <float 0-1>,
                "Angry": <float 0-1>,
                "Surprised": <float 0-1>,
                "Fear": <float 0-1>,
                "Disgust": <float 0-1>
            },
            "valence": <float -1 to 1>,
            "arousal": <float -1 to 1>
        }
        
        Where:
        - Each emotion has a score from 0 (not present) to 1 (strongly present)
        - Valence ranges from -1 (very negative) to 1 (very positive)
        - Arousal ranges from -1 (very low intensity) to 1 (very high intensity)
        
        Respond ONLY with the JSON object, no other text.
        """
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert emotion recognition system. 
            Analyze the text provided and identify the emotions expressed.
            
            {format_instructions}
            """),
            ("user", "{text}")
        ])
        
        self.output_parser = StrOutputParser()
    
    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyze the emotional content of the provided text.
        
        Args:
            text (str): The text to analyze for emotional content.
            
        Returns:
            Dict[str, Any]: A dictionary containing:
                - emotions: Dict of emotion names to intensity scores (0-1)
                - valence: Overall positivity/negativity (-1 to 1)
                - arousal: Overall emotional intensity (-1 to 1)
                
        Raises:
            Exception: If there's an error parsing the model response
        """
        logger.info(f"Analyzing emotion in text: {text[:50]}...")
        
        # Create the chain
        chain = self.prompt | self.llm | self.output_parser
        
        # Run the chain
        response = chain.invoke({
            "text": text,
            "format_instructions": self.format_instructions
        })
        
        # Parse the JSON response
        try:
            result = json.loads(response)
            logger.debug(f"Emotion analysis result: {result}")
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing emotion analysis response: {e}")
            logger.error(f"Raw response: {response}")
            raise Exception(f"Failed to parse emotion analysis response: {e}")