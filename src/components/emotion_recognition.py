"""
Emotion Recognition Model.

This module acts as the "Observation" phase in the emotion regulation process 
(Gross, 2015), providing functionality to analyze text for emotional content.
It maps the user's natural language input into James A. Russell's multidimensional 
affective space, extracting specific emotions, valence (pleasant/unpleasant), 
and arousal (activation/stimulation level) as described in Pico et al. (2024), Section 2.2.
"""

import os
from typing import Dict, Any, Optional
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from src.utils.openai_utils import get_openai_chat_model
from src.config import EMOTION_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Emotions(BaseModel):
    """Specific emotion categories mapped from the text."""
    Happy: float = Field(description="Score from 0 (not present) to 1 (strongly present)")
    Sad: float = Field(description="Score from 0 (not present) to 1 (strongly present)")
    Angry: float = Field(description="Score from 0 (not present) to 1 (strongly present)")
    Surprised: float = Field(description="Score from 0 (not present) to 1 (strongly present)")
    Fear: float = Field(description="Score from 0 (not present) to 1 (strongly present)")
    Disgust: float = Field(description="Score from 0 (not present) to 1 (strongly present)")

class EmotionAnalysis(BaseModel):
    """
    Data model for the output of the emotion recognition analysis.
    Represents the individual's current emotional state (S_a) in the 
    Arousal-Valence 2D space (Pico et al., 2024; Figure 3).
    """
    emotions: Emotions = Field(description="Dictionary of emotion names to intensity scores")
    valence: float = Field(ge=-1.0, le=1.0, description="Overall positivity/negativity from -1 to 1")
    arousal: float = Field(ge=-1.0, le=1.0, description="Overall emotional intensity from -1 to 1")

class EmotionRecognitionModel:
    """
    A model for recognizing emotions in text.
    
    This agent simulates the perception of changes in the individual (Pico et al. Section 1).
    It uses a Large Language Model to map text to the (Arousal, Valence) representation.
    """
    
    def __init__(self, temperature: float = 0.2):
        """
        Initialize the emotion recognition model.
        
        Args:
            temperature (float, optional): The temperature setting for the model. Defaults to 0.2.
        """
        self.llm = get_openai_chat_model(
            temperature=temperature, 
            model_name=os.getenv("EMOTION_MODEL", "gpt-4o-mini")
        )

        self.output_parser = PydanticOutputParser(pydantic_object=EmotionAnalysis)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert emotion recognition system. 
            Analyze the text provided and identify the emotions expressed.
            
            {format_instructions}
            """),
            ("user", "{text}")
        ]).partial(format_instructions=self.output_parser.get_format_instructions())
        
        self.chain = self.prompt | self.llm | self.output_parser
        
        # State variable for observability in case of failure
        self.last_warning: Optional[str] = None

    def analyze_emotion(self, text: str) -> Dict[str, Any]:
        """
        Analyze the emotional content of the provided text.
        
        Acts as the first step of the BDI architecture, establishing the 
        agent's Belief about the user's current affective state (S_a).
        
        Args:
            text (str): The raw text/stimulus from the user.
            
        Returns:
            Dict[str, Any]: Parsed dictionary containing 'emotions', 'valence', and 'arousal'.
                            Falls back to neutral values if the LLM fails.
        """
        logger.info(f"Analyzing emotion in text: {text[:200]}...")
        self.last_warning = None # Reset warning state
         
        try:
            response = self.chain.with_retry().invoke({
                "text": text,
            })
            result = response.model_dump()
            logger.debug(f"Emotion analysis result: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error during emotion analysis chain execution: {e}")
            # Graceful Degradation: Set the warning message for the UI
            self.last_warning = "Emotion Recognition API failed. Using a neutral emotional baseline (0.0, 0.0) to continue the session safely."
            # Fallback to a neutral equilibrium state
            return {
                "emotions": {"Happy": 0, "Sad": 0, "Angry": 0, "Surprised": 0, "Fear": 0, "Disgust": 0},
                "valence": 0.0,
                "arousal": 0.0
            }