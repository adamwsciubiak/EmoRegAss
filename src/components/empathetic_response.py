"""
Empathetic Response Agent for Emotion Regulation Assistant.

This module provides an Empathetic Response agent that generates empathetic
responses to the user based on their emotional state, personality traits,
and retrieved emotion regulation techniques.
"""

import json
import os
from typing import Dict, Any, List, Optional
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.utils.openai_utils import get_openai_chat_model
from src.config import RESPONSE_MODEL, RESPONSE_TEMPERATURE

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmpatheticResponseAgent:
    """
    An Empathetic Response agent for the Emotion Regulation Assistant.
    
    This class generates empathetic responses to the user based on their
    emotional state, personality traits, and retrieved emotion regulation
    techniques.
    """
    
    def __init__(self, temperature: float = RESPONSE_TEMPERATURE):
        """
        Initialize the Empathetic Response agent.
        
        Args:
            temperature (float, optional): The temperature setting for the model.
                Defaults to RESPONSE_TEMPERATURE from config.
        """
        self.llm = get_openai_chat_model(
            temperature=temperature, 
            model_name=os.getenv("RESPONSE_MODEL", "gpt-4o")
        )
        
        self.response_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an empathetic emotion regulation assistant.
            Your goal is to help users manage their emotions effectively.
            
            Generate a warm, empathetic response that:
            1. Acknowledges the user's emotions
            2. Validates their experience
            3. Offers appropriate emotion regulation techniques
            4. Encourages them to try the techniques
            5. Ends with an open question to continue the conversation
            
            Adapt your tone and approach based on the user's personality traits.
            For example:
            - For users high in openness: Be creative and offer novel perspectives
            - For users high in conscientiousness: Be structured and practical
            - For users high in extraversion: Be energetic and social
            - For users high in agreeableness: Be warm and supportive
            - For users high in neuroticism: Be gentle and reassuring
            
            Follow the response plan provided.
            """),
            ("user", """
            User message: {user_message}
            
            Emotion analysis: {emotion_analysis}
            
            Personality traits (OCEAN model, scale 0-10):
            - Openness: {openness}
            - Conscientiousness: {conscientiousness}
            - Extraversion: {extraversion}
            - Agreeableness: {agreeableness}
            - Neuroticism: {neuroticism}
            
            Response plan: {response_plan}
            
            Emotion regulation techniques: {regulation_techniques}
            
            Chat history:
            {chat_history}
            """)
        ])
        
        self.output_parser = StrOutputParser()
    
    def generate_response(
        self, 
        user_message: str, 
        emotion_analysis: Dict[str, Any],
        personality_traits: Dict[str, int],
        response_plan: Dict[str, Any],
        regulation_techniques: Dict[str, Any],
        chat_history: List[Dict[str, str]]
    ) -> str:
        """
        Generate an empathetic response to the user.
        
        Args:
            user_message (str): The user's message.
            emotion_analysis (Dict[str, Any]): The emotion analysis results.
            personality_traits (Dict[str, int]): The user's personality traits
                on a scale of 0-10.
            response_plan (Dict[str, Any]): The plan for responding to the user.
            regulation_techniques (Dict[str, Any]): The retrieved emotion
                regulation techniques.
            chat_history (List[Dict[str, str]]): The chat history.
                
        Returns:
            str: The generated response.
        """
        logger.info("Generating empathetic response...")
        
        # Format chat history as a string
        formatted_history = ""
        for msg in chat_history[-5:]:  # Use last 5 messages for context
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted_history += f"{role.capitalize()}: {content}\n\n"
        
        # Create the chain
        chain = self.response_prompt | self.llm | self.output_parser
        
        # Run the chain
        response = chain.invoke({
            "user_message": user_message,
            "emotion_analysis": json.dumps(emotion_analysis, indent=2),
            "openness": personality_traits.get("openness", 5),
            "conscientiousness": personality_traits.get("conscientiousness", 5),
            "extraversion": personality_traits.get("extraversion", 5),
            "agreeableness": personality_traits.get("agreeableness", 5),
            "neuroticism": personality_traits.get("neuroticism", 5),
            "response_plan": json.dumps(response_plan, indent=2),
            "regulation_techniques": json.dumps(regulation_techniques, indent=2),
            "chat_history": formatted_history
        })
        
        logger.debug(f"Generated response: {response[:100]}...")
        return response