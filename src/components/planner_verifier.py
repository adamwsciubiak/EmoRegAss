"""
Planner-Verifier Agent for Emotion Regulation Assistant.

This module provides a Planner-Verifier agent that creates plans for information
retrieval and verifies the quality of responses.
"""

import json
import os
import re
from typing import Dict, Any, List
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.utils.openai_utils import get_openai_chat_model
from src.utils.format_utils import format_chat_history
from src.config import PLANNER_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydantic models for structured output ---
class ResponsePlan(BaseModel):
    goal: str = Field(description="The main goal of the response.")
    steps: List[str] = Field(description="A list of concrete steps the final response should take.")
    considerations: List[str] = Field(description="Important factors to consider.")

class PlanOutput(BaseModel):
    plan: ResponsePlan

class Verification(BaseModel):
    quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score (0.0 to 1.0).")
    strengths: List[str] = Field(description="What the response does well.")
    weaknesses: List[str] = Field(description="Areas for improvement.")
    improvement_suggestions: List[str] = Field(description="Actionable suggestions for improvement.")

class VerificationOutput(BaseModel):
    verification: Verification


FALLBACK_PLAN = PlanOutput(plan=ResponsePlan(
    goal="To provide comfort and actionable steps to help the user manage their emotions.",
    steps=["Acknowledge feelings.", "Suggest techniques.", "Encourage action.", "Offer more help."],
    considerations=["Be empathetic.", "Tailor to personality.", "Consider emotion intensity."]
))
FALLBACK_VERIFICATION = VerificationOutput(verification=Verification(
    quality_score=0.5,
    strengths=["Response is likely empathetic."],
    weaknesses=["Response could be more personalized or specific."],
    improvement_suggestions=["Ensure the suggestions directly address the user's stated problem."]
))





class PlannerVerifierAgent:
    """
    A Planner-Verifier agent for the Emotion Regulation Assistant.
    
    This class creates plans for information retrieval based on the user's
    emotional state and personality traits, and verifies the quality of responses.
    """
    
    def __init__(self, temperature: float = 0.2):
        """
        Initialize the Planner-Verifier agent.
        
        Args:
            temperature (float, optional): The temperature setting for the model.
                Defaults to 0.2.
        """
        self.llm = get_openai_chat_model(
            temperature=temperature, 
            model_name=os.getenv("PLANNER_MODEL", "gpt-4o")
        )


        # --- Setup for Planning Chain ---
        planning_parser = PydanticOutputParser(pydantic_object=PlanOutput)
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert planner for an empathetic AI assistant. Create a detailed plan to respond to the user based on their situation.\n\n{format_instructions}"),
            ("user", """User message: {user_message}
            Emotion analysis: {emotion_analysis}
            Personality traits (OCEAN 0-10): {personality_traits}
            Recent Chat history:\n{chat_history}""")
        ]).partial(format_instructions=planning_parser.get_format_instructions())
        
        self.planning_chain = planning_prompt | self.llm | planning_parser


        # --- Setup for Verification Chain ---
        verification_parser = PydanticOutputParser(pydantic_object=VerificationOutput)
        verification_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert quality assurance agent for an emotion regulation AI assistant. Evaluate the quality of the generated response.\n\n{format_instructions}"),
            ("user", """User message: {user_message}
            Emotion analysis: {emotion_analysis}
            Personality traits (OCEAN 0-10): {personality_traits}
            RESPONSE TO VERIFY:\n---\n{response_to_verify}\n---""")
        ]).partial(format_instructions=verification_parser.get_format_instructions())

        self.verification_chain = verification_prompt | self.llm | verification_parser
    




    
    def create_plan(
        self, 
        user_message: str, 
        emotion_analysis: Dict[str, Any],
        personality_traits: Dict[str, int],
        chat_history: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Create a plan for responding to the user.
        
        Args:
            user_message (str): The user's message.
            emotion_analysis (Dict[str, Any]): The emotion analysis results.
            personality_traits (Dict[str, int]): The user's personality traits
                on a scale of 0-10.
            chat_history (List[Dict[str, str]]): The chat history.
                
        Returns:
            Dict[str, Any]: A dictionary containing the plan.
        """

        logger.info("Creating response plan...")

        try:
            response = self.planning_chain.with_retry().invoke({
                "user_message": user_message,
                "emotion_analysis": json.dumps(emotion_analysis),
                "personality_traits": json.dumps(personality_traits),
                "chat_history": format_chat_history(chat_history[-5:])
            })
            return response.model_dump()
        except Exception as e:
            logger.error(f"Error in planning chain: {e}", exc_info=True)
            logger.warning("Using fallback plan due to an error.")
            return FALLBACK_PLAN.model_dump()
    
    
    def verify_response(
        self, 
        user_message: str, 
        emotion_analysis: Dict[str, Any],
        personality_traits: Dict[str, int],
        response_to_verify: str
    ) -> Dict[str, Any]:
        """
        Verify the quality of a response to the user.
        
        Args:
            user_message (str): The user's message.
            emotion_analysis (Dict[str, Any]): The emotion analysis results.
            personality_traits (Dict[str, int]): The user's personality traits
                on a scale of 0-10.
            response_to_verify (str): The response to verify.
                
        Returns:
            Dict[str, Any]: A dictionary containing the verification results.
        """
   
        logger.info("Verifying response quality...")
        try:
            response = self.verification_chain.with_retry().invoke({
                "user_message": user_message,
                "emotion_analysis": json.dumps(emotion_analysis),
                "personality_traits": json.dumps(personality_traits),
                "response_to_verify": response_to_verify
            })
            return response.model_dump()
        
        
        except Exception as e:
            logger.error(f"Error in verification chain: {e}", exc_info=True)
            logger.warning("Using fallback verification due to an error.")
            return FALLBACK_VERIFICATION.model_dump()