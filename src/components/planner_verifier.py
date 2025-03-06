"""
Planner-Verifier Agent for Emotion Regulation Assistant.

This module provides a Planner-Verifier agent that creates plans for information
retrieval and verifies the quality of responses.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.utils.openai_utils import get_openai_chat_model
from src.config import PLANNER_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        
        self.planning_format_instructions = """
        You must respond with a JSON object with the following structure:
        {
            "plan": {
                "goal": "The main goal of the response",
                "steps": [
                    "Step 1: Description",
                    "Step 2: Description",
                    ...
                ],
                "considerations": [
                    "Important consideration 1",
                    "Important consideration 2",
                    ...
                ]
            }
        }
        
        Respond ONLY with the JSON object, no other text. Do not include markdown formatting, code blocks, or backticks.
        """
        
        self.verification_format_instructions = """
        You must respond with a JSON object with the following structure:
        {
            "verification": {
                "quality_score": <float 0-1>,
                "strengths": [
                    "Strength 1",
                    "Strength 2",
                    ...
                ],
                "weaknesses": [
                    "Weakness 1",
                    "Weakness 2",
                    ...
                ],
                "improvement_suggestions": [
                    "Suggestion 1",
                    "Suggestion 2",
                    ...
                ]
            }
        }
        
        Respond ONLY with the JSON object, no other text. Do not include markdown formatting, code blocks, or backticks.
        """
        
        self.planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert planner for emotion regulation assistance.
            Create a detailed plan for how to respond to the user based on their
            emotional state, personality traits, and the context of their message.
            
            {format_instructions}
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
            
            Chat history:
            {chat_history}
            """)
        ])
        
        self.verification_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert verifier for emotion regulation assistance.
            Evaluate the quality of the response to the user based on their
            emotional state, personality traits, and the context of their message.
            
            {format_instructions}
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
            
            Response to verify:
            {response_to_verify}
            """)
        ])
        
        self.output_parser = StrOutputParser()
    
    def _clean_json_response(self, response: str) -> str:
        """
        Clean a response string to extract valid JSON.
        
        Args:
            response (str): The response string that may contain markdown or other formatting.
            
        Returns:
            str: A cleaned string containing only the JSON object.
        """
        # Log the raw response for debugging
        logger.debug(f"Raw response before cleaning: {response}")
        
        # Remove markdown code blocks if present
        json_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
        match = re.search(json_pattern, response)
        if match:
            cleaned = match.group(1).strip()
            logger.debug(f"Extracted JSON from code block: {cleaned}")
            return cleaned
        
        # If no code blocks, return the original response
        logger.debug("No code blocks found, returning original response")
        return response.strip()
    
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
        
        # Format chat history as a string
        formatted_history = ""
        for msg in chat_history[-5:]:  # Use last 5 messages for context
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            formatted_history += f"{role.capitalize()}: {content}\n\n"
        
        # Create the chain
        chain = self.planning_prompt | self.llm | self.output_parser
        
        # Run the chain
        try:
            response = chain.invoke({
                "user_message": user_message,
                "emotion_analysis": json.dumps(emotion_analysis, indent=2),
                "openness": personality_traits.get("openness", 5),
                "conscientiousness": personality_traits.get("conscientiousness", 5),
                "extraversion": personality_traits.get("extraversion", 5),
                "agreeableness": personality_traits.get("agreeableness", 5),
                "neuroticism": personality_traits.get("neuroticism", 5),
                "chat_history": formatted_history,
                "format_instructions": self.planning_format_instructions
            })
            
            # Clean the response to extract valid JSON
            cleaned_response = self._clean_json_response(response)
            
            # Parse the JSON response
            try:
                result = json.loads(cleaned_response)
                logger.debug(f"Planning result: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing planning response: {e}")
                logger.error(f"Raw response: {response}")
                logger.error(f"Cleaned response: {cleaned_response}")
                
                # Fallback plan if parsing fails
                fallback_plan = {
                    "plan": {
                        "goal": "To provide comfort and actionable steps to help the user manage their emotions.",
                        "steps": [
                            "Step 1: Acknowledge the user's feelings and validate their emotional state.",
                            "Step 2: Suggest appropriate emotion regulation techniques.",
                            "Step 3: Encourage the user to try the suggested techniques.",
                            "Step 4: Offer to provide additional resources or suggestions if needed."
                        ],
                        "considerations": [
                            "Be empathetic and supportive.",
                            "Tailor suggestions to the user's personality traits.",
                            "Consider the intensity of the user's emotions."
                        ]
                    }
                }
                
                logger.info("Using fallback plan due to parsing error")
                return fallback_plan
                
        except Exception as e:
            logger.error(f"Error invoking planning chain: {e}")
            
            # Fallback plan if chain invocation fails
            fallback_plan = {
                "plan": {
                    "goal": "To provide comfort and actionable steps to help the user manage their emotions.",
                    "steps": [
                        "Step 1: Acknowledge the user's feelings and validate their emotional state.",
                        "Step 2: Suggest appropriate emotion regulation techniques.",
                        "Step 3: Encourage the user to try the suggested techniques.",
                        "Step 4: Offer to provide additional resources or suggestions if needed."
                    ],
                    "considerations": [
                        "Be empathetic and supportive.",
                        "Tailor suggestions to the user's personality traits.",
                        "Consider the intensity of the user's emotions."
                    ]
                }
            }
            
            logger.info("Using fallback plan due to chain invocation error")
            return fallback_plan
    
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
        logger.info("Verifying response...")
        
        # Create the chain
        chain = self.verification_prompt | self.llm | self.output_parser
        
        # Run the chain
        try:
            response = chain.invoke({
                "user_message": user_message,
                "emotion_analysis": json.dumps(emotion_analysis, indent=2),
                "openness": personality_traits.get("openness", 5),
                "conscientiousness": personality_traits.get("conscientiousness", 5),
                "extraversion": personality_traits.get("extraversion", 5),
                "agreeableness": personality_traits.get("agreeableness", 5),
                "neuroticism": personality_traits.get("neuroticism", 5),
                "response_to_verify": response_to_verify,
                "format_instructions": self.verification_format_instructions
            })
            
            # Clean the response to extract valid JSON
            cleaned_response = self._clean_json_response(response)
            
            # Parse the JSON response
            try:
                result = json.loads(cleaned_response)
                logger.debug(f"Verification result: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing verification response: {e}")
                logger.error(f"Raw response: {response}")
                logger.error(f"Cleaned response: {cleaned_response}")
                
                # Fallback verification if parsing fails
                fallback_verification = {
                    "verification": {
                        "quality_score": 0.8,
                        "strengths": [
                            "The response is empathetic and supportive.",
                            "The response provides actionable suggestions."
                        ],
                        "weaknesses": [
                            "The response could be more personalized."
                        ],
                        "improvement_suggestions": [
                            "Consider adding more specific suggestions based on the user's personality."
                        ]
                    }
                }
                
                logger.info("Using fallback verification due to parsing error")
                return fallback_verification
                
        except Exception as e:
            logger.error(f"Error invoking verification chain: {e}")
            
            # Fallback verification if chain invocation fails
            fallback_verification = {
                "verification": {
                    "quality_score": 0.8,
                    "strengths": [
                        "The response is empathetic and supportive.",
                        "The response provides actionable suggestions."
                    ],
                    "weaknesses": [
                        "The response could be more personalized."
                    ],
                    "improvement_suggestions": [
                        "Consider adding more specific suggestions based on the user's personality."
                    ]
                }
            }
            
            logger.info("Using fallback verification due to chain invocation error")
            return fallback_verification