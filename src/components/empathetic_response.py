"""
Empathetic Response Agent for Emotion Regulation Assistant.

This module provides an Empathetic Response agent that generates empathetic
responses to the user based on their emotional state, personality traits,
and retrieved emotion regulation techniques.
"""

import json
import os
from typing import Dict, Any, List
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.utils.openai_utils import get_openai_chat_model
from src.utils.format_utils import format_chat_history
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
            ("system", """You are an expert, empathetic emotion regulation coach. Your primary goal is to execute a response plan perfectly, making the user feel heard, validated, and skillfully guided.

            **Your Core Principles:**
            1.  **Strictly Follow the Plan:** The 'Response Plan to Follow' is your script. You MUST adhere to its goal, steps, and considerations. The 'Regulation Techniques to Offer' is the content you will use to fill in the steps of that plan.
            2.  **Implicit Adaptation:** You MUST adapt your tone and style based on the user's personality and emotions, but you must NEVER explicitly mention the traits or emotions you've detected. This adaptation is critical. For example:
            - If personality indicates high neuroticism, your tone must be exceptionally gentle and reassuring.
            - If personality indicates high conscientiousness, your language must be structured and logical.
            This adaptation should be subtly woven into your word choice, NOT explicitly mentioned.
            3.  **One Primary Technique:** Your main task is to select and present ONE primary technique from the provided list. This should be the most appropriate one for the user's current situation.

            **Response Structure and Content:**
            1.  **Acknowledge & Validate:** Begin by executing the first steps of the plan, which typically involve acknowledging and validating the user's feelings.
            2.  **Introduce the Primary Technique:** Seamlessly introduce the single best technique from the provided list.
            3.  **Explain the "Why":** Provide a clear, simple explanation of how the technique works and why it is effective for the kinds of feelings the user is experiencing.
            4.  **Provide Detailed Step-by-Step Guidance:** Give clear, actionable, step-by-step instructions for the technique. Write them as if you are guiding the user through the exercise in real-time.
            5.  **Offer Alternatives:** After detailing the primary technique, briefly name the other techniques from the list as clear alternatives. This gives the user agency.
            6.  **Encourage & Inquire:** End the response with an open-ended question that asks the user for direction. This should give them an explicit choice on how to proceed. For example: "So, that's the [Primary Technique Name]. We could also explore [Alternative 1] or [Alternative 2]. Would you like a more detailed walkthrough of [Primary Technique Name], or would you prefer to hear about one of the other options first?"

            **Handling Follow-up Questions:**
            If the user asks for more details about a technique you have already described, DO NOT repeat the same information. Instead, provide a "deeper dive." This means offering additional details, metaphors, examples, or tips for common difficulties. Treat it as a chance to elaborate, not to repeat.

            **Crucial Rule:** Your main focus is the detailed explanation of the ONE primary technique. The mention of alternatives should be a brief, single sentence to empower the user, not a list of other options.
            """),



                # The user prompt with the context is also slightly updated for clarity.
                ("user", """CONTEXT FOR YOUR INTERNAL USE:
            - User's Message: {user_message}
            - Emotion Analysis: {emotion_analysis_str}
            - Personality Profile: {personality_traits_str}
            - Recent Chat History:
            {chat_history_str}

            YOUR SCRIPT AND CONTENT:
            - **Response Plan to Follow (Your Script):** {response_plan_str}
            - **Regulation Techniques to Offer (Your Content - CHOOSE ONE as primary):** {regulation_techniques_str}
            """)
        ])




        
        self.output_parser = StrOutputParser()
        # --- Response chain ---
        self.response_chain = self.response_prompt | self.llm | self.output_parser
    
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
        
        try:
            # Populate the prompt template.
            response = self.response_chain.invoke({
                "user_message": user_message,
                # Convert complex objects to JSON strings for the prompt
                "emotion_analysis_str": json.dumps(emotion_analysis, indent=2),
                "personality_traits_str": json.dumps(personality_traits, indent=2),
                "response_plan_str": json.dumps(response_plan, indent=2),
                "regulation_techniques_str": json.dumps(regulation_techniques, indent=2),
                "chat_history_str": format_chat_history(chat_history[-5:]) # Use shared utility
            })

            logger.debug(f"Generated response: {response[:100]}...")
            return response
        
        except Exception as e:
            logger.error(f"Error in response generation chain: {e}", exc_info=True)
            return "I'm very sorry, but I encountered an error while trying to formulate a response. Could you perhaps rephrase your last message?"