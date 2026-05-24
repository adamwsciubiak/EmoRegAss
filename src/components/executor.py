"""
Executor (NLG Agent) for the Emotion Regulation Assistant.

This is the final actuator of the system. It receives the 'dry' tactical plan 
from the Dynamic Planner and fuses it with the raw user_message and episodic memory 
to generate a natural, empathetic therapeutic response.
"""

import json
import os
import logging
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.utils.openai_utils import get_openai_chat_model
from src.utils.format_utils import format_chat_history
from src.config import RESPONSE_TEMPERATURE
from src.dynamic_knowledge.episodic_memory import EpisodicMemory

logger = logging.getLogger(__name__)

class Executor:
    """
    Natural Language Generation (NLG) module. 
    It translates the cognitive plan into human-like empathetic conversation.
    """
    
    def __init__(self, temperature: float = RESPONSE_TEMPERATURE):
        self.llm = get_openai_chat_model(
            temperature=temperature, 
            model_name=os.getenv("RESPONSE_MODEL", "gpt-4o")
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert, empathetic emotion regulation coach. 
            Your primary goal is to execute the 'Therapeutic Plan' provided by the system's cognitive core.

            **Your Core Principles:**
            1.  **Execute the Plan:** The 'Therapeutic Plan' contains the specific emotion regulation technique and knowledge retrieved from the psychological database. You must guide the user through these exact steps.
            2.  **Implicit Adaptation:** Adapt your tone based on the user's personality and emotions, but NEVER explicitly state them. 
                - High neuroticism: Gentle, exceptionally reassuring.
                - High conscientiousness: Structured, logical, step-by-step.
            3.  **Acknowledge & Validate:** ALWAYS start by validating the feelings expressed in the User's Message. Make them feel heard before jumping into the technique.
            4.  **Actionable Guidance:** Translate the dry steps from the Plan into a conversational, real-time walkthrough.
            5.  **Agency:** End with a gentle, open-ended question asking how they feel about trying this, or if they'd prefer a different approach.

            **Crucial Rule:** Do NOT invent new therapeutic techniques. Stick exclusively to the material provided in the 'Therapeutic Plan'.
            """),
            ("user", """--- CONTEXT ---
            User's Message: {user_message}
            Emotion Analysis (For tone adaptation): {emotion_analysis_str}
            Personality Profile (For tone adaptation): {personality_traits_str}
            
            --- RECENT CHAT HISTORY ---
            {chat_history_str}

            --- THERAPEUTIC PLAN TO EXECUTE ---
            {response_plan_str}
            """)
        ])
        
        self.chain = self.prompt | self.llm | StrOutputParser()
        logger.info("Executor (NLG Agent) initialized.")

    def generate_response(
        self, 
        user_message: str, 
        emotion_analysis: Dict[str, Any],
        personality_traits: Dict[str, float],
        response_plan: Dict[str, Any],
        episodic_memory: EpisodicMemory
    ) -> str:
        """
        Fuses the tactical plan with the user's message to generate the final response.
        """
        logger.info("Executor is generating the empathetic response...")
        
        recent_history = episodic_memory.get_messages(include_timestamps=False)[-5:]
        
        try:
            response = self.chain.invoke({
                "user_message": user_message,
                "emotion_analysis_str": json.dumps(emotion_analysis, indent=2),
                "personality_traits_str": json.dumps(personality_traits, indent=2),
                "response_plan_str": json.dumps(response_plan, indent=2),
                "chat_history_str": format_chat_history(recent_history)
            })

            # NAPRAWA: Zmiana logger.debug na logger.info oraz zapis całego stringa do plików
            logger.info(f"Response generated successfully:\n{response}")
            return response
            
        except Exception as e:
            logger.error(f"Error during response generation: {e}", exc_info=True)
            raise e