"""
RAG Agent for Emotion Regulation.

This module provides a Retrieval-Augmented Generation agent that retrieves
relevant emotion regulation techniques based on the user's emotional state
and personality traits.
"""

import json
import os
import re
from typing import Dict, Any, List, Optional
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.utils.openai_utils import get_openai_chat_model
from src.config import RAG_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGAgent:
    """
    A Retrieval-Augmented Generation agent for emotion regulation.
    
    This class retrieves relevant emotion regulation techniques based on
    the user's emotional state and personality traits.
    """
    
    def __init__(self, vector_store, temperature: float = 0.2):
        """
        Initialize the RAG agent.
        
        Args:
            vector_store: The vector store containing emotion regulation techniques.
            temperature (float, optional): The temperature setting for the model.
                Defaults to 0.2.
        """
        self.vector_store = vector_store
        self.llm = get_openai_chat_model(
            temperature=temperature, 
            model_name=os.getenv("RAG_MODEL", "gpt-4o")
        )
        
        self.format_instructions = """
        You must respond with a JSON object with the following structure:
        {
            "techniques": [
                {
                    "name": "Technique Name",
                    "description": "Brief description of the technique",
                    "steps": ["Step 1", "Step 2", ...],
                    "effectiveness": <float 0-1>,
                    "personality_match": <float 0-1>,
                    "reasoning": "Why this technique is appropriate"
                },
                ...
            ]
        }
        
        Include 3-5 techniques that are most appropriate for the user's emotional state and personality.
        Respond ONLY with the JSON object, no other text. Do not include markdown formatting, code blocks, or backticks.
        """
        
        self.retrieval_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert emotion regulation assistant.
            Based on the user's emotional state, personality traits, and message,
            retrieve and rank the most appropriate emotion regulation techniques.
            
            Consider:
            1. The specific emotions detected (e.g., anger, sadness, anxiety)
            2. The intensity of the emotions (arousal) and  whether the emotion is experienced as positive or negative (valence)
            3. The user's personality traits (OCEAN model)
            4. The context of the user's message
            
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
            
            Retrieved knowledge:
            {retrieved_documents}
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
    
    def retrieve_relevant_documents(self, query: str, k: int = 3) -> List[str]:
        """
        Retrieve relevant documents from the vector store.
        
        Args:
            query (str): The query to search for.
            k (int, optional): The number of documents to retrieve. Defaults to 5.
            
        Returns:
            List[str]: A list of retrieved document contents.
        """
        logger.info(f"Retrieving documents for query: {query[:0]}...")
        
        docs = self.vector_store.similarity_search(query, k=k)
        return [doc.page_content for doc in docs]
    
    def get_regulation_techniques(
        self, 
        user_message: str, 
        emotion_analysis: Dict[str, Any],
        personality_traits: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Get appropriate emotion regulation techniques for the user.
        
        Args:
            user_message (str): The user's message.
            emotion_analysis (Dict[str, Any]): The emotion analysis results.
            personality_traits (Dict[str, int]): The user's personality traits
                on a scale of 0-10.
                
        Returns:
            Dict[str, Any]: A dictionary containing recommended techniques.
        """
        logger.info("Retrieving emotion regulation techniques...")
        
        # Create a query from the user message and emotion analysis
        emotions_str = ", ".join([f"{e}: {s:.2f}" for e, s in emotion_analysis["emotions"].items() if s > 0.3])
        query = f"User feeling {emotions_str}. Message: {user_message}"
        
        # Retrieve relevant documents
        retrieved_docs = self.retrieve_relevant_documents(query)
        retrieved_text = "\n\n".join(retrieved_docs)
        
        # Create the chain
        chain = self.retrieval_prompt | self.llm | self.output_parser
        
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
                "retrieved_documents": retrieved_text,
                "format_instructions": self.format_instructions
            })
            
            # Clean the response to extract valid JSON
            cleaned_response = self._clean_json_response(response)
            
            # Parse the JSON response
            try:
                result = json.loads(cleaned_response)
                logger.debug(f"Regulation techniques result: {result}")
                return result
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing regulation techniques response: {e}")
                logger.error(f"Raw response: {response}")
                logger.error(f"Cleaned response: {cleaned_response}")
                
                # Fallback techniques if parsing fails
                fallback_techniques = {
                    "techniques": [
                        {
                            "name": "Deep Breathing",
                            "description": "A simple technique to calm the nervous system by taking slow, deep breaths.",
                            "steps": [
                                "Find a quiet place to sit or stand comfortably",
                                "Inhale slowly through your nose for a count of 4",
                                "Hold your breath for a count of 4",
                                "Exhale slowly through your mouth for a count of 6",
                                "Repeat for 5-10 cycles"
                            ],
                            "effectiveness": 0.8,
                            "personality_match": 0.7,
                            "reasoning": "Deep breathing is effective for most people and helps reduce physiological arousal."
                        },
                        {
                            "name": "Progressive Muscle Relaxation",
                            "description": "A technique to reduce physical tension by tensing and then relaxing muscle groups.",
                            "steps": [
                                "Find a comfortable position sitting or lying down",
                                "Start with your feet and work up to your head",
                                "Tense each muscle group for 5 seconds",
                                "Release and relax for 10 seconds",
                                "Notice the difference between tension and relaxation"
                            ],
                            "effectiveness": 0.7,
                            "personality_match": 0.6,
                            "reasoning": "Progressive muscle relaxation helps reduce physical tension associated with negative emotions."
                        },
                        {
                            "name": "Mindfulness Meditation",
                            "description": "A practice of focusing on the present moment without judgment.",
                            "steps": [
                                "Find a quiet place and sit comfortably",
                                "Focus on your breath or a specific sensation",
                                "When your mind wanders, gently bring it back",
                                "Start with 5 minutes and gradually increase",
                                "Practice regularly for best results"
                            ],
                            "effectiveness": 0.8,
                            "personality_match": 0.5,
                            "reasoning": "Mindfulness helps create distance from intense emotions and reduces reactivity."
                        }
                    ]
                }
                
                logger.info("Using fallback techniques due to parsing error")
                return fallback_techniques
                
        except Exception as e:
            logger.error(f"Error invoking retrieval chain: {e}")
            
            # Fallback techniques if chain invocation fails
            fallback_techniques = {
                "techniques": [
                    {
                        "name": "Deep Breathing",
                        "description": "A simple technique to calm the nervous system by taking slow, deep breaths.",
                        "steps": [
                            "Find a quiet place to sit or stand comfortably",
                            "Inhale slowly through your nose for a count of 4",
                            "Hold your breath for a count of 4",
                            "Exhale slowly through your mouth for a count of 6",
                            "Repeat for 5-10 cycles"
                        ],
                        "effectiveness": 0.8,
                        "personality_match": 0.7,
                        "reasoning": "Deep breathing is effective for most people and helps reduce physiological arousal."
                    },
                    {
                        "name": "Progressive Muscle Relaxation",
                        "description": "A technique to reduce physical tension by tensing and then relaxing muscle groups.",
                        "steps": [
                            "Find a comfortable position sitting or lying down",
                            "Start with your feet and work up to your head",
                            "Tense each muscle group for 5 seconds",
                            "Release and relax for 10 seconds",
                            "Notice the difference between tension and relaxation"
                        ],
                        "effectiveness": 0.7,
                        "personality_match": 0.6,
                        "reasoning": "Progressive muscle relaxation helps reduce physical tension associated with negative emotions."
                    },
                    {
                        "name": "Mindfulness Meditation",
                        "description": "A practice of focusing on the present moment without judgment.",
                        "steps": [
                            "Find a quiet place and sit comfortably",
                            "Focus on your breath or a specific sensation",
                            "When your mind wanders, gently bring it back",
                            "Start with 5 minutes and gradually increase",
                            "Practice regularly for best results"
                        ],
                        "effectiveness": 0.8,
                        "personality_match": 0.5,
                        "reasoning": "Mindfulness helps create distance from intense emotions and reduces reactivity."
                    }
                ]
            }
            
            logger.info("Using fallback techniques due to chain invocation error")
            return fallback_techniques