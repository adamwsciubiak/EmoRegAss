"""
RAG Agent for Emotion Regulation.

This module provides a Retrieval-Augmented Generation agent that retrieves
relevant emotion regulation techniques based on the user's emotional state
and personality traits.
"""

import json
import os
from typing import Dict, Any, List
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
# --- We will use RunnableParallel and itemgetter correctly this time ---
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter
from langchain_core.documents import Document
from pydantic import BaseModel, Field

from src.utils.openai_utils import get_openai_chat_model
from src.config import RAG_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Pydantic models for structured output ---
class Technique(BaseModel):
    name: str = Field(description="The name of the emotion regulation technique.")
    description: str = Field(description="A brief, one-sentence description of the technique.")
    steps: List[str] = Field(description="A list of actionable steps to perform the technique.")
    effectiveness: float = Field(ge=0.0, le=1.0, description="Estimated effectiveness for the user's situation (0-1).")
    personality_match: float = Field(ge=0.0, le=1.0, description="How well this technique matches the user's personality (0-1).")
    reasoning: str = Field(description="A brief explanation of why this technique is appropriate.")

class RecommendedTechniques(BaseModel):
    """A data model for a list of recommended emotion regulation techniques."""
    techniques: List[Technique] = Field(description="A list of 3-5 recommended techniques.")


# --- fallback techniques using the Pydantic model ---
FALLBACK_TECHNIQUES = RecommendedTechniques(
    techniques=[
        Technique(name="Deep Breathing", description="A foundational self-regulation method that activates the parasympathetic nervous system, helping to reduce physiological arousal and restore a sense of calm.",  steps=["Inhale slowly and deeply through your nose for a count of 4.", "Gently hold your breath for a count of 4.", "Exhale fully and slowly through your mouth for a count of 6.", "Repeat the cycle 5–10 times, focusing on the rhythm of your breath."], effectiveness=0.5, personality_match=0.5, reasoning="Broadly effective across populations. Helps interrupt acute stress responses and fosters a sense of internal safety by regulating breathing patterns."),
        Technique(name="5-4-3-2-1 Grounding", description="A sensory-based mindfulness practice designed to anchor attention to the present moment and reduce dissociative or anxious symptoms.", steps=["Identify 5 things you can see around you.", "Notice 4 things you can physically feel (e.g., your feet on the floor, the texture of clothing).", "Listen for 3 sounds in your environment.",   "Bring awareness to 2 distinct smells, or recall familiar ones.", "Notice 1 thing you can taste, or imagine a comforting taste if nothing is present."], effectiveness=0.5, personality_match=0.5, reasoning="Particularly helpful during anxiety or emotional flooding. Engaging multiple senses redirects attention away from intrusive thoughts.")
   ]
)

def format_docs(docs: List[Document]) -> str:
    """Helper function to format retrieved documents into a single string."""
    return "\n\n---\n\n".join(doc.page_content for doc in docs)




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
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        self.llm = get_openai_chat_model(
            temperature=temperature, 
            model_name=os.getenv("RAG_MODEL", "gpt-4o")
        )
        self.output_parser = PydanticOutputParser(pydantic_object=RecommendedTechniques)

        self.retrieval_prompt = ChatPromptTemplate.from_template(
            """You are an expert emotion regulation assistant. Your task is to analyze the user's situation and the provided knowledge to recommend the MOST appropriate emotion regulation technique.

You must explain your reasoning for each choice, linking it to the user's specific emotions and personality.

CONTEXT:
User's Message: {user_message}
Emotion Analysis: {emotion_analysis_str}
Personality Profile (OCEAN model, 0-10 scale): {personality_traits_str}

RETRIEVED KNOWLEDGE:
{context}

Based on all the information, select and describe the best technique.

{format_instructions}
            """
        )


        # --- THE PARALLEL CHAIN ---

        # 1. This small runnable creates the retrieval query string. It receives the
        #    initial input dict and outputs only the string.
        retrieval_query_creator = (
            lambda x: f"Techniques for a user feeling {', '.join([f'{e}: {s:.1f}' for e, s in x['emotion_analysis']['emotions'].items() if s > 0.3])}. User's situation: {x['user_message']}"
        )

        # 2. This is the main parallel step. It takes the initial input dictionary and
        #    creates all the variables needed for the final prompt.
        setup_and_retrieval = RunnableParallel(
            # For the 'context' key, we run a sub-chain:
            # 1. Run the query creator
            # 2. Pipe the resulting string to the retriever
            # 3. Pipe the documents to the formatter
            context=retrieval_query_creator | self.retriever | format_docs,

            # For the other keys, we just pull them from the initial input.
            user_message=itemgetter("user_message"),
            emotion_analysis_str=lambda x: json.dumps(x["emotion_analysis"], indent=2),
            personality_traits_str=lambda x: json.dumps(x["personality_traits"], indent=2),
            format_instructions=lambda _: self.output_parser.get_format_instructions(),
        )

        # 3. The final chain is simple: setup -> prompt -> llm -> parse.
        self.rag_chain = setup_and_retrieval | self.retrieval_prompt | self.llm | self.output_parser

    def get_regulation_techniques(
        self, 
        user_message: str, 
        emotion_analysis: Dict[str, Any],
        personality_traits: Dict[str, int]
    ) -> Dict[str, Any]:
        logger.info("Retrieving emotion regulation techniques using parallel RAG chain...")

        try:
            # The invoke call is simple. The chain handles all internal data flow.
            response = self.rag_chain.with_retry().invoke({
                "user_message": user_message,
                "emotion_analysis": emotion_analysis,
                "personality_traits": personality_traits,
            })
            return response.model_dump()
        
        except Exception as e:
            logger.error(f"Error in RAG chain execution: {e}", exc_info=True)
            logger.warning("Falling back to default techniques due to RAG chain error.")
            return FALLBACK_TECHNIQUES.model_dump()
    