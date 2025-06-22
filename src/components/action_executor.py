"""
Orchestrates RAGAgent and EmpatheticResponseAgent to execute a planned action.

Pico suggests implementing a database with simple descriptions of emo.reg. methods, however it'd lead to very poor UX.
For testing this component utilizes RAG for fetching better descriptions.
Later this simple approach will be replaced with more sophisticated database AND/OR with LLM fine-tunned on mental consulting data (such as ChatCounselor/ChatPsychiatrist by Liu et al., 2023)

"""


import logging
from typing import Dict, Any, List

from src.components.rag_agent import RAGAgent
from src.components.empathetic_response import EmpatheticResponseAgent
from src.utils.action_catalog import RegulationAction
from src.utils.memory import ChatMemory

logger = logging.getLogger(__name__)

class ActionExecutor:
    """
    A meta-agent that executes a planned action by orchestrating
    the RAG and Empathetic Response agents.
    """

    def __init__(self, rag_agent: RAGAgent, response_agent: EmpatheticResponseAgent):
        """
        Initializes the executor with the agents it will orchestrate.
        """
        self.rag_agent = rag_agent
        self.response_agent = response_agent
        logger.info("ActionExecutor initialized with RAG and Response agents.")

    def execute_action(
        self,
        chosen_action: RegulationAction,
        user_message: str,
        emotion_analysis: Dict[str, Any],
        personality_traits: Dict[str, int],
        chat_memory: ChatMemory,
        use_rag: bool = True
    ) -> str:
        """
        Generates the final response by running the RAG and response generation pipeline.

        Args:
            chosen_action: The action selected by the PicoPlanner.
            use_rag: A boolean to toggle RAG. If false, we skip the RAG step.
            ... other context needed by the agents.
        
        Returns:
            The final, user-facing empathetic response string.
        """
        logger.info(f"Executor received action '{chosen_action.name}'. RAG: {use_rag}")

        # --- Step 1: Get Regulation Techniques (The RAG Step) ---
        if use_rag:
            logger.info(f"RAG use-case")
            # We use the RAGAgent to get detailed techniques for the *chosen action*
            # The RAGAgent's query should be focused on the chosen action.
            retrieved_techniques = self.rag_agent.get_regulation_techniques(
                user_message=f"Explain the technique: {chosen_action.name}", # A focused query
                emotion_analysis=emotion_analysis,
                personality_traits=personality_traits
            )
        else:
            # If RAG is off, provide a simple placeholder. The EmpatheticResponseAgent
            # will have to rely on its own knowledge.
            retrieved_techniques = {
                "techniques": [{
                    "name": chosen_action.name,
                    "description": "A helpful strategy for managing difficult emotions.",
                    "steps": ["The specific steps for this will depend on the situation."],
                    "reasoning": "This technique was chosen because it matches your current needs."
                }]
            }

        # --- Step 2: Generate the Final Empathetic Response ---
        # We now have all the inputs for our existing EmpatheticResponseAgent.
        # We create a simple, direct plan for it to follow.
        response_plan = {
            "plan": {
                "goal": f"Empathetically introduce and explain the '{chosen_action.name}' technique.",
                "steps": [
                    "Acknowledge the user's feelings from the original message.",
                    f"Introduce '{chosen_action.name}' as a helpful approach.",
                    "Explain the technique using the retrieved information.",
                    "End with an encouraging, open-ended question."
                ]
            }
        }
        
        # We reuse the EmpatheticResponseAgent completely.
        final_response = self.response_agent.generate_response(
            user_message=user_message,
            emotion_analysis=emotion_analysis,
            personality_traits=personality_traits,
            response_plan=response_plan,
            regulation_techniques=retrieved_techniques,
            chat_history=chat_memory.get_messages()
        )
        
        return final_response