"""
Streamlit interface for the Emotion Regulation Assistant.

This module provides a web interface for the Emotion Regulation Assistant
using Streamlit.
"""

import streamlit as st 
import os
import json
import time
from dotenv import load_dotenv
import logging
import traceback
import matplotlib.pyplot as plt
import numpy as np

# --- IMPORTS FOR Q-LEARNING ---
from src.components.q_learning_planner import QLearningPlanner
from src.main import run_q_learning_pipeline  # The new, renamed pipeline function

# --- EXISTING IMPORTS ---
from src.utils.plot_utils import create_emotion_trajectory_plot
from src.components.emotion_recognition import EmotionRecognitionModel
from src.components.rag_agent import RAGAgent
from src.components.empathetic_response import EmpatheticResponseAgent
from src.utils.memory import ChatMemory
from src.utils.vector_store_connector import VectorStoreConnector
from src.components.action_executor import ActionExecutor
from src.utils.action_catalog import EmotionalState # For type hinting


# =================== CONFIG ====================================
activateRAG = True 
goal_state_arousal_valence = (0.3, 0.3)


# Load environment variables
load_dotenv()

# Configure logging with more detailed format
log_file = "assistant_logs.log"
# Only initialize logging with 'w' mode when the app first starts, not when rerun
if "logging_initialized" not in st.session_state:
    # Clear log file on initial startup
    with open(log_file, "w", encoding="utf-8") as f:
        pass
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[
            logging.StreamHandler(),  # Console
            logging.FileHandler(log_file, mode='a', encoding='utf-8')  # Changed back to append mode
        ]
    )
    st.session_state.logging_initialized = True

logger = logging.getLogger(__name__)

def read_logs(max_lines=100):
    try:
        with open("assistant_logs.log", "r", encoding="utf-8") as log_file:
            lines = log_file.readlines()
            return "".join(lines[-max_lines:])
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error reading log file: {e}\n{error_details}")
        return f"Error reading logs: {e}\n{error_details}"

def clear_logs():
    try:
        with open("assistant_logs.log", "w", encoding="utf-8") as log_file:
            pass
        logger.info("Log file cleared")
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error clearing log file: {e}\n{error_details}")


# =================== SESSION STATE INITIALIZATION ====================================

st.set_page_config(
    page_title="Emotion Regulation Assistant",
    page_icon="😌",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "personality_traits" not in st.session_state:
    st.session_state.personality_traits = {"openness": 5, "conscientiousness": 5, "extraversion": 5, "agreeableness": 5, "neuroticism": 5}
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ChatMemory()
if "emotion_recognition_model" not in st.session_state:
    st.session_state.emotion_recognition_model = EmotionRecognitionModel()
if "rag_agent" not in st.session_state:
    connector = VectorStoreConnector()
    vector_store = connector.load_vector_store()
    st.session_state.rag_agent = RAGAgent(vector_store=vector_store)
if "empathetic_response_agent" not in st.session_state:
    st.session_state.empathetic_response_agent = EmpatheticResponseAgent()
if "q_planner" not in st.session_state:
    st.session_state.q_planner = QLearningPlanner()
    logger.info("Initialized QLearningPlanner.")
if "action_executor" not in st.session_state:
    st.session_state.action_executor = ActionExecutor(rag_agent=st.session_state.rag_agent, response_agent=st.session_state.empathetic_response_agent)
if "goal_state" not in st.session_state:
    st.session_state.goal_state: EmotionalState = goal_state_arousal_valence
if "use_rag" not in st.session_state:
    st.session_state.use_rag = activateRAG
if "emotion_analysis" not in st.session_state:
    st.session_state.emotion_analysis = None
if "show_logs" not in st.session_state:
    st.session_state.show_logs = False
if "latest_logs" not in st.session_state:
    st.session_state.latest_logs = read_logs(200)
if "processing" not in st.session_state:
    st.session_state.processing = False
if "processing_phase" not in st.session_state:
    st.session_state.processing_phase = "Initializing..."
if "user_message" not in st.session_state:
    st.session_state.user_message = None
if "valence_history" not in st.session_state:
    st.session_state.valence_history = []
if "arousal_history" not in st.session_state:
    st.session_state.arousal_history = []

# --- STATE VARIABLES FOR Q-LEARNING CONTROL ---
if "q_table_initialized" not in st.session_state:
    st.session_state.q_table_initialized = False
if "last_action" not in st.session_state:
    st.session_state.last_action = None
if "last_emotional_state" not in st.session_state:
    st.session_state.last_emotional_state = st.session_state.goal_state
if "show_q_table" not in st.session_state:
    st.session_state.show_q_table = False
# --- NEW: Tracks personality to trigger re-initialization ---
if 'previous_personality_traits' not in st.session_state:
    st.session_state.previous_personality_traits = None


# =================== UI HANDLERS AND FUNCTIONS ====================================
SUGGESTED_PROMPTS = [
    "I'm feeling anxious about an upcoming presentation.", "I had an argument with my friend and I feel upset.",
    "I'm overwhelmed with work and don't know how to cope.", "I'm feeling sad today but I'm not sure why.",
    "I'm excited about a new opportunity but also nervous.", "Today was a good day. How can I feel like that more often?"
]

def handle_user_input():
    user_message = st.session_state.user_input
    if user_message: st.session_state.user_message = user_message

def use_suggested_prompt(prompt):
    st.session_state.user_message = prompt

def toggle_logs():
    st.session_state.show_logs = not st.session_state.show_logs
    if st.session_state.show_logs: st.session_state.latest_logs = read_logs(200)

def toggle_q_table():
    st.session_state.show_q_table = not st.session_state.show_q_table

def reset_chat():
    st.session_state.chat_history = []
    st.session_state.emotion_analysis = None
    st.session_state.valence_history = []
    st.session_state.arousal_history = []
    st.session_state.q_table_initialized = False
    st.session_state.last_action = None
    st.session_state.last_emotional_state = st.session_state.goal_state
    # --- NEW: Reset previous personality tracker on full reset ---
    st.session_state.previous_personality_traits = None
    clear_logs()
    st.session_state.latest_logs = ""


# =================== MAIN APP LAYOUT ====================================
def main():
    st.title("Emotion Regulation Assistant")
    
    if st.session_state.user_message and not st.session_state.processing:
        st.session_state.chat_history.append({"role": "user", "content": st.session_state.user_message})
        st.session_state.processing = True
        st.session_state.processing_phase = "Message processing... (might take a while in the current version, sorry! 😌)"
        user_message_to_process = st.session_state.user_message
        st.session_state.user_message = None
    else:
        user_message_to_process = None
    
    with st.sidebar:
        st.title("Admin Panel")
        st.write("*won't be visible for study participants.*")
        st.session_state.use_rag = st.toggle("Enable RAG for Responses", value=st.session_state.use_rag, help="ON: The executor uses RAG to generate detailed descriptions. OFF: It uses a simpler placeholder.", disabled=st.session_state.processing)
        st.title("Personality Profile")
        st.write("*professional scales and/or specialized models will be used later... but for now have fun and adjust the sliders*:")
        
        # We store the pre-slider values to check for changes
        pre_change_personality = st.session_state.personality_traits.copy()
        
        st.session_state.personality_traits["openness"] = st.slider("Openness to Experience", 1, 10, pre_change_personality["openness"], help="High: Creative, curious, and open to new experiences. Low: Conventional, practical, and focused on routine.", disabled=st.session_state.processing, key="openness_active")
        st.session_state.personality_traits["conscientiousness"] = st.slider("Conscientiousness", 1, 10, pre_change_personality["conscientiousness"], help="High: Organized, responsible, and hardworking. Low: Spontaneous, flexible, and sometimes careless.", disabled=st.session_state.processing, key="conscientiousness_active")
        st.session_state.personality_traits["extraversion"] = st.slider("Extraversion", 1, 10, pre_change_personality["extraversion"], help="High: Outgoing, energetic, and social. Low: Reserved, thoughtful, and prefer solitude.", disabled=st.session_state.processing, key="extraversion_active")
        st.session_state.personality_traits["agreeableness"] = st.slider("Agreeableness", 1, 10, pre_change_personality["agreeableness"], help="High: Cooperative, compassionate, and trusting. Low: Critical, analytical, and sometimes competitive.", disabled=st.session_state.processing, key="agreeableness_active")
        st.session_state.personality_traits["neuroticism"] = st.slider("Neuroticism", 1, 10, pre_change_personality["neuroticism"], help="High: Sensitive, anxious, and prone to negative emotions. Low: Emotionally stable, calm, and resilient.", disabled=st.session_state.processing, key="neuroticism_active")
        
        # --- NEW: LOGIC TO RE-INITIALIZE ON SLIDER CHANGE ---
        if st.session_state.personality_traits != st.session_state.previous_personality_traits:
            with st.spinner("Personality changed. Recalibrating agent..."):
                logger.info("Personality profile changed. Re-initializing Q-Table.")
                normalized_personality = {k: (v - 1) / 9.0 for k, v in st.session_state.personality_traits.items()}
                st.session_state.q_planner.initialize_q_table(
                    goal_state=st.session_state.goal_state,
                    personality=normalized_personality
                )
                st.session_state.q_table_initialized = True
                # Update the tracker to the new state
                st.session_state.previous_personality_traits = st.session_state.personality_traits.copy()
            st.sidebar.success("Agent recalibrated!", icon="✅")
            time.sleep(1.5) # Give user time to see the message
            st.rerun() # Rerun to clear the success message

        if st.session_state.emotion_analysis:
            st.title("Emotional State (last message)")
            st.write("*For now those values are generated by GPT-4o-mini, but the goal is to used specialized classifier.*")
            emotions = st.session_state.emotion_analysis.get("emotions", {})
            for emotion, score in emotions.items(): st.progress(score, text=f"{emotion}: {score:.2f}")
            valence = st.session_state.emotion_analysis.get("valence", 0)
            arousal = st.session_state.emotion_analysis.get("arousal", 0)
            st.write(f"Valence (Positive/Negative): {valence:.2f}")
            st.write(f"Arousal (Intensity): {arousal:.2f}")
            if st.session_state.valence_history:
                st.title("Emotion Trajectory")
                st.write("Click to fullscren")
                fig = create_emotion_trajectory_plot(st.session_state.valence_history, st.session_state.arousal_history)
                if fig: st.pyplot(fig)
        
        if st.button("Reset Chat", on_click=reset_chat, disabled=st.session_state.processing): st.rerun()
        if st.button("Toggle Logs", on_click=toggle_logs, disabled=st.session_state.processing): st.rerun()
        if st.button("Toggle Q-Table View", on_click=toggle_q_table, disabled=st.session_state.processing): st.rerun()

    st.subheader("Share how you're feeling, and the assistant will help you understand and manage emotions through personalized techniques tuned to your emotional state and personality.")
    
    if not st.session_state.chat_history:
        st.markdown("""*Please note that it's a prototype and although it has access to some mental health related publications, it's yet to be polished (pun intended).* **Do NOT share any sensitive information.**\n\nYou can type any message or try one of these prompts to get started:""")
        cols = st.columns(2)
        for i, prompt in enumerate(SUGGESTED_PROMPTS):
            if cols[i % 2].button(prompt, key=f"prompt_{i}", use_container_width=True, disabled=st.session_state.processing):
                use_suggested_prompt(prompt)
                st.rerun()
    
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if st.session_state.processing:
        with st.chat_message("assistant"):
            with st.spinner(st.session_state.processing_phase):
                if user_message_to_process:
                    process_message(user_message_to_process)
                    st.rerun()
    
    st.chat_input("Type your message here...", key="user_input", on_submit=handle_user_input, disabled=st.session_state.processing)
    
    if st.session_state.show_q_table:
        with st.expander("Q-Table State (Live View)", expanded=True):
            if not st.session_state.q_table_initialized: st.warning("Q-Table has not been initialized. Move a personality slider or send a message to calibrate the agent.")
            else:
                st.info("This table shows the agent's learned values for taking each action in a given emotional state. Higher values are better.")
                st.dataframe(st.session_state.q_planner.get_q_table_as_dataframe())
    
    if st.session_state.show_logs:
        with st.expander("Application Logs", expanded=True):
            st.text_area("Logs", st.session_state.latest_logs, height=300, disabled=True)


# =================== MESSAGE PROCESSING ====================================
def process_message(user_message):
    try:
        logger.info(f"Processing user message: {user_message[:200]}...")
        start_time = time.time()
        
        # --- SIMPLIFIED: Initialization is now handled by the sidebar logic ---
        # We just check if it's ready.
        if not st.session_state.q_table_initialized:
            st.error("Agent not calibrated. Please move a personality slider to initialize.")
            return

        response, emotion_analysis, chosen_action = run_q_learning_pipeline(
            q_planner=st.session_state.q_planner, action_executor=st.session_state.action_executor,
            emotion_model=st.session_state.emotion_recognition_model, user_message=user_message,
            prev_emotional_state=st.session_state.last_emotional_state, goal_state=st.session_state.goal_state,
            last_action=st.session_state.last_action, use_rag=st.session_state.use_rag,
            chat_memory=st.session_state.chat_memory, personality_traits=st.session_state.personality_traits
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Pipeline finished in {processing_time:.2f} seconds")
        
        st.session_state.emotion_analysis = emotion_analysis
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.chat_memory.add_message("assistant", response)
        
        if emotion_analysis:
            valence, arousal = emotion_analysis.get("valence", 0), emotion_analysis.get("arousal", 0)
            current_emotional_state = (arousal, valence)
            st.session_state.valence_history.append(valence)
            st.session_state.arousal_history.append(arousal)
            st.session_state.last_emotional_state = current_emotional_state
            st.session_state.last_action = chosen_action
            logger.info(f"Updated history. Next turn's prev_state={current_emotional_state}, last_action={chosen_action.name if chosen_action else 'None'}")

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in pipeline execution: {e}\n{error_details}")
        st.session_state.chat_history.append({"role": "assistant", "content": f"I'm sorry, there was an error processing your message: {e}"})
    
    st.session_state.processing = False
    st.session_state.processing_phase = "Initializing..."
    if st.session_state.show_logs: st.session_state.latest_logs = read_logs(200)

if __name__ == "__main__":
    main()