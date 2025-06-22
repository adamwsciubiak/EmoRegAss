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

from src.utils.plot_utils import create_emotion_trajectory_plot


from src.components.emotion_recognition import EmotionRecognitionModel
from src.components.rag_agent import RAGAgent
from src.components.empathetic_response import EmpatheticResponseAgent
from src.utils.memory import ChatMemory
from src.utils.vector_store_connector import VectorStoreConnector



from src.main import run_pico_pipeline  # We will rename the pipeline function
from src.components.pico_planner import PicoPlanner
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
            # Return the last 'max_lines' lines for more content
            return "".join(lines[-max_lines:])
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error reading log file: {e}\n{error_details}")
        return f"Error reading logs: {e}\n{error_details}"

def clear_logs():
    try:
        # Clear the log file by opening it in write mode
        with open("assistant_logs.log", "w", encoding="utf-8") as log_file:
            pass  # Just open and close in write mode to clear it
        logger.info("Log file cleared")
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error clearing log file: {e}\n{error_details}")






# Set page configuration
st.set_page_config(
    page_title="Emotion Regulation Assistant",
    page_icon="😌",
    layout="wide",
    initial_sidebar_state="expanded"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "personality_traits" not in st.session_state:
    st.session_state.personality_traits = {
        "openness": 5, "conscientiousness": 5, "extraversion": 5,
        "agreeableness": 5, "neuroticism": 5
    }

# Initialize chat memory
if "chat_memory" not in st.session_state:
    st.session_state.chat_memory = ChatMemory()
    logger.info("Initialized ChatMemory.")


if "emotion_recognition_model" not in st.session_state:
    st.session_state.emotion_recognition_model = EmotionRecognitionModel()
    logger.info("Initialized EmotionRecognitionModel.")

# Initialize RAG Agent by first creating a vector store connection
if "rag_agent" not in st.session_state:
    # 1. Create an instance of the new connector.
    connector = VectorStoreConnector()
    # 2. Use it to load the vector store.
    vector_store = connector.load_vector_store()
    # 3. Inject the loaded vector store into the RAGAgent during its initialization.
    st.session_state.rag_agent = RAGAgent(vector_store=vector_store)
    logger.info("Initialized VectorStoreConnector and RAGAgent.")


if "empathetic_response_agent" not in st.session_state:
    st.session_state.empathetic_response_agent = EmpatheticResponseAgent()
    logger.info("Initialized EmpatheticResponseAgent.")


# Initialize the PicoPlanner based on the paper
if "pico_planner" not in st.session_state:
    st.session_state.pico_planner = PicoPlanner()
    logger.info("Initialized PicoPlanner.")

# Initialize the ActionExecutor, reusing the other agents
if "action_executor" not in st.session_state:
    st.session_state.action_executor = ActionExecutor(
        rag_agent=st.session_state.rag_agent,
        response_agent=st.session_state.empathetic_response_agent
    )
    logger.info("Initialized ActionExecutor.")

# Define the user's target emotional state (Sε from the paper)
if "goal_state" not in st.session_state:
    # Using (Arousal, Valence) from the paper's case study, Sec 5.
    st.session_state.goal_state: EmotionalState = goal_state_arousal_valence 
    logger.info(f"Set default goal state to {st.session_state.goal_state}")

# Add a state variable for the RAG toggle
if "use_rag" not in st.session_state:
    st.session_state.use_rag = activateRAG 



if "emotion_analysis" not in st.session_state:
    st.session_state.emotion_analysis = None

if "show_logs" not in st.session_state:
    st.session_state.show_logs = False

if "latest_logs" not in st.session_state:
    st.session_state.latest_logs = read_logs(200)  # Show more log lines

if "processing" not in st.session_state:
    st.session_state.processing = False

if "processing_phase" not in st.session_state:
    st.session_state.processing_phase = "Initializing..."

if "user_message" not in st.session_state:
    st.session_state.user_message = None

# Initialize emotion tracking history
if "valence_history" not in st.session_state:
    st.session_state.valence_history = []

if "arousal_history" not in st.session_state:
    st.session_state.arousal_history = []

# Suggested prompts
SUGGESTED_PROMPTS = [
    "I'm feeling anxious about an upcoming presentation.",
    "I had an argument with my friend and I feel upset.",
    "I'm overwhelmed with work and don't know how to cope.",
    "I'm feeling sad today but I'm not sure why.",
    "I'm excited about a new opportunity but also nervous.",
    "Today was a good day. How can I feel like that more often?"
]

# Function to handle user input
def handle_user_input():
    user_message = st.session_state.user_input
    if user_message:
        # The only job of the callback is to set the state.
        st.session_state.user_message = user_message

# Function to process suggested prompt
def use_suggested_prompt(prompt):
    # Store the prompt in session state instead of directly updating chat history
    st.session_state.user_message = prompt


# Toggle log visibility
def toggle_logs():
    st.session_state.show_logs = not st.session_state.show_logs
    if st.session_state.show_logs:
        st.session_state.latest_logs = read_logs(200)  # Show more log lines

# Function to reset chat
def reset_chat():
    st.session_state.chat_history = []
    st.session_state.emotion_analysis = None
    # Clear emotion tracking history
    st.session_state.valence_history = []
    st.session_state.arousal_history = []
    # Clear logs when resetting chat
    clear_logs()
    st.session_state.latest_logs = ""


# Main app layout
def main():
    # Title
    st.title("Emotion Regulation Assistant")
    
    # Check if we need to process a new message
    if st.session_state.user_message and not st.session_state.processing:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": st.session_state.user_message})
        # Mark as processing
        st.session_state.processing = True
        st.session_state.processing_phase = "Message processing... (might take a while in the current version, sorry! 😌)"
        # Store the message for processing
        user_message = st.session_state.user_message
        # Clear the stored message
        st.session_state.user_message = None
        # Process the message later in the UI flow so the spinner can be displayed
    else:
        user_message = None
    
    # Create a container for the sidebar
    with st.sidebar:
        st.title("Admin Panel")
        st.write("*won't be visible for study participants.*")
        st.session_state.use_rag = st.toggle(
            "Enable RAG for Responses",
            value=st.session_state.use_rag,
            help="ON: The executor uses RAG to generate detailed descriptions. OFF: It uses a simpler placeholder."
        )
        st.title("Personality Profile")
        st.write("*professional scales and/or specialized models will be used later... but for now have fun and adjust the sliders*:")
        
        # OCEAN personality trait sliders - disabled during processing
        if st.session_state.processing:
            # Show disabled sliders with current values when processing
            st.slider("Openness to Experience", 1, 10, st.session_state.personality_traits["openness"],
                    help="High: Creative, curious, and open to new experiences. Low: Conventional, practical, and focused on routine.",
                    disabled=True, key="openness_disabled")
            
            st.slider("Conscientiousness", 1, 10, st.session_state.personality_traits["conscientiousness"],
                    help="High: Organized, responsible, and hardworking. Low: Spontaneous, flexible, and sometimes careless.",
                    disabled=True, key="conscientiousness_disabled")
            
            st.slider("Extraversion", 1, 10, st.session_state.personality_traits["extraversion"],
                    help="High: Outgoing, energetic, and social. Low: Reserved, thoughtful, and prefer solitude.",
                    disabled=True, key="extraversion_disabled")
            
            st.slider("Agreeableness", 1, 10, st.session_state.personality_traits["agreeableness"],
                    help="High: Cooperative, compassionate, and trusting. Low: Critical, analytical, and sometimes competitive.",
                    disabled=True, key="agreeableness_disabled")
            
            st.slider("Neuroticism", 1, 10, st.session_state.personality_traits["neuroticism"],
                    help="High: Sensitive, anxious, and prone to negative emotions. Low: Emotionally stable, calm, and resilient.",
                    disabled=True, key="neuroticism_disabled")
        else:
            # Interactive sliders when not processing
            st.session_state.personality_traits["openness"] = st.slider(
                "Openness to Experience", 1, 10, st.session_state.personality_traits["openness"],
                help="High: Creative, curious, and open to new experiences. Low: Conventional, practical, and focused on routine.",
                key="openness_active"
            )
            
            st.session_state.personality_traits["conscientiousness"] = st.slider(
                "Conscientiousness", 1, 10, st.session_state.personality_traits["conscientiousness"],
                help="High: Organized, responsible, and hardworking. Low: Spontaneous, flexible, and sometimes careless.",
                key="conscientiousness_active"
            )
            
            st.session_state.personality_traits["extraversion"] = st.slider(
                "Extraversion", 1, 10, st.session_state.personality_traits["extraversion"],
                help="High: Outgoing, energetic, and social. Low: Reserved, thoughtful, and prefer solitude.",
                key="extraversion_active"
            )
            
            st.session_state.personality_traits["agreeableness"] = st.slider(
                "Agreeableness", 1, 10, st.session_state.personality_traits["agreeableness"],
                help="High: Cooperative, compassionate, and trusting. Low: Critical, analytical, and sometimes competitive.",
                key="agreeableness_active"
            )
            
            st.session_state.personality_traits["neuroticism"] = st.slider(
                "Neuroticism", 1, 10, st.session_state.personality_traits["neuroticism"],
                help="High: Sensitive, anxious, and prone to negative emotions. Low: Emotionally stable, calm, and resilient.",
                key="neuroticism_active"
            )
        
        # Display emotion analysis if available
        if st.session_state.emotion_analysis:
            st.title("Emotional State (last message)")
            st.write("*For now those values are generated by GPT-4o-mini, but the goal is to used specialized classifier.*")

            
            # Display emotions with bar charts
            emotions = st.session_state.emotion_analysis.get("emotions", {})
            for emotion, score in emotions.items():
                st.progress(score, text=f"{emotion}: {score:.2f}")
            
            # Display valence and arousal
            valence = st.session_state.emotion_analysis.get("valence", 0)
            arousal = st.session_state.emotion_analysis.get("arousal", 0)
            
            st.write(f"Valence (Positive/Negative): {valence:.2f}")
            st.write(f"Arousal (Intensity): {arousal:.2f}")
            
            # Display emotion trajectory plot if we have data
            if st.session_state.valence_history:
                st.title("Emotion Trajectory")
                st.write("Click to fullscren")
                fig = create_emotion_trajectory_plot(
                    st.session_state.valence_history, 
                    st.session_state.arousal_history
                )
                
                if fig:
                    st.pyplot(fig)
        
        # Reset and Toggle Logs buttons - conditionally shown based on processing state
        if st.session_state.processing:
            # Show disabled buttons when processing
            st.button("Reset Chat", disabled=True, key="reset_disabled")
            st.button("Toggle Logs", disabled=True, key="toggle_disabled")
        else:
            # Show active buttons when not processing
            if st.button("Reset Chat", key="reset_active"):
                reset_chat()
            
            if st.button("Toggle Logs", key="toggle_active"):
                toggle_logs()
    
    # Main chat interface
    st.subheader("Share how you're feeling, and the assistant will help you understand and manage emotions through personalized techniques tuned to your emotional state and personality.")
    
    # Show welcome message and description if no chat history
    if not st.session_state.chat_history:
        st.markdown("""*Please note that it's a prototype and although it has access to some mental health related publications, it's yet to be polished (pun intended).* **Do NOT share any sensitive information.**
        
        You can type any message or try one of these prompts to get started:""")
        
        # Display suggested prompts as clickable buttons
        cols = st.columns(2)
        for i, prompt in enumerate(SUGGESTED_PROMPTS):
            col_idx = i % 2
            with cols[col_idx]:
                # Fix for suggestion buttons
                if st.session_state.processing:
                    # Disabled buttons during processing
                    st.button(prompt, disabled=True, key=f"prompt_disabled_{i}", use_container_width=True)
                else:
                    # Active buttons when not processing
                    if st.button(prompt, key=f"prompt_active_{i}", use_container_width=True):
                        use_suggested_prompt(prompt)
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # Add processing spinner if needed
    if st.session_state.processing:
        with st.chat_message("assistant"):
            with st.spinner(st.session_state.processing_phase):
                if user_message:
                    # Process the message with the assistant
                    process_message(user_message)
                    # Force UI refresh after processing
                    st.rerun()
    
    # User input - disabled during processing
    if not st.session_state.processing:
        user_input = st.chat_input(
            "Type your message here...",
            key="user_input",
            on_submit=handle_user_input
        )
    
    # Display logs if show_logs is True
    if st.session_state.show_logs:
        with st.expander("Application Logs", expanded=True):
            st.text_area("Logs", st.session_state.latest_logs, height=300, disabled=True)

# Function to process a message
def process_message(user_message):
    try:
        logger.info(f"Processing user message: {user_message[:200]}...")
        start_time = time.time()
        
        # Call the stateless pipeline function with all necessary components from session_state
        # NEW process_message call
        response, emotion_analysis = run_pico_pipeline(
            user_message=user_message,
            personality_traits=st.session_state.personality_traits,
            goal_state=st.session_state.goal_state,
            use_rag=st.session_state.use_rag,
            chat_memory=st.session_state.chat_memory,
            emotion_model=st.session_state.emotion_recognition_model, # This will be your classifier later
            pico_planner=st.session_state.pico_planner,
            action_executor=st.session_state.action_executor
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Pipeline finished in {processing_time:.2f} seconds")
        
        # Store the latest emotion analysis from the return tuple
        st.session_state.emotion_analysis = emotion_analysis
        logger.info(f"Emotion analysis updated from pipeline: {st.session_state.emotion_analysis}")
        
        # Add assistant response to chat history and memory
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.chat_memory.add_message("assistant", response)
        
        # Update emotion trajectory history
        if st.session_state.emotion_analysis:
            valence = st.session_state.emotion_analysis.get("valence", 0)
            arousal = st.session_state.emotion_analysis.get("arousal", 0)
            
            st.session_state.valence_history.append(valence)
            st.session_state.arousal_history.append(arousal)
            
            logger.info(f"Added to emotion history: valence={valence}, arousal={arousal}")
            logger.info(f"Emotion history length: {len(st.session_state.valence_history)}")
        
        logger.info(f"Response added to history ({len(response)} chars): {response[:100]}...")

    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error in pipeline execution: {e}\n{error_details}")
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"I'm sorry, there was an error processing your message: {str(e)}"
        })
    
    # Reset processing state
    st.session_state.processing = False
    st.session_state.processing_phase = "Initializing..."
    
    if st.session_state.show_logs:
        st.session_state.latest_logs = read_logs(200)

if __name__ == "__main__":
    main()