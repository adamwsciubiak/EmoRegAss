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

# Import the Emotion Regulation Assistant
from src.main import EmotionRegulationAssistant

# Set page configuration
st.set_page_config(
    page_title="Emotion Regulation Assistant",
    page_icon="😌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "personality_traits" not in st.session_state:
    st.session_state.personality_traits = {
        "openness": 5,
        "conscientiousness": 5,
        "extraversion": 5,
        "agreeableness": 5,
        "neuroticism": 5
    }

if "assistant" not in st.session_state:
    st.session_state.assistant = EmotionRegulationAssistant()
    logger.info("Initialized EmotionRegulationAssistant instance")

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
        # Store the message in session state for processing
        st.session_state.user_message = user_message
        # This forces streamlit to rerun the script
        st.rerun()

# Function to process suggested prompt
def use_suggested_prompt(prompt):
    # Store the prompt in session state instead of directly updating chat history
    st.session_state.user_message = prompt
    # This forces streamlit to rerun the script
    st.rerun()

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

# Function to create emotion trajectory plot
def create_emotion_trajectory_plot():
    if not st.session_state.valence_history:
        return None
    
    valence = st.session_state.valence_history
    arousal = st.session_state.arousal_history
    
    # If we only have one data point, duplicate it to avoid plot error
    if len(valence) == 1:
        # Use a slightly adjusted point to show direction
        valence = [valence[0], valence[0]]
        arousal = [arousal[0], arousal[0]]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the points
    ax.scatter(valence, arousal, color='royalblue', zorder=3)
    
    # Connect the points with a line to show trajectory
    ax.plot(valence, arousal, linestyle='--', color='gray', zorder=2)
    
    # Annotate each point with its index (step number)
    for i, (x, y) in enumerate(zip(valence, arousal)):
        ax.annotate(f'{i+1}', (x, y), textcoords="offset points", xytext=(5, 5), ha='center', fontsize=9)
    
    # Add quadrant lines (neutral valence and arousal at 0.5)
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    
    # Set axis limits and labels
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    
    # Optional: Add grid and background color for better visibility
    ax.grid(True, linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f7f7f7')
    
    # Add quadrant labels
    # ax.text(0.25, 0.75, "Distressed", ha='center', va='center', fontsize=9)
    # ax.text(0.75, 0.75, "Excited", ha='center', va='center', fontsize=9)
    # ax.text(0.25, 0.25, "Depressed", ha='center', va='center', fontsize=9)
    # ax.text(0.75, 0.25, "Relaxed", ha='center', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

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
                fig = create_emotion_trajectory_plot()
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
        # Update processing phase
        logger.info(f"Processing user message: {user_message[:200]}...")

        # Process the message with the assistant
        start_time = time.time()
        
        # Processing phases

        response = st.session_state.assistant.process_message(
            user_message, 
            st.session_state.personality_traits
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Message processed in {processing_time:.2f} seconds")
        
        # Store the latest emotion analysis
        st.session_state.emotion_analysis = st.session_state.assistant.last_emotion_analysis
        logger.info(f"Emotion analysis updated: {st.session_state.emotion_analysis}")
        
        # Add assistant response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Update emotion trajectory history
        if st.session_state.emotion_analysis:
            valence = st.session_state.emotion_analysis.get("valence", 0.5)
            arousal = st.session_state.emotion_analysis.get("arousal", 0.5)
            
            # Add to history
            st.session_state.valence_history.append(valence)
            st.session_state.arousal_history.append(arousal)
            
            logger.info(f"Added to emotion history: valence={valence}, arousal={arousal}")
            logger.info(f"Emotion history length: {len(st.session_state.valence_history)}")
        
        # Log detailed information about the response
        logger.info(f"Response generated ({len(response)} chars): {response[:100]}...")
        if st.session_state.emotion_analysis:
            emotions_str = ", ".join([f"{k}: {v:.2f}" for k, v in 
                                   st.session_state.emotion_analysis.get("emotions", {}).items()])
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Error processing message: {e}\n{error_details}")
        st.session_state.chat_history.append({
            "role": "assistant", 
            "content": f"I'm sorry, there was an error processing your message: {str(e)}"
        })
    
    # Reset processing state
    st.session_state.processing = False
    st.session_state.processing_phase = "Initializing..."
    
    # Update logs
    if st.session_state.show_logs:
        st.session_state.latest_logs = read_logs(200)  # Show more log lines

if __name__ == "__main__":
    main()