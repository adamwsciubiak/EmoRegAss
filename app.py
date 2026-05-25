"""
Streamlit Interface for the Emotion Regulation Assistant.

This module provides the graphical user interface for the affective intelligent agent 
proposed by Pico et al. (2024). It separates the visualization into contextual layers:
1. Sidebar: Personality configuration based on the Five-Factor Model (FFM).
2. Main View (Dynamic Columns): A primary chat container and an optional diagnostic column.

References:
    Pico, A., Taverner, J., Vivancos, E., Botti, V., & García-Fornes, A. (2024). 
    Towards an Affective Intelligent Agent Model for Extrinsic Emotion Regulation.
"""

import streamlit as st 
import os
import time
import logging
import traceback
from dotenv import load_dotenv

from src.system_orchestrator import EmotionRegulationSystem
from src.components.emotion_recognition import EmotionRecognitionModel
from src.components.planner import DynamicPlanner
from src.components.executor import Executor
from src.dynamic_knowledge.q_learning.q_table_manager import QTableManager
from src.static_knowledge.rag.rag_retriever import RAGRetriever
from src.utils.vector_store_connector import VectorStoreConnector
from src.utils.plot_utils import create_emotion_trajectory_plot
import src.pipeline_config as cfg

SUGGESTED_PROMPTS = [
    "I'm feeling anxious about an upcoming presentation.", 
    "I had an argument with my friend and I feel upset.",
    "I'm overwhelmed with work and don't know how to cope.", 
    "I'm feeling sad today but I'm not sure why.",
    "I'm excited about a new opportunity but also nervous.", 
    "Today was a good day. How can I feel like that more often?"
]

load_dotenv(override=True)
log_file = "assistant_logs.log"

if "logging_initialized" not in st.session_state:
    with open(log_file, "w", encoding="utf-8") as f: 
        pass 
        
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(file_handler)
    st.session_state.logging_initialized = True

logger = logging.getLogger(__name__)

def read_logs(max_lines: int = 100) -> str:
    try:
        for handler in logging.getLogger().handlers:
            if isinstance(handler, logging.FileHandler):
                handler.flush()
                
        with open(log_file, "r", encoding="utf-8") as f: 
            return "".join(f.readlines()[-max_lines:])
    except Exception as e:
        return f"Error reading logs: {e}"

def initialize_session_state() -> None:
    if "system" not in st.session_state:
        with st.spinner("Initializing Cognitive Architecture..."):
            emotion_model = EmotionRecognitionModel()
            
            # FIX: Przekazujemy ścieżkę z ukrytym plikiem (.q_table.csv), co całkowicie 
            # deaktywuje systemowy Watchdog Streamlita i zapobiega ubijaniu interfejsu.
            hidden_q_filepath = os.path.join("src", "dynamic_knowledge", "q_learning", ".q_table.csv")
            q_manager = QTableManager(filepath=hidden_q_filepath)
            
            vector_store = VectorStoreConnector().load_vector_store()
            rag_retriever = RAGRetriever(vector_store)
            
            planner = DynamicPlanner(
                q_manager=q_manager, 
                rag_retriever=rag_retriever,
                learning_rate=cfg.LEARNING_RATE,
                discount_factor=cfg.DISCOUNT_FACTOR,
                exploration_rate=cfg.EXPLORATION_RATE,
                grid_size=cfg.GRID_SIZE,
                tolerance_threshold=cfg.TOLERANCE_THRESHOLD
            )
            executor = Executor()
            
            st.session_state.system = EmotionRegulationSystem(
                emotion_model=emotion_model,
                planner=planner,
                executor=executor,
                goal_state=cfg.GOAL_STATE_AROUSAL_VALENCE
            )

    if "chat_history" not in st.session_state: st.session_state.chat_history = []
    if "personality_traits" not in st.session_state: st.session_state.personality_traits = {"openness": 5, "conscientiousness": 5, "extraversion": 5, "agreeableness": 5, "neuroticism": 5}
    if "previous_personality" not in st.session_state: st.session_state.previous_personality = None
    if "emotion_analysis" not in st.session_state: st.session_state.emotion_analysis = None
    if "valence_history" not in st.session_state: st.session_state.valence_history = []
    if "arousal_history" not in st.session_state: st.session_state.arousal_history = []
    if "processing" not in st.session_state: st.session_state.processing = False
    if "user_message" not in st.session_state: st.session_state.user_message = None
        
    if "show_q_table" not in st.session_state: st.session_state.show_q_table = False
    if "show_logs" not in st.session_state: st.session_state.show_logs = False
    if "system_error" not in st.session_state: st.session_state.system_error = None
    if "system_warnings" not in st.session_state: st.session_state.system_warnings = []
    
    if "q_table_key" not in st.session_state: st.session_state.q_table_key = 0

def reset_chat() -> None:
    st.session_state.chat_history = []
    st.session_state.emotion_analysis = None
    st.session_state.valence_history = []
    st.session_state.arousal_history = []
    st.session_state.previous_personality = None
    st.session_state.user_message = None
    st.session_state.system_error = None
    st.session_state.system_warnings = []
    st.session_state.system.episodic_memory.clear()
    st.session_state.system.sensory_memory.clear()
    
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.stream.seek(0)
            handler.stream.truncate()
            
    st.session_state.q_table_key += 1

def use_suggested_prompt(prompt: str):
    st.session_state.user_message = prompt

def main():
    st.set_page_config(page_title="Emotion Regulation Assistant", page_icon="😌", layout="wide")
    initialize_session_state()
    
    st.title("Emotion Regulation Assistant")
    st.subheader("Share how you're feeling, and the assistant will help you understand and manage emotions through personalized techniques tuned to your emotional state and personality.")
    
    if st.session_state.user_message and not st.session_state.processing:
        st.session_state.chat_history.append({"role": "user", "content": st.session_state.user_message})
        # Wymuszenie re-asignacji w sesji, by uniknąć problemu braku odświeżenia przez st.rerun() na cloud
        st.session_state.chat_history = st.session_state.chat_history
        st.session_state.processing = True
        st.session_state.system_error = None 
        st.session_state.system_warnings = []
        user_message_to_process = st.session_state.user_message
        st.session_state.user_message = None
    else:
        user_message_to_process = None

    with st.sidebar:
        st.title("Admin Panel")
        st.write("*won't be visible for study participants.*")
        st.title("Personality Profile")
        
        chat_started = len(st.session_state.chat_history) > 0 or st.session_state.processing or st.session_state.user_message is not None
        if chat_started:
            st.info("Personality is locked during an active session. Reset chat to change traits.")
        
        pre_change = st.session_state.personality_traits.copy()
        
        st.session_state.personality_traits["openness"] = st.slider("Openness to Experience", 1, 10, pre_change["openness"], key="o", disabled=chat_started)
        st.session_state.personality_traits["conscientiousness"] = st.slider("Conscientiousness", 1, 10, pre_change["conscientiousness"], key="c", disabled=chat_started)
        st.session_state.personality_traits["extraversion"] = st.slider("Extraversion", 1, 10, pre_change["extraversion"], key="e", disabled=chat_started)
        st.session_state.personality_traits["agreeableness"] = st.slider("Agreeableness", 1, 10, pre_change["agreeableness"], key="a", disabled=chat_started)
        st.session_state.personality_traits["neuroticism"] = st.slider("Neuroticism", 1, 10, pre_change["neuroticism"], key="n", disabled=chat_started)

        if st.session_state.emotion_analysis:
            st.title("Emotional State")
            emotions = st.session_state.emotion_analysis.get("emotions", {})
            for e, s in emotions.items(): 
                st.progress(s, text=f"{e}: {s:.2f}")
            if st.session_state.valence_history:
                st.write("Emotion Trajectory")
                fig = create_emotion_trajectory_plot(st.session_state.valence_history, st.session_state.arousal_history)
                if fig: st.pyplot(fig)
        
        st.button("Reset Chat", on_click=reset_chat, disabled=st.session_state.processing)
        
        st.toggle("Show Q-Table View", key="show_q_table")
        st.toggle("Show Application Logs", key="show_logs")
        
        # Przeniesione na koniec sidebar'a, aby zapobiec czyszczeniu wyżej pokazanych kontrolek przez st.rerun()
        if st.session_state.personality_traits != st.session_state.previous_personality:
            with st.spinner("Calibrating agent to new personality..."):
                normalized = {k: (v - 1) / 9.0 for k, v in st.session_state.personality_traits.items()}
                st.session_state.system.calibrate_system(normalized)
                st.session_state.previous_personality = st.session_state.personality_traits.copy()
            st.session_state.q_table_key += 1  
            st.sidebar.success("Agent recalibrated!", icon="✅")
            time.sleep(1)
            st.rerun()

    show_right_panel = st.session_state.show_q_table or st.session_state.show_logs
    if show_right_panel:
        chat_col, diag_col = st.columns([7, 3], gap="large")
    else:
        _, chat_col, _ = st.columns([1, 8, 1])

    if show_right_panel:
        with diag_col:
            st.markdown("### System Diagnostics")
            if st.session_state.show_q_table:
                st.markdown("**Live Q-Table (Section 4.3)**")
                st.dataframe(
                    st.session_state.system.get_q_table_dataframe().copy(deep=True), 
                    use_container_width=True,
                    key=f"q_table_view_{st.session_state.q_table_key}"
                )
            
            if st.session_state.show_logs:
                st.markdown("**Cognitive Process Logs**")
                st.text_area("Live Logs", read_logs(200), height=500, disabled=True, label_visibility="collapsed")

    with chat_col:
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
                
        if st.session_state.system_warnings:
            for warning in st.session_state.system_warnings:
                st.warning(warning, icon="⚠️")
                
        if st.session_state.system_error:
            st.error(st.session_state.system_error, icon="🚨")
                
        if st.session_state.processing and user_message_to_process:
            with st.chat_message("assistant"):
                with st.spinner("Analyzing and planning intervention..."):
                    try:
                        normalized_pers = {k: (v - 1) / 9.0 for k, v in st.session_state.personality_traits.items()}
                        
                        response, emotion_analysis, action, warnings = st.session_state.system.process_interaction(
                            user_message_to_process, 
                            normalized_pers
                        )
                        
                        st.markdown(response)
                        
                        st.session_state.chat_history.append({"role": "assistant", "content": response})
                        st.session_state.chat_history = st.session_state.chat_history
                        
                        st.session_state.emotion_analysis = emotion_analysis
                        
                        st.session_state.valence_history.append(emotion_analysis.get("valence", 0))
                        st.session_state.valence_history = st.session_state.valence_history
                        
                        st.session_state.arousal_history.append(emotion_analysis.get("arousal", 0))
                        st.session_state.arousal_history = st.session_state.arousal_history
                        
                        st.session_state.system_warnings = warnings
                        
                    except Exception as e:
                        st.session_state.system_error = (
                            "An error occurred while processing your message. Please refresh the page. "
                            "If the error persists, please turn on 'Show Application Logs' in the sidebar, "
                            "copy the logs, and email them to adawsc@st.amu.edu.pl."
                        )
                        logger.error(f"Error during interaction processing: {e}\n{traceback.format_exc()}")
                        
            st.session_state.processing = False
            st.session_state.q_table_key += 1
            st.rerun()

    def submit_chat():
        st.session_state.user_message = st.session_state.chat_widget_input
        st.session_state.system_error = None 
        st.session_state.system_warnings = []

    st.chat_input("Type your message here...", 
                  key="chat_widget_input", 
                  on_submit=submit_chat, 
                  disabled=st.session_state.processing)

if __name__ == "__main__":
    main()