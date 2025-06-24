# Emotion Regulation Assistant (Pico Architecture Implementation)

A prototype of an "Affective Intelligent AI" powered assistant that helps users regulate their emotions.

*Please note that this is a prototype implementation for a master's thesis project. It is intended for research purposes and it's yet to be polished (pun intended).* **Do NOT share any sensitive information and do NOT use it for serious mental advice.**

This version implements the formal architecture described in the paper: **"Towards an Affective Intelligent Agent Model for Extrinsic Emotion Regulation" by Pico et al. (2024)**.

It replaces the purely LLM-based planning of the original prototype with a deterministic `PicoPlanner` that selects actions based on user emotional state, a target equilibrium, and personality traits, as detailed in the paper. The most recent version also features a **dynamic, learning-based planner** that evolves over time through Q-learning, as detailed in the paper (however this aspect still needs some love).

---

### Features

-   **Classifier-Ready Emotion Recognition:** The emotion recognition module is designed to be swapped with a formal classifier model.
-   **Dynamic, Learning-Based Planning:** Implements a `PicoPlanner` enhanced with a `QLearningManager`. The agent's strategy begins with a "warm start" based on the paper's static formulas and then **learns and adapts** from user interactions via Q-learning.
-   **Toggleable RAG Executor:** The `ActionExecutor` can generate responses using either static, pre-defined templates or a dynamic RAG pipeline for detailed technique descriptions.
-   User-configurable personality traits (OCEAN model).
-   Interactive chat interface with real-time emotion trajectory visualization.

### Installation

1.  Clone the repository and switch to this branch:
    ```bash
    git clone https://github.com/adamwsciubiak/EmoRegAss.git
    cd EmoRegAss
    git checkout Legit_implementation_of_Pico_pipeline_workbranch
    ```

2.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3.  Create a `.env` file by copying `env_template.txt` and fill in your API keys.

### Usage

Run the Streamlit app:
```bash
streamlit run app.py
```
Then open your browser and go to `http://localhost:8501`. Use the "Developer Controls" in the sidebar to toggle RAG functionality.

---

### How It Works

1.  The user enters a message in the chat interface.
2.  The system determines the user's current emotional state (`Sa`) (Arousal/Valence).
3.  **Learning Step:** If this is not the first turn, the system calculates a `reward` based on how the last action affected the user's emotional state (i.e., did it move closer to the goal state `Sε`?). It uses this reward to update its internal **Q-table**.
4.  **Planning Step:** The `PicoPlanner` queries the `QLearningManager`. IT calculates the optimal regulation action by evaluating each option against the user's current state (`Sa`), goal state (`Sε`), and personality profile (`t`), Using an epsilon-greedy strategy on the Q-table, the manager selects the optimal action for the current state.
5.  The chosen action (e.g., "Distraction") is passed to the `ActionExecutor`.
6.  The `ActionExecutor` optionally uses the `RAGAgent` to retrieve detailed information about the chosen action.
7.  It then uses the `EmpatheticResponseAgent` to synthesize this information into a final, user-facing message.
8.  The response is displayed to the user.


---
### Project Structure

The architecture is modular, separating the deterministic planner from the response generation logic.

-   `src/`
    -   `components/`: Core runtime components of the agent (`pico_planner`, `action_executor`, etc.).
    -   `utils/`: Shared utility functions.
        -   `RAG upload/`: Scripts for the offline data ingestion pipeline (`drive_to_supabase`, etc.).
        -   `action_catalog.py`: Defines the regulation actions and their properties as per the paper.
-   `app.py`: The Streamlit web interface.
-   `main.py`: Contains the core `run_pico_pipeline` orchestration logic.

---

### Future Work & Research Roadmap

This project is under active development. The key areas for future work include:

-   **Persistence:** Save and load the agent's learned Q-table between sessions to create a truly personalized, long-term assistant.
-   **Language Localization:** The final version of the agent is intended to operate entirely in **Polish**.
-   **Open-Source Model Integration:**
    -   **Emotion Classifier:** Replace the current LLM-based emotion recognition with a dedicated, open-source classifier model that maps text directly to Arousal-Valence values.
    -   **Executor:** Transition from OpenAI models to open-source models for the response generation and RAG components. This includes exploring Polish-language models like `Bielik` or `PLLuM`.Test various approaches for the executor, including using a pre-trained model like "ChatCounselor" (Liu et al., 2023), fine-tuning a model on similar therapeutic conversation data...
    -   **RAG and Vector Database:** Migrate from the current vector store solution to a fully open-source alternative like `Qdrant`. The data ingestion pipeline needs adaptation after changes. Tokenizing is also yet to be improved.
    -   **Content Improvement:** Refine the `action_executor`'s internal plans, improve the static descriptions for RAG-less runs, and enhance the RAG knowledge base with more detailed documents on regulation techniques.

---


### License

MIT