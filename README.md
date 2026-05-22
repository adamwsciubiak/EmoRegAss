# Emotion Regulation Assistant (Pico Architecture Implementation)

A prototype of an "Affective Intelligent AI" powered assistant that helps users regulate their emotions.

*Please note that this is a prototype implementation for a master's thesis project. It is intended for research purposes and it's yet to be polished (pun intended).* **Do NOT share any sensitive information and do NOT use it for serious mental advice.**

This version implements the formal architecture described in the paper: **"Towards an Affective Intelligent Agent Model for Extrinsic Emotion Regulation" by Pico et al. (2024)**. 

It replaces the purely LLM-based planning of original prototypes with a strictly structured, multi-agent cognitive architecture. It features a **dynamic, learning-based planner** that evolves over time through Q-learning, seamlessly integrated with J. Gross's theoretical framework of emotion regulation.

---

### Features

-   **Classifier-Ready Emotion Recognition:** The emotion recognition module maps user input to a 2D Arousal-Valence space (Russell's circumplex model). Designed to be swapped with a formal ML classifier model.
-   **Dynamic Cognitive Planner:** A unified `DynamicPlanner` implementing the exact mathematical models from Pico et al. (2024), enhanced with a stabilizing scaling factor. It executes a 3-step cognitive cycle: Evaluation (Critic), Strategy (Actor), and Tactics (Retrieval).
-   **Persistent Q-Learning:** The agent learns from user interactions. Its policy (Q-Table) is continuously updated and saved as a human-readable CSV, preserving the personalization of the agent across sessions.
-   **Cognitive Memory Separation:** Implements strict separation between **Sensory Memory** (tracking Markov Decision Process states for Reinforcement Learning) and **Episodic Memory** (maintaining conversational context for Natural Language Generation).
-   **Latency-Optimized RAG Engine:** The static knowledge base uses a pure semantic retrieval pipeline (bypassing redundant LLM calls) to fetch psychological technique manuals quickly and securely.
-   **Strict Data Privacy Bottleneck:** The raw user message is shielded from the mathematical and planning cores, entering the system only at the perception (Emotion Recognition) and final actuation (Executor) stages.
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


To upload files to the vector database:
```bash
python "src/static_knowledge/rag/local_to_supabase.py" --source-dir "src/static_knowledge/rag/documents_for_RAG"
```


---

### How It Works (The Cognitive Pipeline)

The system enforces a strict "narrow bottleneck" data flow, conforming to the three-stage architecture proposed by Pico et al. (2024): Perception, Planning, and Execution.

1.  **Phase 1: Recognition (Perception)**
    *   The user enters a message. The `EmotionRecognitionModel` translates the raw text into a quantitative state ($S_a$) in the Arousal-Valence space.
2.  **Phase 2: Dynamic Planning (Cognitive Core)**
    *   *Evaluation (Appraisal):* The planner retrieves the previous state/action from the **Sensory Memory**. It calculates the Euclidean distance to the emotional equilibrium ($S_\epsilon$) to derive a `reward` (Pico, Eq. 1) and updates the Q-Table.
    *   *Strategy (Action Selection):* Using an $\epsilon$-greedy policy, the planner selects the optimal regulation strategy based on the Q-Table and the user's OCEAN personality traits (Pico, Eq. 2, 3, 4).
    *   *Tactics (Retrieval):* The planner queries the **Static Knowledge Base** (RAG) using *only* the strategy name to fetch a dry, step-by-step psychological intervention plan.
3.  **Phase 3: Execution (Actuation)**
    *   The `Executor` agent receives the tactical plan, the conversational history (**Episodic Memory**), and the raw user message. It fuses these elements via a single LLM call to generate a highly contextual, empathetic, and personalized therapeutic response.


---
### Project Structure

The project follows a "Screaming Architecture" paradigm, explicitly separating dynamic user knowledge, static psychological theory, and cognitive processing components.

-   `src/`
    -   `dynamic_knowledge/` *(DB1: The Agent's Belief Base)*
        -   `q_learning/q_table_manager.py`: Handles Q-Table CSV persistence.
        -   `episodic_memory.py`: Tracks conversational history (semantic context).
        -   `sensory_memory.py`: Tracks MDP tuples ($S_{t-1}, A_{t-1}$) for the RL Critic.
    -   `static_knowledge/` *(DB2: The Domain Knowledge)*
        -   `action_catalog.py`: Defines the 5 regulation strategies and their OCEAN correlation weights (Pico, Table 1).
        -   `rag/`: Contains the pure-retrieval RAG engine and database upload scripts.
    -   `components/` *(The Cognitive Flow)*
        -   `emotion_recognition.py`: The perception module.
        -   `planner.py`: The unified Q-Learning & heuristic decision core.
        -   `executor.py`: The Natural Language Generation (NLG) actuator.
    -   `system_orchestrator.py`: The Facade orchestrating the strict data pipeline.
    -   `config.py`: Environment configurations.
-   `app.py`: The lightweight Streamlit UI (completely decoupled from internal state logic).

---

### Future Work & Research Roadmap

This project is under active development. The key areas for future work include:

-   **Language Localization:** The final version of the agent is intended to operate entirely in **Polish**.
-   **Open-Source Model Integration:**
    -   **Emotion Classifier:** Replace the current LLM-based emotion recognition with a dedicated, open-source classifier model that maps text directly to Arousal-Valence values to reduce latency and API costs.
    -   **Executor:** Transition from OpenAI models to open-source models for the response generation component. This includes exploring Polish-language models like `Bielik` or `PLLuM`. Test various approaches for the executor, including using a pre-trained model like "ChatCounselor" (Liu et al., 2023) or fine-tuning a model on similar therapeutic conversation data.
    -   **Vector Database Optimization:** Migrate from the current vector store solution to a fully open-source alternative like `Qdrant`. Tokenization and reranking strategies for the psychological RAG are yet to be improved.
-   **Long-Term Semantic Memory:** Implement a module that summarizes episodic interactions after a session concludes, transferring core insights into a permanent user knowledge graph.
- **RAG upload:** Debug and adjust file paths due to changes in the file structure.
- **RAG literature:** Extend and extract just descriptions of emotion regulation techniques with step-by-step descriptions.
- **Action description:** Add reasoning why the given action is adequate to the given state and user's life expiriences (based on long-term memory) (to boost UX) 

---
### Last updated 
*May 22, 2026*

### License

MIT