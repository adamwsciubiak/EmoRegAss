# Emotion Regulation Assistant (LLM-based Prototype)

A prototype of an "Affective Intelligent AI" powered assistant that helps users regulate their emotions.

*Please note that this is a prototype. While it has access to some mental health related publications, it is still under development.* **Do NOT share any sensitive information and do NOT use it for serious medical advice.**

This version uses a chain of Large Language Models (LLMs) to orchestrate planning, retrieval, and response generation, serving as the initial prototype for the project.

> **Note on Project Development:**
> This `main` branch contains the original LLM-driven prototype. A newer version, which more closely implements the formal architecture described in the **Pico et al. (2024)** paper using a deterministic planner, is under active development on a separate branch.
>
> You can find this new, theory-grounded version on the **`Legit_implementation_of_Pico_pipeline_workbranch`** branch.

---

### Features

-   **LLM-based Reasoning:** Uses a chain of LLM agents for emotion recognition, planning, and response generation.
-   **Retrieval-Augmented Generation (RAG):** Retrieves personalized emotion regulation techniques from a knowledge base.
-   User-configurable personality traits (OCEAN model).
-   Interactive chat interface with real-time emotion analysis visualization.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/adamwsciubiak/EmoRegAss.git
    cd EmoRegAss
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
Then open your browser and go to `http://localhost:8501`.

---

### How It Works

1.  The user enters a message in the chat interface.
2.  An `EmotionRecognition` agent analyzes the emotional content.
3.  A `PlannerVerifier` agent creates a high-level plan for the response.
4.  A `RAGAgent` retrieves relevant emotion regulation techniques.
5.  An `EmpatheticResponse` agent generates a response synthesizing the plan and techniques.
6.  The `PlannerVerifier` can trigger a self-correction loop if the response quality is low.

---

### Project Structure

-   `src/`
    -   `components/`: Core runtime components of the agent (`planner_verifier`, `rag_agent`, etc.).
    -   `utils/`: Shared utility functions.
        -   `RAG upload/`: Scripts for the offline data ingestion pipeline (`drive_to_supabase`, etc.).
-   `app.py`: The Streamlit web interface.
-   `main.py`: Contains the core orchestration logic for the LLM agent pipeline.

---

### Future Work & Research Roadmap

The insights from this LLM-based prototype are informing the primary development track of this project, which aims to build a more formal, theory-grounded agent. I aim to possibly develop this (LLM-based) approach further in the future, but for now that part of the project is postponed.
The key areas for current improvements are being implemented on the `Legit_implementation_of_Pico_pipeline_workbranch` and include:

-   **Q-Learning Implementation:** The highest priority is to implement the Q-learning-based planner improvement and personalization layer, as described in Section 4.3 of the Pico et al. paper. This will allow the agent to learn from user interactions and dynamically adapt its strategy.
-   **Language Localization:** The final version of the agent is intended to operate entirely in **Polish**.
-   **Open-Source Model Integration:**
    -   **Emotion Classifier:** Replace the current LLM-based emotion recognition with a dedicated, open-source classifier model that maps text directly to Arousal-Valence values.
    -   **Executor:** Transition from OpenAI models to open-source models for the response generation and RAG components. This includes exploring Polish-language models like `Bielik` or `PLLuM`.Test various approaches for the executor, including using a pre-trained model like "ChatCounselor" (Liu et al., 2023), fine-tuning a model on similar therapeutic conversation data...
    -   **RAG and Vector Database:** Migrate from the current vector store solution to a fully open-source alternative like `Qdrant`. The data ingestion pipeline needs adaptation after changes. Tokenizing is also yet to be improved.
    -   **Content Improvement:** Refine the `action_executor`'s internal plans, improve the static descriptions for RAG-less runs, and enhance the RAG knowledge base with more detailed documents on regulation techniques.

---


### License

MIT