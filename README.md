Emotion Regulation Assistant


*Please note that it's a prototype and although it has access to some mental health related publications, it's yet to be polished (pun intended).* **Do NOT share any sensitive information, do NOT use it for a serious advice.**
        


An AI-powered assistant that helps users regulate their emotions by providing personalized emotion regulation techniques based on their emotional state and personality traits.

Features

- Emotion recognition from text
- Personalized emotion regulation techniques
- User-configurable personality traits (OCEAN model)
- Interactive chat interface
- Real-time emotion analysis visualization

Installation

1. Clone the repository:
   git clone https://github.com/adamwsciubiak/EmoRegAss.git
   cd emotion-regulation-assistant

2. Install the required packages:
   pip install -r requirements.txt

3. Create a .env file following env_template.txt


Usage

Run the Streamlit app:
streamlit run app.py

Then open your browser and go to http://localhost:8501

Project Structure

- src/: Source code
  - components/: Individual components
    - emotion_recognition.py: Emotion recognition model
    - empathetic_response.py: Empathetic response generator
    - planner_verifier.py: Planner-Verifier agent
    - rag_agent.py: Retrieval-Augmented Generation agent
  - pipeline/: RAG pipeline components
  - utils/: Utility functions
    - openai_utils.py: OpenAI API utilities
    - memory.py: Chat memory management
    - google_drive_utils.py: Google Drive utilities (RAG)
    - pdf_utilis.py: PDF utilities
    - vector_store.py: Vector store management
  - tests/: Test files: not published yet
  - config.py: Configuration file
  - main.py: Main application
- app.py: Streamlit web interface
- assistant_logs.log: Log file for assistant activities
- README.md: Project description and installation instructions
- requirements.txt: Python dependencies
- run_pipeline.py: RAG pipeline
- runtime.txt: Python config for streamlit

How It Works

1. The user enters a message in the chat interface
2. The system analyzes the emotional content of the message
3. Based on the user's emotional state and personality traits, the system retrieves relevant emotion regulation techniques
4. The system generates an empathetic response that incorporates the techniques
5. The response is verified for quality and regenerated if necessary
6. The response is displayed to the user

License

MIT













