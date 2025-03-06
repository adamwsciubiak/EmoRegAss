Emotion Regulation Assistant

An AI-powered assistant that helps users regulate their emotions by providing personalized emotion regulation techniques based on their emotional state and personality traits.

Features

- Emotion recognition from text
- Personalized emotion regulation techniques
- User-configurable personality traits (OCEAN model)
- Interactive chat interface
- Real-time emotion analysis visualization

Installation

1. Clone the repository:
   git clone https://github.com/yourusername/emotion-regulation-assistant.git
   cd emotion-regulation-assistant

2. Install the required packages:
   pip install -r requirements.txt

3. Create a .env file with your OpenAI API key:
   OPENAI_API_KEY=your_api_key_here

Usage

Run the Streamlit app:
streamlit run app.py

Then open your browser and go to http://localhost:8501

Project Structure

- app.py: Streamlit web interface
- src/: Source code
  - main.py: Main application
  - components/: Individual components
    - emotion_recognition.py: Emotion recognition model
    - rag_agent.py: Retrieval-Augmented Generation agent
    - planner_verifier.py: Planner-Verifier agent
    - empathetic_response.py: Empathetic response generator
  - utils/: Utility functions
    - openai_utils.py: OpenAI API utilities
    - memory.py: Chat memory management
    - vector_store.py: Vector store management
  - config.py: Configuration settings
- data/: Data files
  - knowledge_base/: Emotion regulation techniques
- tests/: Test files

How It Works

1. The user enters a message in the chat interface
2. The system analyzes the emotional content of the message
3. Based on the user's emotional state and personality traits, the system retrieves relevant emotion regulation techniques
4. The system generates an empathetic response that incorporates the techniques
5. The response is verified for quality and regenerated if necessary
6. The response is displayed to the user

License

MIT













venv\Scripts\activate
streamlit run app.py