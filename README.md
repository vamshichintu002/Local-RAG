# Local RAG System with Google's Gemini LLM

This project implements a Retrieval-Augmented Generation (RAG) system using Google's Gemini LLM and LlamaIndex, designed to work with local system files as a knowledge base.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your environment variables:
Create a `.env` file in the project root and add your Google API key:
```
GOOGLE_API_KEY=your_api_key_here
```

4. Add your documents:
Place your knowledge base documents in the `data/my_knowledge_base/` directory.

## Usage

1. Run the RAG system:
```bash
python rag_system.py
```

2. Enter your queries when prompted. The system will:
   - Search through your local knowledge base
   - Retrieve relevant information
   - Generate a response using Gemini LLM

3. Type 'quit' to exit the system.

## Project Structure

- `rag_system.py`: Main implementation file
- `requirements.txt`: Project dependencies
- `data/my_knowledge_base/`: Directory for knowledge base documents
- `.env`: Environment variables configuration

## Features

- Local file-based knowledge base
- Google Gemini LLM integration
- Vector store indexing for efficient retrieval
- Interactive query interface
