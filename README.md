# RAG with LangChain

A Retrieval-Augmented Generation (RAG) pipeline built with LangChain, FAISS, Sentence Transformers, and Groq LLM API.

Loads documents (PDF, TXT, CSV, DOCX, JSON) from `data/`, indexes them in a local FAISS vector store, and performs vector similarity search to answer queries using Groq.

## Tech Stack
- Python
- LangChain
- FAISS (`faiss-cpu`)
- HuggingFace Embeddings (`sentence-transformers`)
- Groq API (`langchain-groq`)

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root folder and add your Groq API key:
   ```env
   GROQ_API_KEY=your_groq_api_key
   ```

## Running the App

Add target documents inside the `data/` folder, then run:

```bash
python app.py
```
