# 🎥 AI-Powered YouTube Video Summarizer & Notes Generator 

An AI-powered YouTube video summarization system using LLMs, generating structured notes and enabling question-answering via retrieval-based methods

## Features
- Paste any YouTube URL and get a summary instantly
- Ask follow-up questions about the video
- Powered by Groq (LLaMA 3.3) and RAG pipeline

## Tech Stack
- Streamlit
- Groq API (LLaMA 3.3 70B)
- ChromaDB
- LangChain
- Sentence Transformers

## Live Demo
[Click here](https://yourusername-video-summarizer.streamlit.app)

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```