from youtube_transcript_api import YouTubeTranscriptApi
import re
import os
from dotenv import load_dotenv

load_dotenv()

def get_transcript(youtube_url):
    video_id = youtube_url.split("v=")[1].split("&")[0]
    
    ytt_api = YouTubeTranscriptApi()
    
    try:
        fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
    except:
        try:
            fetched_transcript = ytt_api.fetch(video_id, languages=['hi'])
        except:
            fetched_transcript = ytt_api.fetch(video_id)
    
    full_transcript = " ".join([snippet.text for snippet in fetched_transcript])
    return full_transcript

###print(get_transcript("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))    

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_transcript(transcript):
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,      # each chunk = ~1000 characters
        chunk_overlap = 200     # chunks overlap a bit so context isn't lost
                                # at the edges (like pages sharing a sentence)
    )
    
    chunks = splitter.split_text(transcript)
    # chunks is now a list like:
    # ["first 1000 chars...", "next 1000 chars...", ...]
    
    return chunks

import numpy as np
from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

def store_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    return chunks, embeddings


from groq import Groq

def generate_summary(transcript):
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""You are given the transcript of a YouTube video.
                The transcript may be in Hindi, English, or both. Always respond in English.
                Give me:
                - A 5 line overall summary
                - 5 to 7 key points from the video
                - Any important terms or concepts mentioned

                Transcript:
                {transcript[:20000]}"""
            }
        ]
    )
    return response.choices[0].message.content

def answer_question(question, chunks, embeddings):
    # embed the question
    question_embedding = embedding_model.encode([question])[0]
    
    # find most similar chunks using cosine similarity
    similarities = np.dot(embeddings, question_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
    
    # get top 3 most relevant chunks
    top_indices = np.argsort(similarities)[-3:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices]
    context = "\n\n".join(relevant_chunks)
    
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "user",
                "content": f"""Answer the question in English using ONLY the context below.
                The context may be in Hindi or English.
                If the answer is not in the context, say "I couldn't find that in the video."

                Context:
                {context}

                Question: {question}"""
            }
        ]
    )
    return response.choices[0].message.content

import streamlit as st

st.set_page_config(page_title="Video Summarizer", page_icon="🎥")
st.title("🎥 YouTube Video Summarizer")
st.write("Paste any YouTube link — get a summary and ask questions about it")

# --- Input ---
url = st.text_input("YouTube URL")

if st.button("Analyze Video"):
    with st.spinner("Fetching transcript..."):
        transcript = get_transcript(url)
    
    with st.spinner("Generating summary..."):
        summary = generate_summary(transcript)
    
    with st.spinner("Setting up Q&A..."):
        chunks = chunk_transcript(transcript)
        chunks, embeddings = store_embeddings(chunks)
    
    st.session_state.chunks = chunks
    st.session_state.embeddings = embeddings
    st.session_state.analyzed = True
    
    st.subheader("📝 Summary")
    st.write(summary)

# --- Q&A Section ---
# Only shows after video is analyzed

if st.session_state.get("analyzed"):
    st.subheader("💬 Ask Anything About The Video")
    question = st.text_input("Your question")
    
    if st.button("Ask"):
        with st.spinner("Thinking..."):
            answer = answer_question(
                question,
                st.session_state.chunks,
                st.session_state.embeddings
            )
        st.write(answer)