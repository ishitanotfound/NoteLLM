from youtube_transcript_api import YouTubeTranscriptApi
import re
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

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

import requests
from bs4 import BeautifulSoup

def get_video_metadata(video_id):
    # Thumbnail directly from YouTube — no scraping needed
    thumbnail = f"http://img.youtube.com/vi/{video_id}/0.jpg"
    
    # Scrape only title and channel
    url = f"https://www.youtube.com/watch?v={video_id}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    try:
        title = soup.find('meta', property='og:title')['content']
    except:
        title = "Unknown Title"
    
    try:
        channel_tag = soup.find('link', itemprop='name')
        channel = channel_tag['content'] if channel_tag else "Unknown Channel"
    except:
        channel = "Unknown Channel"
    
    return {
        "title": title,
        "thumbnail": thumbnail,
        "channel": channel
    }

def chunk_transcript(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(transcript)

def store_embeddings(chunks):
    embeddings = embedding_model.encode(chunks)
    return chunks, embeddings

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
    question_embedding = embedding_model.encode([question])[0]
    similarities = np.dot(embeddings, question_embedding) / (
        np.linalg.norm(embeddings, axis=1) * np.linalg.norm(question_embedding)
    )
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

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import io

def generate_pdf(summary, title="Video Summary"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph(title, styles['Title']))
    story.append(Spacer(1, 0.3*inch))
    
    # Summary content
    for line in summary.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['Normal']))
            story.append(Spacer(1, 0.1*inch))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# --- Streamlit UI ---
st.set_page_config(page_title="Video Summarizer", page_icon="🎥")
st.title("🎥 YouTube Video Summarizer")
st.write("Paste any YouTube link — get a summary and ask questions about it")

url = st.text_input("YouTube URL")

if st.button("Analyze Video"):
    if not url.strip():
        st.warning("⚠️ Please enter a YouTube URL first!")
        st.stop()
    
    with st.spinner("Fetching video info..."):
        video_id = url.split("v=")[1].split("&")[0]
        metadata = get_video_metadata(video_id)

    # Show video info at top
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(metadata['thumbnail'])
    with col2:
        st.subheader(metadata['title'])
        st.write(f"Channel: {metadata['channel']}")

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
    st.session_state.summary = summary
    st.session_state.video_title = metadata['title']

    st.subheader("📝 Summary")
    st.write(summary)

# --- Q&A Section ---
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

# --- Download Summary as PDF ---
if st.session_state.get("analyzed"):
    pdf_buffer = generate_pdf(
        st.session_state.get("summary", ""),
        title=st.session_state.get("video_title", "Video Summary")
    )
    st.download_button(
        label="📥 Download Summary as PDF",
        data=pdf_buffer,
        file_name="summary.pdf",
        mime="application/pdf"
    )