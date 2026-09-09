# 🎥 AI Video & Meeting Assistant

> An AI-powered system that transforms long video and meeting recordings into searchable transcripts, summaries, action items, decisions, follow-ups, and intelligent Q&A using RAG.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Gemini](https://img.shields.io/badge/LLM-Gemini-orange)
![Whisper](https://img.shields.io/badge/Speech-Whisper-green)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red?logo=streamlit)
![RAG](https://img.shields.io/badge/AI-RAG-yellow)

---

## 📌 Overview

The **AI Video & Meeting Assistant** converts long video or meeting recordings into useful, structured information.

Instead of watching an entire recording manually, users can:

- 🎙️ Generate transcripts
- 🌍 Process multiple languages
- 📝 Generate meeting summaries
- ✅ Extract action items
- 📌 Identify decisions
- ❓ Find open questions
- 🔄 Identify follow-ups
- 💬 Ask questions about the meeting using RAG
- 📄 Generate TXT/PDF reports

The project combines **speech processing, Generative AI, embeddings, vector search, and RAG** into a single application.

---

## 🎯 Problem Statement

Long meetings and video recordings contain important information such as:

- Decisions
- Tasks
- Deadlines
- Discussions
- Questions
- Responsibilities

Manually reviewing long recordings is time-consuming and important information can easily be missed.

### 💡 Solution

This system automatically processes the recording and converts it into searchable and structured knowledge.

---

## ✨ Key Features

### 🎙️ Multilingual Transcription

Supports:

- English → Whisper
- Indian languages → Sarvam AI
- Auto Detect → Detects the spoken language

The system preserves the selected/original language during transcription.

### 📝 AI Summarization

Gemini generates a structured summary containing the most important information from the recording.

### ✅ Action Item Extraction

Automatically identifies:

- Task
- Owner
- Deadline
- Status

### 📌 Decision Extraction

Extracts important decisions made during the meeting.

### ❓ Open Questions

Identifies questions that remain unanswered.

### 🔄 Follow-ups

Extracts follow-up tasks and future discussion points.

### 💬 RAG-based Meeting Chat

Users can ask questions about the transcript instead of manually searching through the entire recording.

Example:

```text
Who is responsible for the backend integration?

What decisions were made?

What needs to be completed by Friday?

The system retrieves relevant transcript chunks and generates an answer using Gemini.

📄 Report Generation

Meeting information can be exported into:

TXT
PDF

🏗️ System Architecture

                ┌─────────────────────┐
                │   Video / Audio     │
                │    / YouTube URL    │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Audio Extraction    │
                │ yt-dlp + FFmpeg     │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │   Transcription    │
                │ Whisper / Sarvam AI │
                └──────────┬──────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ Transcript Cleaning │
                │ & Language Handling │
                └──────────┬──────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
       ┌────────────────┐    ┌────────────────┐
       │ Gemini AI      │    │ HuggingFace    │
       │ Summarization  │    │ Embeddings     │
       │ & Extraction   │    └───────┬────────┘
       └───────┬────────┘            │
               │                     ▼
               │             ┌───────────────┐
               │             │   ChromaDB    │
               │             │ Vector Store  │
               │             └───────┬───────┘
               │                     │
               │                     ▼
               │             ┌───────────────┐
               └────────────►│ RAG Question  │
                             │    Answering   │
                             └───────┬───────┘
                                     │
                                     ▼
                            ┌─────────────────┐
                            │    Streamlit    │
                            │       UI        │
                            └─────────────────┘

🛠️ Tech Stack

| Technology      | Purpose                                 |
| --------------- | --------------------------------------- |
| **Python**      | Core development                        |
| **Streamlit**   | Web interface                           |
| **Gemini**      | Summarization, extraction & RAG         |
| **Whisper**     | English speech transcription            |
| **Sarvam AI**   | Indian language transcription           |
| **yt-dlp**      | YouTube/media processing                |
| **FFmpeg**      | Audio extraction & processing           |
| **HuggingFace** | Text embeddings                         |
| **ChromaDB**    | Vector database                         |
| **Pydantic**    | Structured AI outputs                   |
| **uv**          | Python environment & package management |

🌍 Multilingual Support

The application supports multilingual speech transcription.

Supported languages include:

English • Telugu • Hindi • Tamil • Kannada • Malayalam • Marathi • Bengali • Gujarati • Punjabi • Hinglish

Users can either:

Select a specific supported language
Use Auto Detect to identify the spoken language

The system is designed to preserve the spoken language during transcription.

💬 RAG-Based Question Answering

The application allows users to ask questions about a processed meeting instead of manually searching through the transcript.

Example Questions
Who is responsible for the backend integration?

What decisions were made?

What are the important action items?

What is the deadline?

What topics were discussed?
RAG Pipeline
Transcript
    ↓
Split into chunks
    ↓
Generate embeddings
    ↓
Store in ChromaDB
    ↓
User Question
    ↓
Retrieve relevant information
    ↓
Gemini generates the answer

RAG helps the system retrieve relevant information from the transcript before generating an answer.


⚙️ Installation & Run
1. Clone the Repository
git clone https://github.com/Ishwaryambika92/ai-video-meeting-assistant.git
cd ai-video-meeting-assistant
2. Create Virtual Environment
uv venv
3. Activate Environment

Windows:

.venv\Scripts\activate
4. Install Dependencies
uv pip install -r requirements.txt
5. Configure Environment Variables

Create a .env file in the project root:

GEMINI_API_KEY=your_gemini_api_key
SARVAM_API_KEY=your_sarvam_api_key

⚠️ Never share or commit your API keys to GitHub.

6. Run the Application
uv run streamlit run main.py

Open the local Streamlit URL displayed in the terminal.

🔒 Security

Sensitive credentials and generated/local files are excluded from the GitHub repository using .gitignore.

API keys are stored using environment variables instead of being hardcoded in the source code.

Never commit:
.env
API keys
Private credentials

🚀 Future Enhancements:

🧠 Persistent Meeting Memory
🔍 Meeting-to-Meeting Comparison
📊 Automatic Action-Item Tracking
⚠️ Contradiction Detection
⏱️ Timestamp-Based Evidence
👥 Speaker Identification
🌍 Improved Multilingual RAG
☁️ Cloud Deployment


👩‍💻 Author
Ishwaryambika Patta

AI/ML & Software Development Enthusiast

⭐ Project

AI Video & Meeting Assistant

Turn long recordings into searchable, structured, and actionable knowledge.