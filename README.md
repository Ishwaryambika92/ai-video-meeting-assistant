# 🎬 AI Video & Meeting Assistant (Video Agent)

> An enterprise-grade, agentic AI meeting assistant that converts long video and audio recordings into searchable transcripts, structured summaries, action items with owners and deadlines, decisions repository, cross-meeting memory, and grounded multilingual Q&A using RAG.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit)](https://streamlit.io/)
[![Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-orange?logo=google)](https://ai.google.dev/)
[![Whisper](https://img.shields.io/badge/Speech%20to%20Text-OpenAI%20Whisper-green?logo=openai)](https://github.com/openai/whisper)
[![Sarvam AI](https://img.shields.io/badge/Indic%20STT-Sarvam%20AI-purple)](https://www.sarvam.ai/)
[![VectorDB](https://img.shields.io/badge/Vector%20Store-ChromaDB-blueviolet)](https://www.trychroma.com/)
[![SQLite](https://img.shields.io/badge/Database-SQLite%20(WAL)-003B57?logo=sqlite)](https://www.sqlite.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 1. Project Overview

Manual review of lengthy meeting recordings and video lectures is inefficient, prone to oversight, and disconnects discussions from actionable execution.

**Video Agent** solves this by establishing an end-to-end intelligence pipeline:
1. **Media Ingestion:** Ingests YouTube URLs or local audio/video uploads, converts them to standardized mono 16kHz WAV format, and slices long recordings into manageable audio segments.
2. **Dual-Engine Multilingual Speech-to-Text:** Dispatches English audio to local **OpenAI Whisper** and Indian/Indic languages to **Sarvam AI** with parallel chunk processing.
3. **AI Editorial Cleaning:** Repairs phonetic speech recognition slips, strips verbal fillers (`um`, `uh`, `like`, stutters), corrects syntax, and produces a publication-ready transcript while strictly preserving 100% of facts, names, and timestamps.
4. **Structured Generative Extraction:** Leverages **Google Gemini** with Pydantic schemas to extract executive summaries, key takeaways, assigned action items (with owner, deadline, and priority), decisions, open questions, and follow-ups.
5. **Single-Meeting & Cross-Meeting RAG:** Indexes transcripts into **ChromaDB** using **HuggingFace** embeddings (`all-MiniLM-L6-v2`) and pairs them with an authoritative SQLite meeting memory store to enable conversational Q&A over both individual sessions and historical organizational memory.
6. **Executive Multi-Format Reporting:** Renders comprehensive documentation into downloadable TXT, print-ready PDF, and high-resolution JPEG infographic cards.

---

## ✨ 2. Key Features

### 🎙️ Multilingual Speech-to-Text (Local + Cloud)
- **English:** Transcribed locally via OpenAI Whisper (`base`, `small`, or `medium` models) with zero external API fees.
- **10+ Indian Languages:** Transcribed via Sarvam AI REST API (`saaras:v3`) with concurrent multi-worker chunk dispatching:
  * Telugu, Hindi, Tamil, Kannada, Malayalam, Marathi, Bengali, Gujarati, Punjabi, and Hinglish.
- **Auto Detect:** Analyzes audio to determine the spoken language and route to the optimal speech engine.

### ✍️ Intelligent Transcript Cleaning & Side-by-Side Verification
- Removes speech disfluencies and filler words (`um`, `uh`, `er`, `you know`, `basically`, `actually`, repeated phrases).
- Contextually repairs speech-to-text recognition errors (e.g., `cooper nettles` → `Kubernetes`, `sink between data base` → `sync between database`, `p ninety nine` → `P99`).
- Corrects grammar, punctuation, and capitalization without summarizing or omitting content.
- Preserves speaker tags (`Speaker 1 (00:00):`) and all factual details.
- Interactive **Side-by-Side Comparison View** in the UI allows direct visual comparison between raw audio transcripts and AI-cleaned versions.

### 📝 Structured Intelligence & Extraction (Pydantic + Gemini)
- **Executive Summary:** Concise title (≤12 words), 1-2 sentence overview, and comprehensive paragraph synthesis.
- **Action Items Tracker:** Automatically extracts tasks, assignees, deadlines, and priorities into an editable SQLite task tracker.
- **Decisions Repository:** Extracts and deduplicates concrete decisions made across meetings.
- **Open Questions & Follow-ups:** Flags unresolved inquiries and scheduled next steps.

### 💬 Grounded Conversational RAG with Script Enforcement
- Retrieves evidence from ChromaDB vector collections with timestamp citations.
- Pre-retrieval language command detection (e.g., *"Naku eppudu Telugu lo cheppu"*, *"Naku Hindi lo cheppu"*).
- Strict script enforcement: answers in Telugu script (తెలుగు), Devanagari Hindi (हिंदी), Tamil script (தமிழ்), Roman Telugu-English, or clean English.
- Resolves short follow-up messages (*"cheppu"*, *"explain"*, *"continue"*) based on conversational context.

### 🧠 Cross-Meeting Memory & Organization Intelligence
- Queries across all historical meetings stored in SQLite and ChromaDB.
- Answers macro questions such as *"What decisions did we make about database architecture across all meetings?"* or *"What tasks are assigned to Alex this month?"*.

### 📊 Progress Dashboard & Multi-Tenant Authentication
- Visual KPI cards: total meetings, completion rates, overdue tasks, and pending action items.
- Dynamic task completion charts powered by Altair.
- Dual-mode authentication:
  * **Local Vault Mode:** Secure `bcrypt` password hashing and SQLite token sessions with zero cloud dependencies.
  * **Cloud Mode:** Optional integration with Supabase Auth.
- Dark, Light, and System appearance modes with responsive CSS design tokens.

---

## 🛠️ 3. Technologies Used

| Layer | Technology | Purpose in Codebase |
| :--- | :--- | :--- |
| **Frontend UI** | Streamlit (`>=1.40.0`) | Reactive web interface, tab routing, and responsive controls |
| **Styling** | Custom Vanilla CSS | Theme variables (`--bg-main`, `--primary-accent`), dark/light modes |
| **Charts & Analytics** | Altair & Pandas | Interactive task progress charts and metric computations |
| **Language Model** | Google Gemini (`google-genai>=1.0.0`) | Structured summarization, action item extraction, cleanup, and RAG |
| **Speech-to-Text (Local)** | OpenAI Whisper (`>=20250625`) | Local English audio transcription via PyTorch |
| **Speech-to-Text (Cloud)** | Sarvam AI API (`saaras:v3`) | Indic language transcription with concurrent thread workers |
| **Translation** | Deep Translator (`deep-translator`) | Multi-language translation with Gemini fallback |
| **Vector Store** | ChromaDB (`chromadb>=0.5.20`, `langchain-chroma`) | Persistent local vector database for semantic chunk retrieval |
| **Embeddings** | HuggingFace (`sentence-transformers>=3.4.0`) | `all-MiniLM-L6-v2` dense vector embedding generation |
| **Relational Database** | SQLite 3 | WAL-mode persistent storage for meetings, tasks, and auth sessions |
| **Security & Hashing** | Bcrypt (`bcrypt>=5.0.0`) | Salted cryptographic password hashing and token validation |
| **Cloud Auth (Optional)** | Supabase (`supabase>=2.33.0`) | Optional cloud authentication provider |
| **Audio Processing** | FFmpeg, `yt-dlp`, `pydub` | Media downloading, 16kHz WAV normalization, and audio chunking |
| **Document Export** | `fpdf2`, `Pillow` | Downloadable reports in TXT, PDF, and high-resolution JPEG formats |
| **Validation** | Pydantic v2 (`>=2.10.0`) | Typed data validation for all Gemini structured outputs |

---

## 📂 4. Project Structure

```
video-agent/
├── .streamlit/
│   └── config.toml              # Streamlit server and theme configuration
├── core/                        # Modular core application subsystems
│   ├── auth.py                  # Dual-mode authentication engine (Local bcrypt + Supabase)
│   ├── auth_ui.py               # Authentication views (Login, Signup, Reset Password, Profile)
│   ├── cross_meeting_rag.py     # Multi-meeting Q&A synthesizing SQLite records + ChromaDB
│   ├── dashboard_ui.py          # Dashboard views (Overview metrics, History, Tasks, Decisions)
│   ├── extractor.py             # Gemini structured extraction (Action items, Decisions, Questions)
│   ├── meeting_store.py         # SQLite CRUD manager for meetings, tasks, and progress metrics
│   ├── rag_engine.py            # Grounded single-meeting RAG chat with script enforcement
│   ├── report_generator.py      # Multi-format report builder (TXT, PDF via fpdf2, JPG via Pillow)
│   ├── summariser.py            # Gemini structured summarization (Title, Overview, Key Points)
│   ├── transcriber.py           # Speech-to-text dispatcher (OpenAI Whisper + Sarvam AI)
│   ├── translator.py            # Transcript cleaning, phonetic repair, and Indic translation
│   └── vector_store.py          # ChromaDB index manager and HuggingFace embeddings
├── utils/
│   └── audio_processor.py       # Media acquisition via yt-dlp, FFmpeg conversion, and audio chunking
├── tests/                       # Subsystem verification test suite
│   ├── run_all_tests.py         # Master test runner executing all 14 subsystem tests
│   ├── test_auth.py             # Unit tests for password hashing, session tokens, and auth
│   ├── test_chatbot_language.py # Verifies multi-language commands and script enforcement
│   ├── test_cross_meeting_rag.py# Verifies cross-meeting synthesis and memory queries
│   ├── test_decisions_repo.py   # Verifies decision normalization and deduplication
│   ├── test_extractor.py        # Verifies structured action item extraction via Gemini
│   ├── test_meeting_store.py    # Verifies SQLite CRUD operations and overdue calculations
│   ├── test_report_generator.py # Verifies TXT, PDF, and JPG report generation
│   ├── test_summariser.py       # Verifies Pydantic structured summarization
│   ├── test_transcriber.py      # Verifies Whisper and Sarvam transcription
│   ├── test_translator.py       # Verifies speech cleaning, filler removal, and translation
│   └── test_vector_store.py     # Verifies ChromaDB indexing and similarity search
├── config.py                    # Centralized settings singleton, paths, and hardware detection
├── main.py                      # Application entry point, router, CSS injector, and layout
├── requirements.txt             # Pip dependency specifications
├── pyproject.toml               # Project metadata and type checker configuration
├── packages.txt                 # Linux/Debian system dependencies (specifies 'ffmpeg')
├── .env.example                 # Sanitized environment configuration template
└── README.md                    # Project documentation
```

---

## ⚙️ 5. Installation Instructions

### Prerequisites
- **Python:** Version 3.10 or 3.11 recommended.
- **FFmpeg:** Required for audio extraction and conversion.
  * **Windows:** `winget install Gyan.FFmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html).
  * **macOS:** `brew install ffmpeg`
  * **Linux (Ubuntu/Debian):** `sudo apt update && sudo apt install ffmpeg`

### Step 1: Clone the Repository
```bash
git clone https://github.com/Ishwaryambika92/ai-video-meeting-assistant.git
cd ai-video-meeting-assistant
```

### Step 2: Create and Activate a Virtual Environment
Using standard Python venv:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

*(Alternatively, using `uv`: `uv venv && .venv\Scripts\activate`)*

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔑 6. Environment Variable Setup

Copy the template configuration file to create your local `.env`:
```bash
# Windows (PowerShell)
Copy-Item .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and fill in your API credentials:

```ini
# ==============================================================================
# AI Video & Meeting Assistant - Environment Configuration
# ==============================================================================

# ---- LLM & AI Services ----
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash-lite

# Optional: Sarvam AI (Required only for Indian languages / Hinglish)
SARVAM_API_KEY=your_sarvam_api_key_here
SARVAM_STT_MODEL=saaras:v3
SARVAM_MAX_WORKERS=4

# ---- Transcription & Audio ----
WHISPER_MODEL=base
AUDIO_CHUNK_MINUTES=10
TRANSCRIPT_CHUNK_SIZE=1200
TRANSCRIPT_CHUNK_OVERLAP=150

# ---- RAG & Vector Database ----
HF_TOKEN=your_huggingface_token_here
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RAG_TOP_K=4

# ---- Authentication & Security ----
REQUIRE_AUTH=true
AUTH_SECRET_KEY=generate_a_random_32_char_secret_key_here

# Optional: Supabase Cloud Credentials (leave empty for local SQLite bcrypt auth)
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_ROLE_KEY=

# ---- Application Logging ----
LOG_LEVEL=INFO
```

> [!NOTE]
> - **`GEMINI_API_KEY`** is required for summarization, action item extraction, transcript cleaning, and RAG chat. Obtain a key from [Google AI Studio](https://aistudio.google.com/).
> - **`SARVAM_API_KEY`** is optional; required only when transcribing Indic languages (Telugu, Hindi, Tamil, etc.). English transcription runs 100% locally with Whisper.
> - If **`SUPABASE_URL`** is left blank, the application automatically uses its local cryptographic SQLite vault with `bcrypt` encryption.

---

## 🚀 7. Running the Application

Launch the Streamlit web application:

```bash
python -m streamlit run main.py
```

The application will start and open in your default browser at:
```text
http://localhost:8501
```

### Application Workflow:
1. **Sign Up / Login:** Register an account using the local authentication screen (or login with existing credentials).
2. **Ingest Media:** Paste any YouTube video URL or upload a local audio/video file (`.mp4`, `.wav`, `.mp3`, `.m4a`).
3. **Select Language:** Choose **English** (Whisper), a specific Indic language (Sarvam AI), or **Auto Detect**.
4. **Process Meeting:** Click **Process Meeting** to run the 6-step intelligence pipeline.
5. **Explore Views:**
   - **Summary Tab:** Review executive summary, important points, and takeaways.
   - **Action Items Tab:** View and manage tasks with assignees and deadlines.
   - **Transcript Tab:** Inspect the **Side-by-Side Comparison** between raw audio speech and AI-cleaned transcript.
   - **Export Report Tab:** Download the documentation as a text file, print-ready PDF, or JPEG infographic.
   - **Overview Dashboard:** Monitor organizational KPIs and task progress charts.
   - **Floating AI Assistant:** Chat with your transcript or query across past meeting memory.

---

## 🧪 8. Testing Instructions

The repository includes a comprehensive subsystem test suite to verify every component independently:

### Run the Master Test Runner:
```bash
python run_all_tests.py
```

### Run Individual Subsystem Tests:
```bash
# Test Gemini API connectivity
python test_gemini.py

# Test transcript cleaning, filler removal, and phonetic error repair
python test_translator.py

# Test structured summarization with Pydantic
python test_summariser.py

# Test action item and decision extraction
python test_extractor.py

# Test ChromaDB embeddings and vector storage
python test_vector_store.py

# Test RAG chat and multi-language script enforcement
python test_chatbot_language.py

# Test SQLite meeting store and task tracking
python test_meeting_store.py

# Test cross-meeting memory intelligence
python test_cross_meeting_rag.py

# Test report generation (TXT, PDF, JPG)
python test_report_generator.py

# Test authentication and password hashing
python test_auth.py
```

---

## 🔒 9. Security & Privacy Notes

- **Zero Hardcoded Secrets:** No API keys, credentials, or private secrets are stored in the codebase. All configurations are loaded via environment variables in [config.py](file:///d:/ISHU/video-agent/config.py).
- **Git Exclusions:** The [.gitignore](file:///d:/ISHU/video-agent/.gitignore) configuration strictly excludes `.env`, `data/` (local SQLite databases and ChromaDB vectors), `downloads/` (raw audio files), report outputs, and backup files from entering version control.
- **Local Data Isolation:** Meeting transcripts, vector embeddings, and audio downloads remain on the user's local machine and are never transmitted to third parties except for the designated AI model endpoints requested by the user.
- **Enterprise-Grade Password Storage:** Passwords in the local authentication backend are hashed using `bcrypt` with cryptographic salts. Plaintext passwords are never stored or logged.

---

## 👩‍💻 Author

**Ishwaryambika Patta**  
*AI/ML & Software Development Enthusiast*  
GitHub: [@Ishwaryambika92](https://github.com/Ishwaryambika92)