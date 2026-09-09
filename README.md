# AI Video/Meeting Assistant

Turn a YouTube URL, meeting recording, or local audio/video file into a
transcript, structured summary, action items, and a searchable chat
interface — with TXT/PDF export.

```
YouTube URL / Audio / Video
    -> Audio extraction (yt-dlp / FFmpeg)
    -> Chunking
    -> Transcription (Whisper for English, Sarvam AI for Hindi/Hinglish)
    -> Translation to clean English (if needed)
    -> Structured summarization (Mistral)
    -> Action items / decisions / questions (Mistral)
    -> Embeddings (HuggingFace) -> ChromaDB
    -> RAG chat
    -> Streamlit UI + PDF/TXT report
```

## Setup

1. **Install FFmpeg** and make sure it's on your PATH.
   - Windows: `winget install ffmpeg` (then restart your terminal)
   - macOS: `brew install ffmpeg`
   - Linux: `apt install ffmpeg`

2. **Install Python dependencies** (Python 3.11 or 3.12 recommended):
   ```bash
   pip install -r requirements.txt
   ```
   If you have an NVIDIA GPU and want CUDA-accelerated Whisper, install the
   matching CUDA build of PyTorch first per https://pytorch.org/get-started/locally/ —
   the app auto-detects and uses the GPU if `torch.cuda.is_available()`.

3. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```
   Then fill in `MISTRAL_API_KEY` (required for summarization/extraction/chat)
   and `SARVAM_API_KEY` (only required if you'll transcribe Hindi/Hinglish).

4. **Run the app**:
   ```bash
   streamlit run main.py
   ```

## ⚠️ About the keys in the original upload

The `.env` file in the uploaded project contained real, live Mistral and
Sarvam API keys in plaintext. Treat both as compromised — **rotate them
in the Mistral and Sarvam dashboards** and put fresh keys only in your own
local `.env` (which is already covered by the `.gitignore` below).

## Project layout

```
video-agent/
  config.py              # env loading, logging, device detection
  main.py                # Streamlit UI
  test.py                # CLI smoke test for the audio+transcription pipeline
  utils/
    audio_processor.py   # YouTube download, WAV conversion, chunking
  core/
    transcriber.py       # Whisper (English) + Sarvam AI (Hindi/Hinglish)
    translator.py         # Hindi/Hinglish -> clean English + transcript cleanup
    summariser.py         # structured summary (title/overview/points/takeaways)
    extractor.py           # action items / decisions / questions / follow-ups
    vector_store.py        # chunking + embeddings + ChromaDB
    rag_engine.py           # retrieval-augmented chat
    report_generator.py     # TXT/PDF report export
  .env.example
  requirements.txt
```

## Notes on changes from the uploaded project

See the chat summary for the full list. Highlights:
- `utils/audio_processor.py` had module-level code that downloaded a real
  YouTube video on `import` — removed; every step is now an explicit
  function call.
- Sarvam AI transcription now calls the current `/speech-to-text` endpoint
  (`model=saaras:v3`, `mode=transcribe|translate`) instead of the legacy
  `/speech-to-text-translate` endpoint used by the original tutorial.
- `test1.py` was a byte-for-byte duplicate of `core/transcriber.py` and was
  removed.
- Whisper device selection was already correct (CUDA if available, else
  CPU) and has been preserved, now centralized in `config.get_device()`.
