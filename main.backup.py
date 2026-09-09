"""
AI Video/Meeting Assistant - Streamlit UI.

YouTube URL / local file -> audio extraction -> transcription
    -> (translation) -> summarization -> extraction -> vector store
    -> RAG chat, with TXT/PDF report export.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import streamlit as st

from config import get_logger, validate_settings
from core.extractor import ExtractionError, extract_action_items
from core.rag_engine import RagError, ask
from core.report_generator import ReportData, build_pdf_report, build_txt_report
from core.summariser import SummarizationError, summarize_transcript
from core.transcriber import TranscriptionError, transcribe_all, transcribe_all_original
from core.translator import TranslationError, clean_transcript, translate_to_clean_english
from core.vector_store import VectorStoreError, index_transcript
from utils.audio_processor import AudioProcessingError, cleanup_chunks, process_input

logger = get_logger(__name__)

st.set_page_config(page_title="AI Video/Meeting Assistant", page_icon="🎬", layout="wide")


# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        "session_id": None,
        "processed": False,
        "clean_transcript": "",
        "original_transcript": "",
        "summary": None,
        "extraction": None,
        "chat_history": [],
        "processing_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def run_pipeline(source: str, language: str, chunk_minutes: int) -> None:
    session_id = uuid.uuid4().hex[:12]
    chunk_paths: list[str] = []

    status = st.status("Processing...", expanded=True)
    try:
        status.write("Step 1/6 - Extracting & chunking audio...")
        chunk_paths = process_input(source, chunk_minutes=chunk_minutes)

        status.write(f"Step 2/6 - Transcribing {len(chunk_paths)} chunk(s)...")
        transcript_result = transcribe_all(chunk_paths, language=language, chunk_seconds=chunk_minutes * 60)

        original_text = transcript_result.full_text
        if language.lower() == "hinglish":
            status.write("Step 3/6 - Preserving original-language transcript...")
            original_text = transcribe_all_original(chunk_paths, language=language)
            status.write("Step 3/6 - Translating to clean English...")
            clean_text = transcript_result.full_text  # already English via Sarvam mode=translate
        else:
            status.write("Step 3/6 - Cleaning transcript...")
            clean_text = clean_transcript(transcript_result.full_text)

        if not clean_text.strip():
            raise RuntimeError("Transcription produced no text. The audio may be silent or unsupported.")

        status.write("Step 4/6 - Generating summary...")
        summary = summarize_transcript(clean_text)

        status.write("Step 5/6 - Extracting action items, decisions & questions...")
        extraction = extract_action_items(clean_text)

        status.write("Step 6/6 - Building searchable index for chat...")
        index_transcript(session_id, transcript_result.segments)

        st.session_state.session_id = session_id
        st.session_state.clean_transcript = clean_text
        st.session_state.original_transcript = original_text
        st.session_state.summary = summary
        st.session_state.extraction = extraction
        st.session_state.processed = True
        st.session_state.processing_error = None
        st.session_state.chat_history = []

        status.update(label="Processing complete!", state="complete", expanded=False)

    except AudioProcessingError as exc:
        status.update(label="Audio processing failed", state="error")
        st.session_state.processing_error = f"Audio error: {exc}"
    except TranscriptionError as exc:
        status.update(label="Transcription failed", state="error")
        st.session_state.processing_error = f"Transcription error: {exc}"
    except TranslationError as exc:
        status.update(label="Translation failed", state="error")
        st.session_state.processing_error = f"Translation error: {exc}"
    except SummarizationError as exc:
        status.update(label="Summarization failed", state="error")
        st.session_state.processing_error = f"Summarization error: {exc}"
    except ExtractionError as exc:
        status.update(label="Extraction failed", state="error")
        st.session_state.processing_error = f"Extraction error: {exc}"
    except VectorStoreError as exc:
        status.update(label="Indexing failed", state="error")
        st.session_state.processing_error = f"Vector store error: {exc}"
    except Exception as exc:  # noqa: BLE001
        status.update(label="Unexpected error", state="error")
        logger.exception("Unexpected pipeline failure")
        st.session_state.processing_error = f"Unexpected error: {exc}"
    finally:
        cleanup_chunks(chunk_paths)


# --------------------------------------------------------------------------
# Sidebar - input
# --------------------------------------------------------------------------
st.title("🎬 AI Video/Meeting Assistant")
st.caption("Transcribe, summarize, and chat with any video or meeting recording.")

config_problems = validate_settings(require_mistral=True, require_sarvam=False)
if config_problems:
    st.warning("\n".join(config_problems))

with st.sidebar:
    st.header("Input")
    input_mode = st.radio("Source", ["YouTube URL", "Upload file"], horizontal=True)

    source_value: str | None = None
    uploaded_temp_path: str | None = None

    if input_mode == "YouTube URL":
        source_value = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    else:
        uploaded = st.file_uploader("Audio or video file", type=["mp4", "mp3", "wav", "m4a", "mov", "mkv", "webm"])
        if uploaded is not None:
            suffix = Path(uploaded.name).suffix or ".bin"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                uploaded_temp_path = tmp.name
            source_value = uploaded_temp_path

    language = st.selectbox("Language", [ "English",
        "Telugu",
        "Hindi",
        "Hinglish",
        "Tamil",
        "Kannada",
        "Malayalam",
        
        "Marathi",
        "Bengali",
        "Gujarati",
        "Punjabi",], help="Hinglish uses Sarvam AI; English uses local Whisper.")
    chunk_minutes = st.slider("Audio chunk length (minutes)", 2, 20, 10)

    process_clicked = st.button("🚀 Process", type="primary", use_container_width=True, disabled=not source_value)

    if st.session_state.processed:
        st.divider()
        if st.button("🔄 Start new session", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.processed = False
            st.session_state.clean_transcript = ""
            st.session_state.original_transcript = ""
            st.session_state.summary = None
            st.session_state.extraction = None
            st.session_state.chat_history = []
            st.session_state.processing_error = None
            st.rerun()

if process_clicked and source_value:
    run_pipeline(source_value, language.lower(), chunk_minutes)

if st.session_state.processing_error:
    st.error(st.session_state.processing_error)


# --------------------------------------------------------------------------
# Main content
# --------------------------------------------------------------------------
if not st.session_state.processed:
    st.info("Provide a YouTube URL or upload a file, then click **Process** to get started.")
else:
    summary = st.session_state.summary
    extraction = st.session_state.extraction

    st.subheader(summary.title)
    st.write(summary.overview)

    tabs = st.tabs(["📝 Summary", "✅ Action Items", "📄 Transcript", "💬 Chat", "⬇️ Export"])

    with tabs[0]:
        st.markdown("### Summary")
        st.write(summary.summary)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Important Points")
            for p in summary.important_points or []:
                st.markdown(f"- {p}")
        with col2:
            st.markdown("### Key Takeaways")
            for t in summary.key_takeaways or []:
                st.markdown(f"- {t}")

    with tabs[1]:
        st.markdown("### Action Items")
        if extraction.action_items:
            st.table(
                [
                    {"Action": i.action, "Owner": i.owner, "Deadline": i.deadline, "Status": i.status}
                    for i in extraction.action_items
                ]
            )
        else:
            st.caption("No action items found.")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### Decisions")
            for d in extraction.decisions or ["None found."]:
                st.markdown(f"- {d}")
        with col2:
            st.markdown("### Open Questions")
            for q in extraction.open_questions or ["None found."]:
                st.markdown(f"- {q}")
        with col3:
            st.markdown("### Follow-ups")
            for f in extraction.follow_ups or ["None found."]:
                st.markdown(f"- {f}")

    with tabs[2]:
        if st.session_state.original_transcript.strip() != st.session_state.clean_transcript.strip():
            st.markdown("### Original Transcript")
            st.text_area("Original", st.session_state.original_transcript, height=200, label_visibility="collapsed")
        st.markdown("### Clean Transcript")
        st.text_area("Clean", st.session_state.clean_transcript, height=300, label_visibility="collapsed")

    with tabs[3]:
        st.markdown("### Chat with this video/meeting")
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)

        question = st.chat_input("Ask something about this video/meeting...")
        if question:
            st.session_state.chat_history.append(("user", question))
            with st.chat_message("user"):
                st.write(question)
            try:
                with st.spinner("Thinking..."):
                    result = ask(st.session_state.session_id, question)
                answer_text = result.answer
                if result.timestamps:
                    answer_text += f"\n\n*Referenced around: {', '.join(result.timestamps)}*"
            except RagError as exc:
                answer_text = f"⚠️ {exc}"
            st.session_state.chat_history.append(("assistant", answer_text))
            with st.chat_message("assistant"):
                st.write(answer_text)

    with tabs[4]:
        st.markdown("### Export report")
        report_data = ReportData(
            title=summary.title,
            summary=summary,
            extraction=extraction,
            transcript=st.session_state.clean_transcript,
            original_transcript=st.session_state.original_transcript,
        )
        txt_report = build_txt_report(report_data)
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "⬇️ Download TXT", data=txt_report, file_name="report.txt",
                mime="text/plain", use_container_width=True,
            )
        with col2:
            try:
                pdf_bytes = build_pdf_report(report_data)
                st.download_button(
                    "⬇️ Download PDF", data=pdf_bytes, file_name="report.pdf",
                    mime="application/pdf", use_container_width=True,
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not build PDF: {exc}")
