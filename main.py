"""
AI Video/Meeting Assistant - Streamlit UI.

YouTube URL / local file
    -> audio extraction
    -> transcription
    -> translation / cleanup
    -> summarization
    -> extraction
    -> vector store
    -> RAG chat
    -> TXT / PDF / JPG report export.
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

import streamlit as st

from config import get_logger, validate_settings

from core.extractor import (
    ExtractionError,
    extract_action_items,
)

from core.rag_engine import (
    RagError,
    ask,
)

from core.report_generator import (
    ReportData,
    build_pdf_report,
    build_txt_report,
    build_jpg_report,
)

from core.summariser import (
    SummarizationError,
    summarize_transcript,
)

from core.transcriber import (
    TranscriptionError,
    transcribe_all,
    transcribe_all_original,
)

from core.translator import (
    TranslationError,
    clean_transcript,
    translate_to_clean_english,
)

from core.vector_store import (
    VectorStoreError,
    index_transcript,
)

from utils.audio_processor import (
    AudioProcessingError,
    cleanup_chunks,
    process_input,
)


logger = get_logger(__name__)


# --------------------------------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="AI Video/Meeting Assistant",
    page_icon="🎬",
    layout="wide",
)


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

def run_pipeline(
    source: str,
    language: str,
    chunk_minutes: int,
) -> None:

    session_id = uuid.uuid4().hex[:12]

    chunk_paths: list[str] = []

    status = st.status(
        "Processing...",
        expanded=True,
    )

    try:

        # --------------------------------------------------------------
        # Step 1 - Extract and chunk audio
        # --------------------------------------------------------------

        status.write(
            "Step 1/6 - Extracting & chunking audio..."
        )

        chunk_paths = process_input(
            source,
            chunk_minutes=chunk_minutes,
        )

        if not chunk_paths:
            raise RuntimeError(
                "No audio chunks were created from the input."
            )

        # --------------------------------------------------------------
        # Step 2 - Transcription
        # --------------------------------------------------------------

        status.write(
            f"Step 2/6 - Transcribing "
            f"{len(chunk_paths)} chunk(s)..."
        )

        transcript_result = transcribe_all(
            chunk_paths,
            language=language,
            chunk_seconds=chunk_minutes * 60,
        )

        original_text = transcript_result.full_text

        # --------------------------------------------------------------
        # Step 3 - Translation / cleanup
        # --------------------------------------------------------------

        if language.lower() == "hinglish":

            status.write(
                "Step 3/6 - Preserving original-language transcript..."
            )

            original_text = transcribe_all_original(
                chunk_paths,
                language=language,
            )

            status.write(
                "Step 3/6 - Translating to clean English..."
            )

            # Sarvam translation mode already returns English.
            clean_text = transcript_result.full_text

        elif language.lower() == "english":

            status.write(
                "Step 3/6 - Cleaning English transcript..."
            )

            clean_text = clean_transcript(
                transcript_result.full_text
            )

        else:

            status.write(
                "Step 3/6 - Translating to clean English..."
            )

            clean_text = translate_to_clean_english(
                transcript_result.full_text,
                source_language=language.lower(),
            )

        # --------------------------------------------------------------
        # Validate transcript
        # --------------------------------------------------------------

        if not clean_text.strip():

            raise RuntimeError(
                "Transcription produced no text. "
                "The audio may be silent or unsupported."
            )

        # --------------------------------------------------------------
        # Step 4 - Summarization
        # --------------------------------------------------------------

        status.write(
            "Step 4/6 - Generating summary..."
        )

        summary = summarize_transcript(
            clean_text
        )

        # --------------------------------------------------------------
        # Step 5 - Extraction
        # --------------------------------------------------------------

        status.write(
            "Step 5/6 - Extracting action items, "
            "decisions & questions..."
        )

        extraction = extract_action_items(
            clean_text
        )

        # --------------------------------------------------------------
        # Step 6 - Vector store
        # --------------------------------------------------------------

        status.write(
            "Step 6/6 - Building searchable index for chat..."
        )

        index_transcript(
            session_id,
            transcript_result.segments,
        )

        # --------------------------------------------------------------
        # Save results in Streamlit session
        # --------------------------------------------------------------

        st.session_state.session_id = session_id

        st.session_state.clean_transcript = clean_text

        st.session_state.original_transcript = original_text

        st.session_state.summary = summary

        st.session_state.extraction = extraction

        st.session_state.processed = True

        st.session_state.processing_error = None

        st.session_state.chat_history = []

        status.update(
            label="Processing complete!",
            state="complete",
            expanded=False,
        )

    # ------------------------------------------------------------------
    # Known errors
    # ------------------------------------------------------------------

    except AudioProcessingError as exc:

        status.update(
            label="Audio processing failed",
            state="error",
        )

        st.session_state.processing_error = (
            f"Audio error: {exc}"
        )

    except TranscriptionError as exc:

        status.update(
            label="Transcription failed",
            state="error",
        )

        st.session_state.processing_error = (
            f"Transcription error: {exc}"
        )

    except TranslationError as exc:

        status.update(
            label="Translation failed",
            state="error",
        )

        st.session_state.processing_error = (
            f"Translation error: {exc}"
        )

    except SummarizationError as exc:

        status.update(
            label="Summarization failed",
            state="error",
        )

        st.session_state.processing_error = (
            f"Summarization error: {exc}"
        )

    except ExtractionError as exc:

        status.update(
            label="Extraction failed",
            state="error",
        )

        st.session_state.processing_error = (
            f"Extraction error: {exc}"
        )

    except VectorStoreError as exc:

        status.update(
            label="Indexing failed",
            state="error",
        )

        st.session_state.processing_error = (
            f"Vector store error: {exc}"
        )

    except Exception as exc:

        status.update(
            label="Unexpected error",
            state="error",
        )

        logger.exception(
            "Unexpected pipeline failure"
        )

        st.session_state.processing_error = (
            f"Unexpected error: {exc}"
        )

    finally:

        cleanup_chunks(chunk_paths)


# --------------------------------------------------------------------------
# Sidebar - Input
# --------------------------------------------------------------------------

st.title(
    "🎬 AI Video/Meeting Assistant"
)

st.caption(
    "Transcribe, summarize, and chat with any video "
    "or meeting recording."
)


# --------------------------------------------------------------------------
# Configuration check
# --------------------------------------------------------------------------

config_problems = validate_settings(
    require_gemini=True,
    require_sarvam=False,
)

if config_problems:

    st.warning(
        "\n".join(config_problems)
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

with st.sidebar:

    st.header("Input")

    input_mode = st.radio(
        "Source",
        [
            "YouTube URL",
            "Upload file",
        ],
        horizontal=True,
    )

    source_value: str | None = None

    uploaded_temp_path: str | None = None

    # --------------------------------------------------------------
    # YouTube
    # --------------------------------------------------------------

    if input_mode == "YouTube URL":

        source_value = st.text_input(
            "YouTube URL",
            placeholder=(
                "https://www.youtube.com/watch?v=..."
            ),
        )

    # --------------------------------------------------------------
    # Upload
    # --------------------------------------------------------------

    else:

        uploaded = st.file_uploader(
            "Audio or video file",
            type=[
                "mp4",
                "mp3",
                "wav",
                "m4a",
                "mov",
                "mkv",
                "webm",
            ],
        )

        if uploaded is not None:

            suffix = (
                Path(uploaded.name).suffix
                or ".bin"
            )

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as tmp:

                tmp.write(
                    uploaded.getbuffer()
                )

                uploaded_temp_path = tmp.name

            source_value = uploaded_temp_path

    # --------------------------------------------------------------
    # Language
    # --------------------------------------------------------------

    language = st.selectbox(
        "Language",
        [
            "Auto Detect",
            "English",
            "Telugu",
            "Hindi",
            "Hinglish",
            "Tamil",
            "Kannada",
            "Malayalam",
            "Marathi",
            "Bengali",
            "Gujarati",
            "Punjabi",
        ],
        help=(
            "Auto Detect automatically identifies the spoken language. "
            "English uses Whisper; Indic languages use Sarvam AI."
        ),
    )

    # --------------------------------------------------------------
    # Audio chunk length
    # --------------------------------------------------------------

    chunk_minutes = st.slider(
        "Audio chunk length (minutes)",
        2,
        20,
        10,
    )

    # --------------------------------------------------------------
    # Process button
    # --------------------------------------------------------------

    process_clicked = st.button(
        "🚀 Process",
        type="primary",
        use_container_width=True,
        disabled=not source_value,
    )

    # --------------------------------------------------------------
    # New session
    # --------------------------------------------------------------

    if st.session_state.processed:

        st.divider()

        if st.button(
            "🔄 Start new session",
            use_container_width=True,
        ):

            st.session_state.session_id = None

            st.session_state.processed = False

            st.session_state.clean_transcript = ""

            st.session_state.original_transcript = ""

            st.session_state.summary = None

            st.session_state.extraction = None

            st.session_state.chat_history = []

            st.session_state.processing_error = None

            st.rerun()


# --------------------------------------------------------------------------
# Process input
# --------------------------------------------------------------------------

if process_clicked and source_value:

    # Convert UI language names into language codes
    language_map = {
        "auto detect": "auto",
        "english": "en",
        "telugu": "te",
        "hindi": "hi",
        "hinglish": "hinglish",
        "tamil": "ta",
        "kannada": "kn",
        "malayalam": "ml",
        "marathi": "mr",
        "bengali": "bn",
        "gujarati": "gu",
        "punjabi": "pa",
    }

    selected_language = language_map.get(
        language.lower(),
        language.lower(),
    )

    run_pipeline(
        source_value,
        selected_language,
        chunk_minutes,
    )


# --------------------------------------------------------------------------
# Processing error
# --------------------------------------------------------------------------

if st.session_state.processing_error:

    st.error(
        st.session_state.processing_error
    )


# --------------------------------------------------------------------------
# Main content
# --------------------------------------------------------------------------

if not st.session_state.processed:

    st.info(
        "Provide a YouTube URL or upload a file, "
        "then click **Process** to get started."
    )

else:

    summary = st.session_state.summary

    extraction = st.session_state.extraction

    # ----------------------------------------------------------------------
    # Summary heading
    # ----------------------------------------------------------------------

    st.subheader(
        summary.title
    )

    st.write(
        summary.overview
    )

    # ----------------------------------------------------------------------
    # Tabs
    # ----------------------------------------------------------------------

    tabs = st.tabs(
        [
            "📝 Summary",
            "✅ Action Items",
            "📄 Transcript",
            "⬇️ Export",
        ]
    )

    # ======================================================================
    # Summary tab
    # ======================================================================

    with tabs[0]:

        st.markdown(
            "### Summary"
        )

        st.write(
            summary.summary
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                "### Important Points"
            )

            for point in (
                summary.important_points
                or []
            ):

                st.markdown(
                    f"- {point}"
                )

        with col2:

            st.markdown(
                "### Key Takeaways"
            )

            for takeaway in (
                summary.key_takeaways
                or []
            ):

                st.markdown(
                    f"- {takeaway}"
                )

    # ======================================================================
    # Action Items tab
    # ======================================================================

    with tabs[1]:

        st.markdown(
            "### Action Items"
        )

        if extraction.action_items:

            st.table(
                [
                    {
                        "Action": item.action,
                        "Owner": item.owner,
                        "Deadline": item.deadline,
                        "Status": item.status,
                    }
                    for item in extraction.action_items
                ]
            )

        else:

            st.caption(
                "No action items found."
            )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                "### Decisions"
            )

            for decision in (
                extraction.decisions
                or ["None found."]
            ):

                st.markdown(
                    f"- {decision}"
                )

        with col2:

            st.markdown(
                "### Open Questions"
            )

            for question in (
                extraction.open_questions
                or ["None found."]
            ):

                st.markdown(
                    f"- {question}"
                )

        with col3:

            st.markdown(
                "### Follow-ups"
            )

            for follow_up in (
                extraction.follow_ups
                or ["None found."]
            ):

                st.markdown(
                    f"- {follow_up}"
                )

    # ======================================================================
    # Transcript tab
    # ======================================================================

    with tabs[2]:

        if (
            st.session_state.original_transcript.strip()
            != st.session_state.clean_transcript.strip()
        ):

            st.markdown(
                "### Original Transcript"
            )

            st.text_area(
                "Original",
                st.session_state.original_transcript,
                height=200,
                label_visibility="collapsed",
            )

        st.markdown(
            "### Clean Transcript"
        )

        st.text_area(
            "Clean",
            st.session_state.clean_transcript,
            height=300,
            label_visibility="collapsed",
        )

    # ======================================================================
    # Chat
    # ======================================================================

    st.divider()

    st.markdown(
        "## 💬 Chat with this video/meeting"
    )

    for role, text in st.session_state.chat_history:

        with st.chat_message(role):

            st.write(text)

    question = st.chat_input(
        "Ask something about this video/meeting..."
    )

    if question:

        st.session_state.chat_history.append(
            ("user", question)
        )

        with st.chat_message("user"):

            st.write(question)

        try:

            with st.spinner(
                "Thinking..."
            ):

                result = ask(
                    st.session_state.session_id,
                    question,
                )

            answer_text = result.answer

            if result.timestamps:

                answer_text += (
                    "\n\n*Referenced around: "
                    + ", ".join(result.timestamps)
                    + "*"
                )

        except RagError as exc:

            answer_text = f"⚠️ {exc}"

        st.session_state.chat_history.append(
            ("assistant", answer_text)
        )

        with st.chat_message("assistant"):

            st.write(answer_text)

    # ======================================================================
    # Export tab
    # ======================================================================

    with tabs[3]:

        st.markdown(
            "### Export report"
        )

        report_data = ReportData(
            title=summary.title,
            summary=summary,
            extraction=extraction,
            transcript=st.session_state.clean_transcript,
            original_transcript=st.session_state.original_transcript,
        )

        # --------------------------------------------------------------
        # TXT report
        # --------------------------------------------------------------

        txt_report = build_txt_report(
            report_data
        )

        # --------------------------------------------------------------
        # JPG report
        # --------------------------------------------------------------

        try:

            jpg_bytes = build_jpg_report(
                report_data
            )

        except Exception as exc:

            jpg_bytes = None

            st.error(
                f"Could not build JPG: {exc}"
            )

        col1, col2, col3 = st.columns(3)

        # --------------------------------------------------------------
        # TXT
        # --------------------------------------------------------------

        with col1:

            st.download_button(
                "⬇️ Download TXT",
                data=txt_report,
                file_name="report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # --------------------------------------------------------------
        # PDF
        # --------------------------------------------------------------

        with col2:

            try:

                pdf_bytes = build_pdf_report(
                    report_data
                )

                st.download_button(
                    "⬇️ Download PDF",
                    data=pdf_bytes,
                    file_name="report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )

            except Exception as exc:

                st.error(
                    f"Could not build PDF: {exc}"
                )

        # --------------------------------------------------------------
        # JPG
        # --------------------------------------------------------------

        with col3:

            if jpg_bytes:

                st.download_button(
                    "🖼️ Download JPG",
                    data=jpg_bytes,
                    file_name="report.jpg",
                    mime="image/jpeg",
                    use_container_width=True,
                )