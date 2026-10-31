"""
Embedding generation and ChromaDB vector storage for RAG.

Each processed video/meeting gets its own Chroma collection (named after a
session id) so multiple sessions never mix context, and old sessions can
be dropped independently.

Timestamps are carried through as chunk metadata (not invented - derived
from the transcript segments produced by the transcriber) so RAG answers
can cite roughly where in the recording something was discussed.
"""

from __future__ import annotations

try:
    import pyarrow
except ImportError:
    pass

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from functools import lru_cache
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from langchain_chroma import Chroma
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
else:
    try:
        from langchain_chroma import Chroma
        from langchain_core.documents import Document
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        Chroma = None
        Document = None
        HuggingFaceEmbeddings = None
        RecursiveCharacterTextSplitter = None

from config import CHROMA_DIR, get_device, get_logger, settings
from core.transcriber import TranscriptSegment, _format_timestamp

logger = get_logger(__name__)


class VectorStoreError(Exception):
    pass


@lru_cache(maxsize=1)
def _get_embeddings() -> Any:
    """Load (and cache) the HuggingFace embedding model on the best device."""
    global HuggingFaceEmbeddings
    if HuggingFaceEmbeddings is None:
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except Exception as exc:
            raise VectorStoreError(f"langchain-huggingface is not installed or failed to load: {exc}") from exc

    logger.info("Loading embedding model '%s' on %s...", settings.embedding_model, get_device())
    try:
        return HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": get_device()},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as exc:
        raise VectorStoreError(f"Failed to initialize HuggingFaceEmbeddings: {exc}") from exc


def _collection_name(session_id: str) -> str:
    # Chroma collection names must be alphanumeric/underscore/hyphen.
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return f"session_{safe}"[:63]


def build_documents_from_segments(segments: list[TranscriptSegment]) -> list[Document]:
    """Turn transcript segments into overlapping Documents with real
    timestamp metadata, further split for embedding-friendly chunk sizes.
    """
    global RecursiveCharacterTextSplitter, Document
    if RecursiveCharacterTextSplitter is None or Document is None:
        try:
            from langchain_core.documents import Document
            from langchain_text_splitters import RecursiveCharacterTextSplitter
        except Exception as exc:
            raise VectorStoreError(f"langchain dependencies are not installed: {exc}") from exc

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.transcript_chunk_size,
        chunk_overlap=settings.transcript_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for seg in segments:
        text = getattr(seg, "text", "") if hasattr(seg, "text") else (seg.get("text", "") if isinstance(seg, dict) else str(seg))
        if not text or not text.strip():
            continue

        start_sec = getattr(seg, "start_seconds", 0.0) if hasattr(seg, "start_seconds") else (seg.get("start_seconds", 0.0) if isinstance(seg, dict) else 0.0)
        end_sec = getattr(seg, "end_seconds", 0.0) if hasattr(seg, "end_seconds") else (seg.get("end_seconds", 0.0) if isinstance(seg, dict) else 0.0)
        
        timestamp = getattr(seg, "start_timestamp", None)
        if not timestamp and isinstance(seg, dict):
            timestamp = seg.get("timestamp") or seg.get("start_timestamp")
        if not timestamp:
            timestamp = _format_timestamp(float(start_sec))

        engine = getattr(seg, "engine", "whisper") if hasattr(seg, "engine") else (seg.get("engine", "whisper") if isinstance(seg, dict) else "whisper")

        sub_texts = splitter.split_text(text)
        for sub in sub_texts:
            clean_sub = sub.strip()
            if not clean_sub:
                continue
            documents.append(
                Document(
                    page_content=clean_sub,
                    metadata={
                        "start_seconds": float(start_sec),
                        "end_seconds": float(end_sec),
                        "timestamp": str(timestamp),
                        "engine": str(engine),
                    },
                )
            )
    return documents


def index_transcript(session_id: str, segments: list[TranscriptSegment]) -> Any:
    """Embed transcript segments and (re)build the Chroma collection for
    this session. Returns the Chroma vector store handle.
    """
    global Chroma
    if Chroma is None:
        try:
            from langchain_chroma import Chroma
        except Exception as exc:
            raise VectorStoreError(f"langchain-chroma is not installed: {exc}") from exc

    documents = build_documents_from_segments(segments)
    if not documents:
        raise VectorStoreError("No transcript content to index.")

    try:
        store = Chroma(
            collection_name=_collection_name(session_id),
            embedding_function=_get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
        # Reset any previous run for this session before adding fresh docs.
        existing_ids = store.get().get("ids", [])
        if existing_ids:
            store.delete(ids=existing_ids)

        # Batch insert documents to avoid ChromaDB single-request payload limits
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            store.add_documents(documents[i : i + batch_size])
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to index transcript into ChromaDB: {exc}") from exc


    logger.info("Indexed %d chunks for session '%s'.", len(documents), session_id)
    return store


def get_retriever(session_id: str, k: int = settings.rag_top_k) -> Any:
    """Get a retriever over an already-indexed session."""
    global Chroma
    if Chroma is None:
        try:
            from langchain_chroma import Chroma
        except Exception as exc:
            raise VectorStoreError(f"langchain-chroma is not installed: {exc}") from exc

    try:
        store = Chroma(
            collection_name=_collection_name(session_id),
            embedding_function=_get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to open ChromaDB collection: {exc}") from exc

    return store.as_retriever(search_kwargs={"k": k})


def delete_session(session_id: str) -> None:
    """Delete a session's vectors entirely (used when starting fresh)."""
    if Chroma is None:
        return
    try:
        store = Chroma(
            collection_name=_collection_name(session_id),
            embedding_function=_get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )
        ids = store.get().get("ids", [])
        if ids:
            store.delete(ids=ids)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not clean up session '%s': %s", session_id, exc)

