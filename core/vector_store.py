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

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHROMA_DIR, get_device, get_logger, settings
from core.transcriber import TranscriptSegment

logger = get_logger(__name__)


class VectorStoreError(Exception):
    pass


@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    """Load (and cache) the HuggingFace embedding model on the best device."""
    logger.info("Loading embedding model '%s' on %s...", settings.embedding_model, get_device())
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": get_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


def _collection_name(session_id: str) -> str:
    # Chroma collection names must be alphanumeric/underscore/hyphen.
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in session_id)
    return f"session_{safe}"[:63]


def build_documents_from_segments(segments: list[TranscriptSegment]) -> list[Document]:
    """Turn transcript segments into overlapping Documents with real
    timestamp metadata, further split for embedding-friendly chunk sizes.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.transcript_chunk_size,
        chunk_overlap=settings.transcript_chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: list[Document] = []
    for seg in segments:
        if not seg.text.strip():
            continue
        sub_texts = splitter.split_text(seg.text)
        for sub in sub_texts:
            documents.append(
                Document(
                    page_content=sub,
                    metadata={
                        "start_seconds": seg.start_seconds,
                        "end_seconds": seg.end_seconds,
                        "timestamp": seg.start_timestamp,
                        "engine": seg.engine,
                    },
                )
            )
    return documents


def index_transcript(session_id: str, segments: list[TranscriptSegment]) -> Chroma:
    """Embed transcript segments and (re)build the Chroma collection for
    this session. Returns the Chroma vector store handle.
    """
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
        store.add_documents(documents)
    except Exception as exc:  # noqa: BLE001
        raise VectorStoreError(f"Failed to index transcript into ChromaDB: {exc}") from exc

    logger.info("Indexed %d chunks for session '%s'.", len(documents), session_id)
    return store


def get_retriever(session_id: str, k: int = settings.rag_top_k):
    """Get a retriever over an already-indexed session."""
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
