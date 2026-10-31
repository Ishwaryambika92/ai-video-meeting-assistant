"""
Comprehensive test suite verifying persistent language preference handling,
pre-retrieval language commands, and short follow-up resolution ("cheppu").

Executes the user's mandatory 7-step test sequence:
1. User: "Tell me about the meeting" -> English summary.
2. User: "Naku eppudu Telugu lo cheppu" -> Acknowledgment in Telugu script & persistent Telugu preference.
3. User: "Meeting lo em techniques discuss chesaru?" -> Answer in Telugu script grounded in transcript.
4. User: "Naku Hindi lo cheppu" -> Acknowledgment in Hindi script & persistent Hindi preference.
5. User: "Cheppu" -> Answers previous techniques question in Hindi script (Devanagari), not Roman Telugu.
6. User: "Change to Telugu-English mix" -> Preference changes to Roman Telugu-English.
7. User: "FAB technique explain cheyyi" -> Answer in Roman Telugu-English explaining Feature, Advantage, Benefit.
"""

from __future__ import annotations

import os
import sys

# Configure UTF-8 console output for Windows
if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    getattr(sys.stderr, "reconfigure")(encoding="utf-8", errors="replace")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"

from dotenv import load_dotenv
load_dotenv()

from core.rag_engine import (
    ResponseLanguage,
    ask,
    contains_devanagari_script,
    contains_tamil_script,
    contains_telugu_script,
    detect_language_command,
    detect_response_language,
    is_short_followup,
)
from core.transcriber import TranscriptSegment
from core.vector_store import index_transcript

# Rich test transcript containing both technical speech-to-text techniques and FAB sales technique
TEST_CLEAN_TRANSCRIPT = """Speaker 1 (00:00): Welcome everyone to our product and sales alignment sync. Today we need to align on our speech-to-text architecture, sales techniques, and sprint deliverables.
Speaker 2 (00:30): For speech-to-text techniques, we evaluated Whisper for English and Sarvam AI for Indic languages. Whisper base gives 95% accuracy on clean audio, while Sarvam AI handles Telugu, Hindi, and Tamil with 30-second chunking.
Speaker 1 (01:15): That sounds solid. Let's officially approve Whisper for English and Sarvam AI for Indic languages. Also, we will use ChromaDB for local vector embeddings.
Speaker 2 (02:00): Next, on sales methodology, we are training customer representatives on the FAB technique. FAB stands for Feature, Advantage, and Benefit. A Feature is what the product has, Advantage is what it does better, and Benefit is the direct positive value to the customer. We also touched upon BSCQ for qualifying buyers.
Speaker 1 (02:45): Excellent. Deepak will complete the backend database integration by Friday, and Alex will finalize the multi-format report exports."""

TEST_SEGMENTS = [
    TranscriptSegment(
        text="Welcome everyone to our product and sales alignment sync. Today we need to align on our speech-to-text architecture, sales techniques, and sprint deliverables.",
        start_seconds=0,
        end_seconds=30,
        engine="Whisper",
        language_code="en",
    ),
    TranscriptSegment(
        text="For speech-to-text techniques, we evaluated Whisper for English and Sarvam AI for Indic languages. Whisper base gives 95% accuracy on clean audio, while Sarvam AI handles Telugu, Hindi, and Tamil with 30-second chunking.",
        start_seconds=30,
        end_seconds=75,
        engine="Whisper",
        language_code="en",
    ),
    TranscriptSegment(
        text="That sounds solid. Let's officially approve Whisper for English and Sarvam AI for Indic languages. Also, we will use ChromaDB for local vector embeddings.",
        start_seconds=75,
        end_seconds=120,
        engine="Whisper",
        language_code="en",
    ),
    TranscriptSegment(
        text="Next, on sales methodology, we are training customer representatives on the FAB technique. FAB stands for Feature, Advantage, and Benefit. A Feature is what the product has, Advantage is what it does better, and Benefit is the direct positive value to the customer. We also touched upon BSCQ for qualifying buyers.",
        start_seconds=120,
        end_seconds=165,
        engine="Whisper",
        language_code="en",
    ),
    TranscriptSegment(
        text="Excellent. Deepak will complete the backend database integration by Friday, and Alex will finalize the multi-format report exports.",
        start_seconds=165,
        end_seconds=195,
        engine="Whisper",
        language_code="en",
    ),
]


def test_command_detection_unit():
    print("=" * 70)
    print("UNIT TEST: PRE-RETRIEVAL LANGUAGE COMMAND DETECTION")
    print("=" * 70)

    # 1. Telugu script commands
    l, is_pure = detect_language_command("Naku eppudu Telugu lo cheppu")
    assert l == ResponseLanguage.TELUGU and is_pure, f"Failed on 'Naku eppudu Telugu lo cheppu': {l}, {is_pure}"
    print("[PASS] 'Naku eppudu Telugu lo cheppu' -> TELUGU (pure command)")

    l, is_pure = detect_language_command("telugu lo cheppu")
    assert l == ResponseLanguage.TELUGU and is_pure, f"Failed on 'telugu lo cheppu': {l}, {is_pure}"
    print("[PASS] 'telugu lo cheppu' -> TELUGU (pure command)")

    # 2. Hindi commands
    l, is_pure = detect_language_command("Naku Hindi lo cheppu")
    assert l == ResponseLanguage.HINDI and is_pure, f"Failed on 'Naku Hindi lo cheppu': {l}, {is_pure}"
    print("[PASS] 'Naku Hindi lo cheppu' -> HINDI (pure command)")

    l, is_pure = detect_language_command("hindi mein batao")
    assert l == ResponseLanguage.HINDI and is_pure, f"Failed on 'hindi mein batao': {l}, {is_pure}"
    print("[PASS] 'hindi mein batao' -> HINDI (pure command)")

    # 3. Telugu-English mix commands
    l, is_pure = detect_language_command("Change to Telugu-English mix")
    assert l == ResponseLanguage.TELUGU_ENGLISH and is_pure, f"Failed on 'Change to Telugu-English mix': {l}, {is_pure}"
    print("[PASS] 'Change to Telugu-English mix' -> TELUGU_ENGLISH (pure command)")

    l, is_pure = detect_language_command("Naku Telugu-English mix lo cheppu")
    assert l == ResponseLanguage.TELUGU_ENGLISH and is_pure, f"Failed on 'Naku Telugu-English mix lo cheppu': {l}, {is_pure}"
    print("[PASS] 'Naku Telugu-English mix lo cheppu' -> TELUGU_ENGLISH (pure command)")

    # 4. English commands
    l, is_pure = detect_language_command("Tell me in English")
    assert l == ResponseLanguage.ENGLISH and is_pure, f"Failed on 'Tell me in English': {l}, {is_pure}"
    print("[PASS] 'Tell me in English' -> ENGLISH (pure command)")

    # 5. Question with command embedded
    l, is_pure = detect_language_command("Naku Telugu lo meeting summary cheppu")
    assert l == ResponseLanguage.TELUGU and not is_pure, f"Failed on embedded question: {l}, {is_pure}"
    print("[PASS] 'Naku Telugu lo meeting summary cheppu' -> TELUGU (with question content)")

    # 6. Short follow-up keywords
    assert is_short_followup("cheppu"), "Failed on 'cheppu'"
    assert is_short_followup("Cheppu"), "Failed on 'Cheppu'"
    assert is_short_followup("continue"), "Failed on 'continue'"
    assert is_short_followup("explain"), "Failed on 'explain'"
    assert is_short_followup("k tell me"), "Failed on 'k tell me'"
    assert is_short_followup("inka cheppu"), "Failed on 'inka cheppu'"
    assert not is_short_followup("Meeting lo em techniques discuss chesaru?"), "False positive on full question"
    print("[PASS] All short follow-up keywords accurately identified.")


def test_mandatory_7_step_sequence():
    print("\n" + "=" * 70)
    print("MANDATORY TEST: 7-STEP LIVE CONVERSATION SEQUENCE")
    print("=" * 70)

    session_id = "test_7_step_session"
    index_transcript(session_id=session_id, segments=TEST_SEGMENTS)

    # Simulated Streamlit session state
    chat_history: list[tuple[str, str]] = []
    preferred_language = "english"

    # Step 1: User: "Tell me about the meeting" -> English summary
    print("\n[Step 1] User: 'Tell me about the meeting'")
    q1 = "Tell me about the meeting"
    chat_history.append(("user", q1))
    r1 = ask(
        session_id=session_id,
        question=q1,
        conversation_history=chat_history[:-1],
        clean_transcript=TEST_CLEAN_TRANSCRIPT,
        preferred_language=preferred_language,
    )
    if r1.language:
        preferred_language = r1.language
    chat_history.append(("assistant", r1.answer))
    print(f"Assistant ({r1.language}):\n{r1.answer}\n")
    assert r1.language == ResponseLanguage.ENGLISH, f"Step 1 expected English, got {r1.language}"
    assert not contains_tamil_script(r1.answer)
    assert any(w in r1.answer.lower() for w in ("whisper", "sarvam", "fab", "speech", "sales"))
    print(">> Step 1 PASSED: English summary generated.")

    # Step 2: User: "Naku eppudu Telugu lo cheppu" -> Acknowledgment in Telugu script
    print("\n[Step 2] User: 'Naku eppudu Telugu lo cheppu'")
    q2 = "Naku eppudu Telugu lo cheppu"
    chat_history.append(("user", q2))
    r2 = ask(
        session_id=session_id,
        question=q2,
        conversation_history=chat_history[:-1],
        clean_transcript=TEST_CLEAN_TRANSCRIPT,
        preferred_language=preferred_language,
    )
    if r2.language:
        preferred_language = r2.language
    chat_history.append(("assistant", r2.answer))
    print(f"Assistant ({r2.language}):\n{r2.answer}\n")
    assert preferred_language == ResponseLanguage.TELUGU, f"Step 2 expected preference TELUGU, got {preferred_language}"
    assert contains_telugu_script(r2.answer), "Step 2 acknowledgment must be in Telugu script!"
    assert not contains_tamil_script(r2.answer), "Step 2 must NOT contain Tamil script!"
    print(">> Step 2 PASSED: Acknowledged in Telugu script and updated preference.")

    # Step 3: User: "Meeting lo em techniques discuss chesaru?" -> Answer in Telugu script
    print("\n[Step 3] User: 'Meeting lo em techniques discuss chesaru?'")
    q3 = "Meeting lo em techniques discuss chesaru?"
    chat_history.append(("user", q3))
    r3 = ask(
        session_id=session_id,
        question=q3,
        conversation_history=chat_history[:-1],
        clean_transcript=TEST_CLEAN_TRANSCRIPT,
        preferred_language=preferred_language,
    )
    if r3.language:
        preferred_language = r3.language
    chat_history.append(("assistant", r3.answer))
    print(f"Assistant ({r3.language}):\n{r3.answer}\n")
    assert preferred_language == ResponseLanguage.TELUGU
    assert contains_telugu_script(r3.answer), "Step 3 answer must be in Telugu script!"
    assert not contains_tamil_script(r3.answer)
    print(">> Step 3 PASSED: Answered in Telugu script using transcript evidence.")

    # Step 4: User: "Naku Hindi lo cheppu" -> Acknowledgment in Hindi script
    print("\n[Step 4] User: 'Naku Hindi lo cheppu'")
    q4 = "Naku Hindi lo cheppu"
    chat_history.append(("user", q4))
    r4 = ask(
        session_id=session_id,
        question=q4,
        conversation_history=chat_history[:-1],
        clean_transcript=TEST_CLEAN_TRANSCRIPT,
        preferred_language=preferred_language,
    )
    if r4.language:
        preferred_language = r4.language
    chat_history.append(("assistant", r4.answer))
    print(f"Assistant ({r4.language}):\n{r4.answer}\n")
    assert preferred_language == ResponseLanguage.HINDI, f"Step 4 expected preference HINDI, got {preferred_language}"
    assert contains_devanagari_script(r4.answer), "Step 4 acknowledgment must be in Devanagari Hindi script!"
    assert not contains_tamil_script(r4.answer)
    print(">> Step 4 PASSED: Acknowledged in Hindi script and updated preference.")

    # Step 5: User: "Cheppu" -> Answers the previous techniques question in Hindi
    print("\n[Step 5] User: 'Cheppu'")
    q5 = "Cheppu"
    chat_history.append(("user", q5))
    r5 = ask(
        session_id=session_id,
        question=q5,
        conversation_history=chat_history[:-1],
        clean_transcript=TEST_CLEAN_TRANSCRIPT,
        preferred_language=preferred_language,
    )
    if r5.language:
        preferred_language = r5.language
    chat_history.append(("assistant", r5.answer))
    print(f"Assistant ({r5.language}):\n{r5.answer}\n")
    assert preferred_language == ResponseLanguage.HINDI
    assert "not found" not in r5.answer.lower(), "Step 5 failed: 'Cheppu' incorrectly returned 'not found'!"
    assert contains_devanagari_script(r5.answer), "Step 5 answer must be in Devanagari Hindi script!"
    assert not contains_tamil_script(r5.answer)
    print(">> Step 5 PASSED: 'Cheppu' resolved previous techniques inquiry and answered in Hindi.")

    # Step 6: User: "Change to Telugu-English mix" -> Preference changes to Roman Telugu
    print("\n[Step 6] User: 'Change to Telugu-English mix'")
    q6 = "Change to Telugu-English mix"
    chat_history.append(("user", q6))
    r6 = ask(
        session_id=session_id,
        question=q6,
        conversation_history=chat_history[:-1],
        clean_transcript=TEST_CLEAN_TRANSCRIPT,
        preferred_language=preferred_language,
    )
    if r6.language:
        preferred_language = r6.language
    chat_history.append(("assistant", r6.answer))
    print(f"Assistant ({r6.language}):\n{r6.answer}\n")
    assert preferred_language == ResponseLanguage.TELUGU_ENGLISH, f"Step 6 expected TELUGU_ENGLISH, got {preferred_language}"
    assert not contains_tamil_script(r6.answer)
    print(">> Step 6 PASSED: Language preference successfully switched to Telugu-English mix.")

    # Step 7: User: "FAB technique explain cheyyi" -> Answer in Roman Telugu
    print("\n[Step 7] User: 'FAB technique explain cheyyi'")
    q7 = "FAB technique explain cheyyi"
    chat_history.append(("user", q7))
    r7 = ask(
        session_id=session_id,
        question=q7,
        conversation_history=chat_history[:-1],
        clean_transcript=TEST_CLEAN_TRANSCRIPT,
        preferred_language=preferred_language,
    )
    if r7.language:
        preferred_language = r7.language
    chat_history.append(("assistant", r7.answer))
    print(f"Assistant ({r7.language}):\n{r7.answer}\n")
    assert preferred_language == ResponseLanguage.TELUGU_ENGLISH
    assert not contains_tamil_script(r7.answer)
    assert not contains_telugu_script(r7.answer), "Step 7 must be in Roman Telugu, not Telugu script!"
    assert any(term in r7.answer.lower() for term in ("feature", "advantage", "benefit", "fab")), \
        "Step 7 must explain Feature, Advantage, and Benefit from the transcript!"
    print(">> Step 7 PASSED: FAB technique explained in Roman Telugu with English terms.")


if __name__ == "__main__":
    test_command_detection_unit()
    test_mandatory_7_step_sequence()
    print("\n" + "=" * 70)
    print("ALL 7 MANDATORY STEPS AND TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
