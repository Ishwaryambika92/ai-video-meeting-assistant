"""
Master test runner for AI Video & Meeting Assistant.
Runs all subsystem test scripts and reports results.
"""

try:
    import pyarrow
except ImportError:
    pass

import os
import subprocess
import sys
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

TESTS = [
    ("Gemini API", "test_gemini.py"),
    ("Action & Decision Extractor", "test_extractor.py"),
    ("Meeting Summariser (Structured)", "test_summariser.py"),
    ("Meeting Summary (Free-form)", "test_summary.py"),
    ("LLM Connectivity", "test_mistral.py"),
    ("Translator & Cleanup", "test_translator.py"),
    ("ChromaDB Vector Store", "test_vector_store.py"),
    ("RAG Engine", "test_rag.py"),
    ("Transcriber & Whisper", "test_transcriber.py"),
    ("Meeting Store & Task Database", "test_meeting_store.py"),
    ("Cross-Meeting Intelligence & Memory", "test_cross_meeting_rag.py"),
    ("Decisions Repository & Normalization", "test_decisions_repo.py"),
    ("Authentication & Security", "test_auth.py"),
    ("Report Generator (TXT, PDF, JPG)", "test_report_generator.py"),
]


def main():
    root = Path(__file__).resolve().parent
    python_exe = sys.executable

    print("\n" + "=" * 70)
    print("RUNNING ALL SUBSYSTEM TESTS")
    print("=" * 70)

    passed = 0
    failed = 0
    failures = []

    for name, test_file in TESTS:
        test_path = root / test_file
        print(f"\n[RUNNING] {name} ({test_file})...", flush=True)
        res = subprocess.run(
            [python_exe, str(test_path)],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if res.returncode == 0:
            print(f"[PASSED]  {name}", flush=True)
            passed += 1
        else:
            print(f"[FAILED]  {name} (Exit code: {res.returncode})", flush=True)
            if res.stderr:
                print(f"Error: {res.stderr.strip()[:300]}", flush=True)
            if res.stdout:
                print(f"Output: {res.stdout.strip()[:300]}")
            failed += 1
            failures.append((name, test_file))

    print("\n" + "=" * 70)
    print(f"TEST SUMMARY: {passed} PASSED, {failed} FAILED (Total: {len(TESTS)})")
    print("=" * 70)

    if failed > 0:
        print("Failed tests:")
        for name, test_file in failures:
            print(f"  - {name} ({test_file})")
        sys.exit(1)
    else:
        print("ALL SUBSYSTEM TESTS PASSED SUCCESSFULLY! ALL COMPONENTS OPERATIONAL.")
        sys.exit(0)

if __name__ == "__main__":
    main()
