import sys
from pathlib import Path

# Configure UTF-8 console output for Windows terminal support
if hasattr(sys.stdout, "reconfigure"):
    getattr(sys.stdout, "reconfigure")(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    getattr(sys.stderr, "reconfigure")(encoding="utf-8", errors="replace")

_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from core.translator import clean_transcript, translate_to_clean_english


print("=" * 60)
print("TEST 1: BASIC ENGLISH TRANSCRIPT CLEANUP")
print("=" * 60)

text = "um hello everyone uh today we are going to discuss the backend integration"

try:
    result = clean_transcript(text)

    print("\nORIGINAL:")
    print(text)

    print("\nCLEANED:")
    print(result)

except Exception as e:
    print("\nTRANSLATION TEST FAILED:")
    print(e)


print("\n" + "=" * 60)
print("TEST 2: ADVANCED CLEANUP (FILLERS, STUTTERS, STT RECOGNITION ERRORS)")
print("=" * 60)

complex_text = (
    "Speaker 1 (00:00): um hello uh hello everyone can you guys hear me okay uh so like basically today "
    "we we are going to talk about the cooper nettles cluster deployment and uh you know the latency issues "
    "that we saw yesterday in the staging environment.\n\n"
    "Speaker 2 (00:25): yeah yeah absolutely Alex i think uh the main problem was with the sink between "
    "the data base and the cash server. like it was taking almost four seconds for each query which is way "
    "to slow and the customers was complaining.\n\n"
    "Speaker 1 (00:48): right right so we need to to fix the query indexing by this friday October 24th and uh "
    "assign someone like Deepak to monitor the p99 latency asap. also don't forget the api end points documentation."
)

try:
    complex_result = clean_transcript(complex_text)

    print("\nORIGINAL RAW TRANSCRIPT:")
    print(complex_text)

    print("\nCLEANED & ENHANCED TRANSCRIPT:")
    print(complex_result)

except Exception as e:
    print("\nADVANCED CLEANUP TEST FAILED:")
    print(e)


print("\n" + "=" * 60)
print("TEST 3: HINDI -> CLEAN ENGLISH")
print("=" * 60)

hindi_text = "नमस्ते, आज हम अपने एप्लिकेशन के बैकएंड के बारे में चर्चा करेंगे।"

try:
    result = translate_to_clean_english(
        hindi_text,
        source_language="hi",
    )

    print("\nORIGINAL (HINDI):")
    print(hindi_text)

    print("\nTRANSLATED & CLEANED (ENGLISH):")
    print(result)

except Exception as e:
    print("\nHINDI TRANSLATION TEST FAILED:")
    print(e)