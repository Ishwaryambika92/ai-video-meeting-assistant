from core.translator import clean_transcript, translate_to_clean_english


print("=" * 60)
print("TEST 1: ENGLISH TRANSCRIPT CLEANUP")
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
print("TEST 2: HINDI → ENGLISH")
print("=" * 60)

hindi_text = "नमस्ते, आज हम अपने एप्लिकेशन के बैकएंड के बारे में चर्चा करेंगे।"

try:
    result = translate_to_clean_english(
        hindi_text,
        source_language="hi",
    )

    print("\nORIGINAL:")
    print(hindi_text)

    print("\nENGLISH:")
    print(result)

except Exception as e:
    print("\nHINDI TRANSLATION TEST FAILED:")
    print(e)