from core.rag_engine import ask


SESSION_ID = "test_session"


questions = [
    "Who will complete the backend integration?",
    "What technology was selected for the application?",
    "Which database was selected?",
]


for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:")
    print(question)
    print("=" * 60)

    try:
        result = ask(
            session_id=SESSION_ID,
            question=question,
        )

        print("\nANSWER:")
        print(result.answer)

        print("\nTIMESTAMPS:")
        for timestamp in result.timestamps:
            print("-", timestamp)

        print("\nSOURCE CHUNKS:")
        print(result.source_chunks)

    except Exception as e:
        print("\nRAG TEST FAILED:")
        print(e)