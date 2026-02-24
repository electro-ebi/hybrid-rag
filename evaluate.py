from data_layer.retrieval import retrieve_chunks

TEST_QUERIES = [
    {
        "question": "What does the paper study?",
        "expected_keyword": "permanents"
    },
    {
        "question": "What distribution does perm(U) follow?",
        "expected_keyword": "Gaussian"
    }
]


def evaluate(mode="hybrid"):
    print(f"\nEvaluating mode: {mode}\n")

    correct = 0

    for test in TEST_QUERIES:
        question = test["question"]
        expected = test["expected_keyword"]

        chunks = retrieve_chunks(question, top_k=3, mode=mode)

        combined = " ".join(chunks).lower()

        hit = expected.lower() in combined

        print(f"Q: {question}")
        print(f"Expected keyword: {expected}")
        print(f"Hit: {hit}")
        print("-" * 50)

        if hit:
            correct += 1

    print(f"\nAccuracy: {correct}/{len(TEST_QUERIES)}")


if __name__ == "__main__":
    evaluate("dense")
    evaluate("sparse")
    evaluate("hybrid")