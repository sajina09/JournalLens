from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

import argparse

CHROMA_PATH = "chroma"

def main():
    # Create CLI. in order to run from terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    model_name = "all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Search the DB.
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    # Add checks for the results
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    # Display results
    print("Matching results with relevance scores:")
    for result, score in results:
        print(f"Document: {result}")
        print(f"Relevance Score: {score:.2f}")
        print("-" * 50)


if __name__ == "__main__":
    main()