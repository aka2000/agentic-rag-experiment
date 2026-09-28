from app.ingestion import load_pdf, split_documents
from app.search import semantic_search


PDF_PATH = "documents/sample.pdf"


# ---------------------------------------
# 1. Load PDF
# ---------------------------------------

documents = load_pdf(PDF_PATH)

print(f"Pages loaded: {len(documents)}")


# ---------------------------------------
# 2. Split into chunks
# ---------------------------------------

chunks = split_documents(documents)

print(f"Chunks created: {len(chunks)}")


# ---------------------------------------
# 3. Search
# ---------------------------------------

query = "Who does this company policy apply to?"

results = semantic_search(
    chunks,
    query,
    top_k=3
)


# ---------------------------------------
# 4. Display results
# ---------------------------------------

print("\n")
print("=" * 70)
print("QUERY")
print("=" * 70)

print(query)


for rank, (score, chunk) in enumerate(
    results,
    start=1
):

    print("\n")
    print("=" * 70)
    print(f"RESULT {rank}")
    print("=" * 70)

    print(f"Similarity: {score:.4f}")

    print(
        f"Source: {chunk.metadata['source']}"
    )

    print(
        f"Page: {chunk.metadata['page']}"
    )

    print("\nText:")

    print(chunk.page_content)