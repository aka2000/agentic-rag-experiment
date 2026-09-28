import numpy as np

from app.embeddings import get_embeddings


def cosine_similarity(vector_a, vector_b):

    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def semantic_search(chunks, query, top_k=3):

    embeddings = get_embeddings()

    # Convert the question into an embedding
    query_vector = embeddings.embed_query(query)

    results = []

    for chunk in chunks:

        # Convert each chunk into an embedding
        chunk_vector = embeddings.embed_query(
            chunk.page_content
        )

        similarity = cosine_similarity(
            query_vector,
            chunk_vector
        )

        results.append(
            (similarity, chunk)
        )

    # Highest similarity first
    results.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return results[:top_k]