import chromadb
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_collection("pdf_chunks")

print(f"Loaded {collection.count()} chunks from disk.")

while True:
    question = input("\nAsk a question (or 'quit'): ")
    if question.lower() == "quit":
        break

    q_vector = model.encode(question).tolist()

    results = collection.query(
        query_embeddings=[q_vector],
        n_results=3,
    )

    # Results come back as lists of lists (one inner list per query)
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for rank, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        similarity = 1 - dist
        print(f"\n#{rank} (similarity {similarity:.2f}) [page {meta['page']}]")
        print(doc[:300])