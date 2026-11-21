import sys
import chromadb
from sentence_transformers import SentenceTransformer
from ingest import extract_pages, chunk_pages

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else "sample.pdf"

model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chroma_db")

# Start fresh so re-running doesn't create duplicates
try:
    client.delete_collection("pdf_chunks")
except Exception:
    pass

# Tell Chroma to use cosine distance (matches what we learned in Step 3)
collection = client.create_collection(
    name="pdf_chunks",
    metadata={"hnsw:space": "cosine"},
)

pages = extract_pages(PDF_PATH)
chunks = chunk_pages(pages)
print(f"{len(pages)} pages -> {len(chunks)} chunks")

texts = [c["text"] for c in chunks]
vectors = model.encode(texts, show_progress_bar=True)

collection.add(
    ids=[f"chunk_{i}" for i in range(len(chunks))],
    embeddings=vectors.tolist(),
    documents=texts,
    metadatas=[{"page": c["page"], "source": PDF_PATH} for c in chunks],
)

print(f"Stored {collection.count()} chunks in chroma_db/")