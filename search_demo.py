from sentence_transformers import SentenceTransformer, util
from ingest import extract_pages, chunk_pages

model = SentenceTransformer("all-MiniLM-L6-v2")

pages = extract_pages("sample.pdf")
chunks = chunk_pages(pages)

# Embed every chunk (one vector per chunk)
texts = [c["text"] for c in chunks]
chunk_vectors = model.encode(texts, show_progress_bar=True)

while True:
    question = input("\nAsk a question (or 'quit'): ")
    if question.lower() == "quit":
        break

    q_vector = model.encode(question)
    scores = util.cos_sim(q_vector, chunk_vectors)[0]

    # Get the indices of the 3 highest scores
    top_ids = scores.argsort(descending=True)[:3]

    for rank, idx in enumerate(top_ids, start=1):
        idx = int(idx)
        print(f"\n#{rank} (score {float(scores[idx]):.2f}) [page {chunks[idx]['page']}]")
        print(chunks[idx]["text"][:300])