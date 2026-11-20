from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I reset my password?",
    "To change your login credentials, go to Settings.",
    "The cat sat on the warm windowsill.",
]

embeddings = model.encode(sentences)

print("Shape:", embeddings.shape)       # (3, 384)
print("First 5 numbers of sentence 1:", embeddings[0][:5])

# Compare sentence 1 against the other two
scores = util.cos_sim(embeddings[0], embeddings[1:])
print("\nPassword question vs credentials sentence:", float(scores[0][0]))
print("Password question vs cat sentence:        ", float(scores[0][1]))