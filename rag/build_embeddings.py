
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import faiss
import os

os.makedirs("embeddings", exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("data/processed/chunks.json", "r") as f:
    chunks = json.load(f)

texts = [c["text"] for c in chunks]

embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")

np.save("embeddings/vectors.npy", embeddings)

with open("embeddings/meta.json", "w") as f:
    json.dump(chunks, f)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

faiss.write_index(index, "embeddings/index.faiss")

print("✅ Embeddings built")
