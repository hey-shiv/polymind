
import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

PERSONA_BOOK_MAP = {
    "Elon Musk": ["elon", "tesla", "spacex", "engineering"],
    "Robert Greene": ["greene", "laws", "mastery", "power"],
    "Steve Jobs": ["jobs", "apple", "design", "product"],
}

class Retriever:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.read_index("embeddings/index.faiss")

        with open("embeddings/meta.json", "r") as f:
            self.meta = json.load(f)

    def search_by_persona(self, query, persona, k=2):
      q_emb = self.model.encode([query]).astype("float32")

      # 🔥 increase search space
      D, I = self.index.search(q_emb, 100)

      persona_results = []
      general_results = []

      for i in I[0]:
          chunk = self.meta[i]
          text = (chunk.get("source", "") + chunk.get("text", "")).lower()

          if persona == "Elon Musk" and ("elon" in text or "tesla" in text or "spacex" in text):
              persona_results.append(chunk)

          elif persona == "Robert Greene" and ("greene" in text or "mastery" in text):
              persona_results.append(chunk)

          elif persona == "Steve Jobs" and ("jobs" in text or "apple" in text):
              persona_results.append(chunk)

          else:
              general_results.append(chunk)

          if len(persona_results) >= k:
              break

      # fallback
      if len(persona_results) < k:
          persona_results.extend(general_results[:k - len(persona_results)])

      return persona_results
