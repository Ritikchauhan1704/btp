import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json

# Load the knowledge base
with open("knowledge_base.json", "r") as f:
    knowledge_base = json.load(f)

# Initialize SentenceTransformer for embedding disease descriptions
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Prepare data for FAISS
disease_names = [entry["disease"] for entry in knowledge_base]
descriptions = [entry["description"] for entry in knowledge_base]
symptoms_list = [entry["symptoms"] for entry in knowledge_base]

# Encode descriptions into embeddings
description_embeddings = model.encode(descriptions)

# Create FAISS index
dimension = description_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(description_embeddings))

print(f"FAISS index created with {index.ntotal} entries.")
