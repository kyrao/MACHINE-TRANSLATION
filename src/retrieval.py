# src/retrieval.py
import json
from pathlib import Path
import faiss
import numpy as np

KB_PATH = Path(__file__).parent.parent / "kb" / "local_kb.json"

def load_kb():
    with open(KB_PATH, encoding='utf8') as f:
        return json.load(f)

def build_kb_index(embeddings, meta_list, dim):
    # embeddings: numpy array (N, D), meta_list: list of metadata (e.g., entity surface forms)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype('float32'))
    return index

# For the demo we won't compute real dense embeddings; we'll use simple lexical match.
def retrieve_by_entity(entity, topk=5):
    kb = load_kb()
    results = []
    for k, v in kb.items():
        if entity.lower() in k.lower() or k.lower() in entity.lower():
            results.append((k, v))
    return results[:topk]
