# backend.py
import os
import re
import json
import numpy as np
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
from typing import List, Dict

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
EMBED_DIM = 384  # embedding dim for all-MiniLM-L6-v2

# initialize model once
embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def extract_text_from_pdf(path: str) -> List[Dict]:
    """
    Extract text from each page of the PDF and return list of dicts:
    [{"doc_id": "file.pdf", "page": 1, "text": "....."}, ...]
    """
    doc = fitz.open(path)
    doc_id = os.path.basename(path)
    chunks = []
    for pno in range(doc.page_count):
        page = doc.load_page(pno)
        text = page.get_text("text")
        if text and text.strip():
            chunks.append({"doc_id": doc_id, "page": pno + 1, "text": text})
    return chunks


def clean_text(txt: str) -> str:
    # basic cleanup
    txt = txt.replace("\x0c", " ").strip()
    txt = re.sub(r"\s+", " ", txt)
    return txt


def chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """
    Break a text into chunks of up to max_chars with overlap characters.
    Keeps sentence boundaries where feasible.
    """
    text = clean_text(text)
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[\.\?\!])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            # start new chunk with overlap from previous chunk
            # keep a suffix of previous sentences approx overlap chars
            current = s
    if current:
        chunks.append(current)

    # apply overlap by merging neighbor chunks slightly if needed
    # ensure chunk lengths less than max_chars + overlap
    return chunks


def prepare_documents(pdf_paths: List[str]) -> List[Dict]:
    """
    Extract and chunk all pages from provided PDFs.
    Returns list of chunk dicts:
      {"id": unique-id, "doc_id": file.pdf, "page": page_num, "chunk": text}
    """
    all_chunks = []
    uid = 0
    for path in pdf_paths:
        page_entries = extract_text_from_pdf(path)
        for p in page_entries:
            text = p["text"]
            chunk_texts = chunk_text(text, max_chars=1000, overlap=200)
            for c in chunk_texts:
                all_chunks.append({
                    "id": f"chunk-{uid}",
                    "doc_id": p["doc_id"],
                    "page": p["page"],
                    "chunk": c
                })
                uid += 1
    return all_chunks


def create_faiss_index(chunks: List[Dict]):
    """
    Create FAISS index and store metadata arrays for retrieval.
    Returns (index, metadata_list)
    metadata_list is list of dicts corresponding to index positions.
    """
    texts = [c["chunk"] for c in chunks]
    # encode to numpy float32
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # metadata
    metadata = [{"id": c["id"], "doc_id": c["doc_id"], "page": c["page"], "text": c["chunk"]} for c in chunks]
    return index, metadata


def save_faiss(index, metadata, dirpath="data/faiss_index"):
    os.makedirs(dirpath, exist_ok=True)
    faiss.write_index(index, os.path.join(dirpath, "index.faiss"))
    with open(os.path.join(dirpath, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)


def load_faiss(dirpath="data/faiss_index"):
    idx_path = os.path.join(dirpath, "index.faiss")
    meta_path = os.path.join(dirpath, "metadata.json")
    if not os.path.exists(idx_path) or not os.path.exists(meta_path):
        return None, None
    index = faiss.read_index(idx_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata


def search(index, metadata, query: str, top_k: int = 4):
    """
    Search FAISS with the query, return top_k metadata dicts with distances.
    """
    q_emb = embed_model.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        entry = metadata[idx].copy()
        entry["score"] = float(dist)
        results.append(entry)
    return results
