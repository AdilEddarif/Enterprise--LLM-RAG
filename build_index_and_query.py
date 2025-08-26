# build_index_and_query.py
"""
Build a local FAISS index from chunks.jsonl, then run retrieval with citations
and a heuristic "final answer" (extracts 'Résultat attendu' when present).

Usage:
  python build_index_and_query.py --data ./processed/chunks.jsonl --top_k 5 \
    --query "Quelle est l'attente pour le buffering adaptatif ?"

Tip:
  Default model is multilingual and great for FR: intfloat/multilingual-e5-base
"""

import argparse
import json
import re
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ---------- IO ----------
def load_chunks(path: str) -> Tuple[List[str], List[dict]]:
    texts, metas = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            metas.append({k: rec[k] for k in rec if k != "text"})
    return texts, metas


# ---------- Embedding ----------
def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()

def embed_corpus(texts: List[str], model_name: str):
    """
    Embed corpus texts. If using an E5 model, prefix with 'passage: '.
    Returns (embeddings np.ndarray, model)
    """
    model = SentenceTransformer(model_name)
    if _is_e5(model_name):
        enc_texts = ["passage: " + t for t in texts]
    else:
        enc_texts = texts
    vecs = model.encode(
        enc_texts,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vecs, model

def embed_query(query: str, model, model_name: str):
    if _is_e5(model_name):
        q = "query: " + query
    else:
        q = query
    return model.encode([q], normalize_embeddings=True)


# ---------- FAISS ----------
def build_faiss_index(vecs: np.ndarray):
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine similarity via normalized vectors
    index.add(vecs.astype("float32"))
    return index

def search(index, q_vec: np.ndarray, top_k=5):
    D, I = index.search(q_vec.astype("float32"), top_k)
    return D[0], I[0]


# ---------- Heuristic final answer ----------
_EXPECTED_PATTERNS = [
    r"(?:Résultat\s*attendu\s*:\s*)(.+)",         # FR
    r"(?:Expected\s*Result\s*:\s*)(.+)",          # EN
    r"(?:Expected\s*:\s*)(.+)",                   # EN short
]

def extract_expected(text: str) -> str:
    # Try single-line capture after the label
    for pat in _EXPECTED_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Fallback: try to capture block until next label-like line
    for pat in _EXPECTED_PATTERNS:
        m = re.search(pat + r"(.*?)(?:\n[A-ZÀ-Ż][^\n:]{2,}:\s|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            return " ".join(m.group(1).split()).strip()

    return ""


# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to chunks.jsonl")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--query", required=True)
    ap.add_argument("--model", default="intfloat/multilingual-e5-base",
                    help="Sentence-Transformers model name")
    args = ap.parse_args()

    # 1) Load chunks
    texts, metas = load_chunks(args.data)

    # 2) Embed corpus & build index
    vecs, model = embed_corpus(texts, args.model)
    index = build_faiss_index(vecs)

    # 3) Embed query & search
    q_vec = embed_query(args.query, model, args.model)
    scores, idxs = search(index, q_vec, args.top_k)

    # 4) Display results
    print("\n=== QUERY ===")
    print(args.query)

    print("\n=== TOP CONTEXTS ===")
    for rank, (i, score) in enumerate(zip(idxs, scores), start=1):
        src = metas[i].get("doc_path")
        scenario = metas[i].get("scenario_id")
        chunk_id = metas[i].get("chunk_id")
        preview = texts[i][:600] + ("..." if len(texts[i]) > 600 else "")
        print(f"\n[{rank}] score={score:.3f} | src={src} | scenario={scenario} | chunk={chunk_id}")
        print(preview)

    # 5) Citations
    print("\n=== CITATIONS ===")
    for k, i in enumerate(idxs, 1):
        src = metas[i].get("doc_path")
        scenario = metas[i].get("scenario_id")
        chunk_id = metas[i].get("chunk_id")
        print(f"[{k}] {src} (scenario {scenario}, chunk {chunk_id})")

    # 6) Heuristic "final answer"
    best_text = texts[idxs[0]] if len(idxs) > 0 else ""
    attendu = extract_expected(best_text)

    print("\n=== FINAL ANSWER (heuristic) ===")
    if attendu:
        print(attendu)
    else:
        # If we can't extract a clean 'expected' section, show a concise summary from top contexts
        snippet = " ".join(texts[i] for i in idxs[:2])
        print(snippet[:800] + ("..." if len(snippet) > 800 else ""))


if __name__ == "__main__":
    main()
