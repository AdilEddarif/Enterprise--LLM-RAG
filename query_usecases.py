# query_usecases.py
"""
Best-practice RAG CLI for your PDF use-cases:
- Deterministic tools first (counting from structured.jsonl)
- Cached embeddings + FAISS (fast subsequent queries)
- Re-ranking by filename keywords + scenario number
- Top-1 generation with a single reference (filename.pdf) via Ollama phi3:mini

Usage (Windows):
  python query_usecases.py ^
    --data "C:/.../processed/chunks.jsonl" ^
    --structured "C:/.../processed/structured.jsonl" ^
    --query "Quelles sont les préconditions du scénario 1 pour le cas de test Adaptation qualité video ?" ^
    --top_k 8

Optional flags:
  --model intfloat/multilingual-e5-small     (fast)
  --ollama_url http://localhost:11434/api/generate
  --ollama_model phi3:mini
  --max_tokens 384
"""

import argparse
import json
import os
import re
import unicodedata
from typing import List, Tuple, Dict, Any, Optional

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# ========== IO ==========
def load_chunks(path: str) -> Tuple[List[str], List[Dict[str, Any]]]:
    texts, metas = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            metas.append({k: rec[k] for k in rec if k != "text"})
    if not texts:
        raise RuntimeError("No chunks found in file. Did you run preprocessing?")
    return texts, metas


def load_structured(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def filename_only(path: str) -> str:
    return path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


# ========== Text utils ==========
def _is_e5(model_name: str) -> bool:
    return "e5" in model_name.lower()


def _strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s or "") if unicodedata.category(c) != "Mn"
    )


def _norm(s: Optional[str]) -> str:
    return _strip_accents(s or "").lower()


def is_count_query(q: str) -> bool:
    qn = _norm(q)
    for t in ["combien", "nombre", "how many", "how much", "count"]:
        if t in qn:
            return True
    return False


def guess_doc_keywords(q: str) -> List[str]:
    """Meaningful filename keywords (keep domain words: adaptation, qualite, video, etc.)."""
    qn = _norm(q)
    tokens = re.findall(r"[a-z0-9]+", qn)
    stop = {
        "de", "du", "des", "la", "le", "les", "un", "une", "et", "pour", "dans",
        "au", "aux", "en", "sur", "a", "il", "y", "est", "que", "d",
        "cas", "test", "scenario", "scenarios", "pdf", "utilisation",
        "combien", "nombre", "how", "many", "much", "count"
    }
    keep = [t for t in tokens if len(t) >= 3 and t not in stop]
    return keep[:8]


def extract_scenario_number(q: str) -> Optional[str]:
    """
    Returns scenario id mentioned in the query, e.g., '1' for 'scénario 1'.
    Handles scenario/scénario/sc/scenario n°/#/: cases, accents stripped.
    """
    qn = _norm(q)
    m = re.search(r"(?:sc|scena?rio)[\s:#n°]*\s*(\d+)", qn)
    return m.group(1) if m else None


# ========== Deterministic tool: counting from structured.jsonl ==========
def deterministic_count(structured_rows: List[Dict[str, Any]], query: str) -> Optional[Dict[str, str]]:
    """
    If 'count' intent: compute exact count from structured.jsonl.
    Returns {"final_answer": str, "source_filename": str} or None if not a count query.
    """
    if not is_count_query(query):
        return None

    if not structured_rows:
        return {"final_answer": "Aucun scénario trouvé (dataset vide).", "source_filename": ""}

    kws = guess_doc_keywords(query)

    def fn_norm(path: str) -> str:
        return _norm(filename_only(path))

    # Score filenames by keyword overlap; pick the best match if keywords exist
    filtered = structured_rows
    if kws:
        scores: Dict[str, int] = {}
        for r in structured_rows:
            fn = fn_norm(r.get("doc_path", ""))
            hit = sum(1 for kw in kws if kw in fn)
            if hit:
                scores[fn] = max(scores.get(fn, 0), hit)
        if scores:
            best_fn = max(scores.items(), key=lambda kv: kv[1])[0]
            filtered = [r for r in structured_rows if fn_norm(r.get("doc_path", "")) == best_fn]
        else:
            # fallback: soft OR over full path
            filtered = [r for r in structured_rows if any(kw in _norm(r.get("doc_path", "")) for kw in kws)]

    # Count unique scenario IDs per doc
    by_doc: Dict[str, set] = {}
    for r in filtered:
        doc = r.get("doc_path", "")
        sid = r.get("scenario_id")
        by_doc.setdefault(doc, set()).add(sid)

    if not by_doc:
        return {"final_answer": "Aucun scénario correspondant trouvé pour cette requête.", "source_filename": ""}

    if len(by_doc) == 1:
        (doc_path, sids) = next(iter(by_doc.items()))
        count = len([s for s in sids if s is not None]) or len(sids)
        return {"final_answer": f"Il y a {count} scénarios.", "source_filename": filename_only(doc_path)}

    # Multiple docs matched: summarize and pick the largest doc as reference
    top_doc, top_ids = max(by_doc.items(), key=lambda kv: len(kv[1]))
    total = sum(len([s for s in ids if s is not None]) or len(ids) for ids in by_doc.values())
    return {
        "final_answer": f"J’ai trouvé {total} scénarios au total sur {len(by_doc)} document(s) correspondants.",
        "source_filename": filename_only(top_doc)
    }


# ========== Embeddings + FAISS with disk cache ==========
def cache_paths_for(data_path: str) -> Tuple[str, str]:
    base_dir = os.path.dirname(os.path.abspath(data_path)) or "."
    return os.path.join(base_dir, "embeddings.npy"), os.path.join(base_dir, "index.faiss")


def embed_corpus_cached(texts: List[str], model_name: str, emb_path: str):
    if os.path.exists(emb_path):
        vecs = np.load(emb_path)
        model = SentenceTransformer(model_name)  # loaded for query embeddings
        return vecs, model

    model = SentenceTransformer(model_name)
    enc_texts = ["passage: " + t for t in texts] if _is_e5(model_name) else texts
    vecs = model.encode(enc_texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    np.save(emb_path, vecs)
    return vecs, model


def build_or_load_index(vecs: np.ndarray, index_path: str):
    if os.path.exists(index_path):
        return faiss.read_index(index_path)
    d = vecs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vecs.astype("float32"))
    faiss.write_index(index, index_path)
    return index


def embed_query(q: str, model, model_name: str):
    q_enc = "query: " + q if _is_e5(model_name) else q
    return model.encode([q_enc], normalize_embeddings=True)


def search(index, q_vec: np.ndarray, top_k: int):
    D, I = index.search(q_vec.astype("float32"), top_k)
    return D[0].tolist(), I[0].tolist()


# ========== Lightweight re-ranking (filename + scenario match) ==========
def _norm_filename(path: str) -> str:
    return _norm(filename_only(path))


def rerank_by_doc_and_scenario(
    idxs: List[int],
    scores: List[float],
    metas: List[Dict[str, Any]],
    query: str
) -> List[int]:
    kws = guess_doc_keywords(query)
    scen = extract_scenario_number(query)

    boosted = []
    for i, base in zip(idxs, scores):
        meta = metas[i]
        fn_norm = _norm_filename(meta.get("doc_path", ""))
        bonus = 0.0

        # filename keyword overlap
        if kws:
            hit = sum(1 for kw in kws if kw in fn_norm)
            if hit > 0:
                bonus += 0.05 * hit  # small additive boost

        # scenario id exact match
        sid = str(meta.get("scenario_id")) if meta.get("scenario_id") is not None else None
        if scen and sid and sid == scen:
            bonus += 0.15  # stronger boost

        boosted.append((i, base + bonus))

    boosted.sort(key=lambda t: t[1], reverse=True)
    return [i for (i, _) in boosted]


# ========== LLM (Ollama) ==========
def call_ollama(prompt: str, url: str, model: str, temperature: float = 0.2, max_tokens: int = 384) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "stream": False,
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip()
    except requests.exceptions.ConnectionError:
        return "[Erreur] Impossible de contacter Ollama. Vérifie que Ollama est installé et en cours d’exécution."
    except Exception as e:
        return f"[Erreur] Appel Ollama a échoué: {e}"


# ========== Prompt (single reference) ==========
SYS_PROMPT = (
    "Tu es un assistant QA. Réponds UNIQUEMENT à partir du contexte fourni. "
    "Si l'information n'est pas présente, réponds: \"Information non trouvée dans le contexte.\" "
    "Réponse courte et claire. Ne fournis qu’une seule référence à la fin."
)


def build_prompt_single_ref(query: str, context: str, ref_tag: str) -> str:
    return (
        f"{SYS_PROMPT}\n\n"
        f"=== CONTEXTE [1: {ref_tag}] ===\n{context}\n\n"
        f"=== QUESTION ===\n{query}\n\n"
        f"=== FORMAT ===\n"
        f"- 2–4 phrases maximum.\n"
        f"- Termine par: SOURCE: {ref_tag}\n\n"
        f"Réponse:"
    )


# ========== Main ==========
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to chunks.jsonl")
    ap.add_argument("--structured", required=True, help="Path to structured.jsonl")
    ap.add_argument("--query", required=True)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--model", default="intfloat/multilingual-e5-small")  # faster
    ap.add_argument("--ollama_url", default="http://localhost:11434/api/generate")
    ap.add_argument("--ollama_model", default="phi3:mini")
    ap.add_argument("--max_tokens", type=int, default=384)
    args = ap.parse_args()

    # 1) Deterministic tool first: counting queries
    rows = load_structured(args.structured)
    det = deterministic_count(rows, args.query)
    if det:
        ans = det.get("final_answer", "").strip()
        src = det.get("source_filename", "").strip()
        print(f"FINAL ANSWER: {ans}")
        print(f"SOURCE: {src}")
        return

    # 2) Cached embeddings + FAISS
    texts, metas = load_chunks(args.data)
    emb_path, index_path = cache_paths_for(args.data)
    vecs, st_model = embed_corpus_cached(texts, args.model, emb_path)
    index = build_or_load_index(vecs, index_path)

    # 3) Retrieve + rerank
    q_vec = embed_query(args.query, st_model, args.model)
    scores, idxs = search(index, q_vec, args.top_k)
    if not idxs:
        print("FINAL ANSWER: Information non trouvée dans le contexte.")
        print("SOURCE: ")
        return
    reranked = rerank_by_doc_and_scenario(idxs, scores, metas, args.query)
    top_i = reranked[0]

    # 4) Single-context generation
    top_text = texts[top_i]
    top_src = metas[top_i].get("doc_path", "")
    top_filename = filename_only(top_src) if top_src else "source.pdf"
    prompt = build_prompt_single_ref(args.query, top_text, top_filename)
    answer = call_ollama(prompt, url=args.ollama_url, model=args.ollama_model, max_tokens=args.max_tokens)

    # Safety: append SOURCE if model forgot
    if "SOURCE:" not in answer:
        answer = f"{answer.strip()}\n\nSOURCE: {top_filename}"

    # 5) Minimal console output
    # If you want strictly two lines, split at SOURCE:
    parts = answer.split("\n")
    # Find the line starting with SOURCE:
    src_line = next((p for p in parts if p.strip().upper().startswith("SOURCE:")), None)
    final_text = "\n".join(p for p in parts if p and p != src_line).strip()
    final_src = src_line.replace("SOURCE:", "").strip() if src_line else top_filename

    print(f"FINAL ANSWER: {final_text}")
    print(f"SOURCE: {final_src}")


if __name__ == "__main__":
    main()
