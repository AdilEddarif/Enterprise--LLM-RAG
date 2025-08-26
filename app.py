# app.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from query_usecases import (
    load_chunks, load_structured, cache_paths_for, embed_corpus_cached,
    build_or_load_index, embed_query, search, rerank_by_doc_and_scenario,
    deterministic_count, filename_only, call_ollama, build_prompt_single_ref
)

class QueryRequest(BaseModel):
    query: str

app = FastAPI()

# Load once at startup
DATA_PATH = "processed/chunks.jsonl"
STRUCTURED_PATH = "processed/structured.jsonl"
MODEL_NAME = "intfloat/multilingual-e5-small"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"

texts, metas = load_chunks(DATA_PATH)
rows = load_structured(STRUCTURED_PATH)
emb_path, index_path = cache_paths_for(DATA_PATH)
vecs, st_model = embed_corpus_cached(texts, MODEL_NAME, emb_path)
index = build_or_load_index(vecs, index_path)

@app.post("/ask")
def ask(req: QueryRequest):
    query = req.query.strip()

    # 1) Deterministic check
    det = deterministic_count(rows, query)
    if det:
        return {
            "answer": det["final_answer"],
            "source": det.get("source_filename", "")
        }

    # 2) Semantic search
    q_vec = embed_query(query, st_model, MODEL_NAME)
    scores, idxs = search(index, q_vec, top_k=8)
    if not idxs:
        return {"answer": "Information non trouvée dans le contexte.", "source": ""}

    reranked = rerank_by_doc_and_scenario(idxs, scores, metas, query)
    top_i = reranked[0]

    top_text = texts[top_i]
    top_src = metas[top_i].get("doc_path", "")
    top_filename = filename_only(top_src) if top_src else "source.pdf"

    prompt = build_prompt_single_ref(query, top_text, top_filename)
    answer = call_ollama(prompt, url=OLLAMA_URL, model=OLLAMA_MODEL)

    if "SOURCE:" not in answer:
        answer += f"\n\nSOURCE: {top_filename}"

    return {"answer": answer, "source": top_filename}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
