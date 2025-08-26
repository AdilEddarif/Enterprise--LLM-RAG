"""
Preprocess a folder of French/English PDF test-case specs into clean chunks with metadata.
- Extracts text via PyMuPDF
- Splits by Scenario blocks (e.g., "Scenario (4) : Buffering adaptatif")
- Parses Category, Priority, Preconditions, Steps, Expected results when present
- Emits JSONL with fields: doc_path, scenario_id, title, category, priorite, preconditions, steps, expected, raw_text
- Also emits chunked JSONL ready for embeddings

Usage:
    python preprocess_usecases.py --pdf_dir "/path/to/your/pdfs" --out_dir "./processed"
"""

import os
import re
import json
import argparse
from pathlib import Path
import fitz  # PyMuPDF

SCENARIO_HDR = re.compile(
    r'^\s*Scenario\s*\((\d+)\)\s*:\s*(.+?)\s*$',
    re.IGNORECASE | re.MULTILINE
)


def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    texts = []
    for page in doc:
        texts.append(page.get_text("text"))
    return "\n".join(texts)


def normalize(s: str) -> str:
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n{2,}', '\n\n', s)
    return s.strip()


def parse_blocks(full_text: str):
    """
    Split the document into scenario blocks using scenario headers.
    Return a list of dicts with parsed fields where possible.
    """
    matches = list(SCENARIO_HDR.finditer(full_text))
    blocks = []
    if not matches:
        blocks.append({
            "scenario_id": None,
            "title": "Document",
            "body": normalize(full_text)
        })
        return blocks

    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = normalize(full_text[start:end])
        blocks.append({
            "scenario_id": m.group(1),
            "title": m.group(2),
            "body": body
        })
    return blocks


def parse_fields(body: str):
    # French labels first, with English fallbacks
    def grab(label_patterns):
        for pat in label_patterns:
            rx = re.compile(pat, re.IGNORECASE | re.DOTALL)
            m = rx.search(body)
            if m:
                return normalize(m.group(1))
        return None

    category = grab([
        r'Catégorie\s*:\s*(.+?)(?:\n[A-Z].+?:|\Z)',
        r'Category\s*:\s*(.+?)(?:\n[A-Z].+?:|\Z)'
    ])
    priorite = grab([
        r'Priorité\s*:\s*(.+?)(?:\n[A-Z].+?:|\Z)',
        r'Priority\s*:\s*(.+?)(?:\n[A-Z].+?:|\Z)'
    ])
    preconditions = grab([
        r'Précondition\s*:\s*(.+?)(?:\n[A-Z].+?:|\Z)',
        r'Precondition\s*:\s*(.+?)(?:\n[A-Z].+?:|\Z)'
    ])
    etapes = grab([
        r'Etapes\s*:\s*(.+?)(?:\n[A-Z].+?:|\Z)',
        r'Steps\s*:\s*(.+?)(?:\n[A-Z].+?:|\Z)'
    ])
    expected = grab([
        r'Résultat attendu\s*:\s*(.+?)(?:\n[A-Z].+?:|\Z)',
        r'Expected\s*Result\s*:\s*(.+?)(?:\n[A-Z].+?:|\Z)',
        r'Expected\s*:\s*(.+?)(?:\n[A-Z].+?:|\Z)'
    ])
    return category, priorite, preconditions, etapes, expected


def chunk_text(text: str, chunk_size=800, overlap=120):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - overlap)
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf_dir", required=True, help="Folder containing your PDF dataset")
    ap.add_argument("--out_dir", default="./processed", help="Output folder")
    ap.add_argument("--chunk_size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=120)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    structured_path = os.path.join(args.out_dir, "structured.jsonl")
    chunks_path = os.path.join(args.out_dir, "chunks.jsonl")

    with open(structured_path, "w", encoding="utf-8") as f_struct, \
         open(chunks_path, "w", encoding="utf-8") as f_chunks:

        for root, _, files in os.walk(args.pdf_dir):
            for name in files:
                if not name.lower().endswith(".pdf"):
                    continue
                pdf_path = os.path.join(root, name)
                try:
                    full_text = extract_text(pdf_path)
                    blocks = parse_blocks(full_text)
                    for b in blocks:
                        category, priorite, prec, steps, expected = parse_fields(b["body"])
                        record = {
                            "doc_path": pdf_path,
                            "scenario_id": b["scenario_id"],
                            "title": b["title"],
                            "category": category,
                            "priorite": priorite,
                            "preconditions": prec,
                            "steps": steps,
                            "expected": expected,
                            "raw_text": b["body"]
                        }
                        f_struct.write(json.dumps(record, ensure_ascii=False) + "\n")

                        # Build a chunk source text
                        parts = []
                        if record["title"]:
                            parts.append(f"Scenario: {record['title']}")
                        if category:
                            parts.append(f"Catégorie: {category}")
                        if priorite:
                            parts.append(f"Priorité: {priorite}")
                        if prec:
                            parts.append(f"Préconditions: {prec}")
                        if steps:
                            parts.append(f"Étapes: {steps}")
                        if expected:
                            parts.append(f"Résultat attendu: {expected}")
                        if not parts:
                            parts.append(record["raw_text"])
                        source_text = "\n".join(parts)

                        for i, chunk in enumerate(
                            chunk_text(source_text, args.chunk_size, args.overlap)
                        ):
                            f_chunks.write(json.dumps({
                                "doc_path": pdf_path,
                                "scenario_id": b["scenario_id"],
                                "chunk_id": i,
                                "text": chunk
                            }, ensure_ascii=False) + "\n")
                except Exception as e:
                    print("Failed on", pdf_path, "->", e)

    print("Wrote:", structured_path)
    print("Wrote:", chunks_path)


if __name__ == "__main__":
    main()
