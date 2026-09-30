"""ingest.py -- build a local vector index from a Wikipedia corpus.

Downloads plaintext extracts for a topic list, chunks them, embeds the chunks
with the same MiniLM family the routewise difficulty classifier uses, and
saves a normalized embedding matrix + chunk metadata under data/.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import requests

DEFAULT_TOPICS = [
    "Large language model",
    "Transformer (deep learning)",
    "Retrieval-augmented generation",
    "Prompt engineering",
    "Hallucination (artificial intelligence)",
    "Fine-tuning (deep learning)",
    "Attention (machine learning)",
    "Semantic search",
    "Tokenization (language model)",
    "Recurrent neural network",
]

CHUNK_SIZE = 600
OVERLAP = 80
DATA_DIR = Path(__file__).parent / "data"


def fetch_extracts(titles: list[str]) -> dict[str, str]:
    url = "https://en.wikipedia.org/w/api.php"
    texts: dict[str, str] = {}
    for i in range(0, len(titles), 50):
        params = {
            "action": "query",
            "titles": "|".join(titles[i : i + 50]),
            "prop": "extracts",
            "explaintext": 1,
            "exlimit": "max",
            "format": "json",
            "origin": "*",
        }
        resp = requests.get(
            url,
            params=params,
            headers={"User-Agent": "routewise-demo/0.1 (RAG demo; contact: demo@example.com)"},
            timeout=60,
        )
        resp.raise_for_status()
        for pid, page in resp.json()["query"]["pages"].items():
            if int(pid) >= 0 and "extract" in page:
                texts[page["title"]] = page["extract"]
    return texts


def chunk_text(title: str, text: str) -> list[dict]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks: list[dict] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            space = text.rfind(" ", start + CHUNK_SIZE // 2, end)
            if space > start:
                end = space
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"id": f"{title}#{len(chunks)}", "title": title, "text": chunk})
        if end >= len(text) - 1:
            break
        start = max(start + 1, end - OVERLAP)
    return chunks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", nargs="*", help="Override the default Wikipedia topics")
    parser.add_argument("--output", default="index.npz", help="Index filename under data/")
    args = parser.parse_args()

    topics = args.topics or DEFAULT_TOPICS
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching {len(topics)} Wikipedia articles...")
    extracts = fetch_extracts(topics)
    print(f"Got {len(extracts)} articles")

    chunks = [c for t, e in extracts.items() for c in chunk_text(t, e)]
    print(f"Produced {len(chunks)} chunks")

    print("Loading MiniLM embedding model...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding chunks...")
    vectors = model.encode([c["text"] for c in chunks], show_progress_bar=True)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    out_npz = DATA_DIR / args.output
    np.savez(out_npz, embeddings=vectors.astype(np.float32))
    (DATA_DIR / "chunks.json").write_text(
        json.dumps(chunks, indent=2), encoding="utf-8"
    )
    print(f"Saved index to {out_npz}")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
