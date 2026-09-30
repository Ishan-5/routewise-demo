"""answer.py -- retrieve relevant chunks, then route the answer through routewise /route.

The bare question is sent as `query` so the difficulty classifier scores the
question itself; the retrieval context travels in `messages` so the model still
sees it when answering.
"""
import json
from pathlib import Path

import numpy as np
import requests

import config

DATA_DIR = Path(__file__).parent / "data"
INDEX_PATH = DATA_DIR / "index.npz"
CHUNKS_PATH = DATA_DIR / "chunks.json"
PROMPT_BUDGET = 950


class Index:
    def __init__(self, index_path: Path = INDEX_PATH, chunks_path: Path = CHUNKS_PATH):
        if not index_path.exists():
            raise SystemExit(
                f"Index not found at {index_path}. Run `python ingest.py` first."
            )
        data = np.load(index_path)
        self.embeddings = data["embeddings"]
        self.chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.model = self._load_model()

    @staticmethod
    def _load_model():
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def search(self, query: str, k: int = 3) -> list[dict]:
        q = self.model.encode([query])
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        sims = (self.embeddings @ q.T).ravel()
        order = np.argsort(-sims)[:k]
        hits = []
        for idx in order:
            hit = dict(self.chunks[int(idx)])
            hit["score"] = float(sims[idx])
            hits.append(hit)
        return hits

    def build_context(self, query: str, k: int = 3) -> str:
        hits = self.search(query, k=k)
        context = "\n\n".join(
            f"[{h['title']}] {h['text'][:260]}" for h in hits
        )
        prompt = (
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer based only on the context. If the context does not contain "
            "the answer, say you don't know."
        )
        return prompt[:PROMPT_BUDGET]


def route_query(
    query: str,
    *,
    messages: list[dict] | None = None,
    override_tier: str | None = None,
    bypass_cache: bool = False,
) -> dict:
    payload = {"query": query}
    if messages:
        payload["messages"] = messages
    if override_tier:
        payload["override_tier"] = override_tier
    if bypass_cache:
        payload["bypass_cache"] = True
    resp = requests.post(
        f"{config.base_url()}/route",
        headers={"Authorization": f"Bearer {config.api_key()}"},
        json=payload,
        timeout=60,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"routewise returned {resp.status_code}: {resp.text[:300]}")
    return resp.json()


def ask(
    index: Index,
    question: str,
    *,
    override_tier: str | None = None,
    bypass_cache: bool = False,
) -> dict:
    context = index.build_context(question)
    result = route_query(
        question,
        messages=[{"role": "user", "content": context}],
        override_tier=override_tier,
        bypass_cache=bypass_cache,
    )
    result["question"] = question
    return result
