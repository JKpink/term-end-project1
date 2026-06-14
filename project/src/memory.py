"""Experience Pool — RAG-based retrieval memory for AgentNet agents."""
from __future__ import annotations

import hashlib
from typing import Optional

import numpy as np

from .config import get_config


class ExperienceFragment:
    """A single experience fragment stored in memory."""

    def __init__(self, observation: str, context: str, action: str, outcome: float = 0.0):
        self.observation = observation
        self.context = context
        self.action = action
        self.outcome = outcome
        self._hash = hashlib.md5(f"{observation}{context}{action}".encode()).hexdigest()

    @property
    def text(self) -> str:
        return f"问题: {self.observation}\n上下文: {self.context}\n操作: {self.action}\n结果: {self.outcome}"

    def __hash__(self):
        return hash(self._hash)

    def __eq__(self, other):
        return isinstance(other, ExperienceFragment) and self._hash == other._hash


class ExperiencePool:
    """Fixed-size memory pool with embedding-based retrieval."""

    def __init__(self, max_size: int | None = None, retrieval_count: int | None = None):
        cfg = get_config()
        self.max_size = max_size or cfg.pool_size
        self.retrieval_count = retrieval_count or cfg.pool_retrieval_count
        self._fragments: list[ExperienceFragment] = []
        self._embedder: Optional[object] = None
        self._embeddings: Optional[np.ndarray] = None

    def _ensure_embedder(self):
        if self._embedder is not None:
            return
        try:
            from FlagEmbedding import FlagModel
            cfg = get_config()
            self._embedder = FlagModel(
                cfg.embedding_model,
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=True,
            )
        except ImportError:
            self._embedder = None

    def add(self, fragment: ExperienceFragment):
        """Add a fragment, evict worst if full."""
        self._fragments.append(fragment)
        self._embeddings = None
        if len(self._fragments) > self.max_size:
            self._evict_worst()

    def _evict_worst(self):
        if not self._fragments:
            return
        worst_idx = min(range(len(self._fragments)), key=lambda i: self._fragments[i].outcome)
        self._fragments.pop(worst_idx)

    def retrieve(self, observation: str, context: str = "") -> list[ExperienceFragment]:
        """Retrieve the k most relevant fragments."""
        if not self._fragments:
            return []

        k = min(self.retrieval_count, len(self._fragments))
        query = f"{observation} {context}"

        try:
            self._ensure_embedder()
            if self._embedder is not None and hasattr(self._embedder, 'encode'):
                if self._embeddings is None:
                    texts = [f"{f.observation} {f.context}" for f in self._fragments]
                    self._embeddings = self._embedder.encode(texts)
                query_emb = self._embedder.encode([query])[0]
                scores = np.dot(self._embeddings, query_emb) / (
                    np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_emb) + 1e-8
                )
                top_indices = np.argsort(scores)[-k:][::-1]
                return [self._fragments[i] for i in top_indices]
        except Exception:
            pass

        # Fallback: keyword overlap scoring
        query_words = set(query.lower().split())
        scored = []
        for f in self._fragments:
            f_words = set(f"{f.observation} {f.context}".lower().split())
            overlap = len(query_words & f_words) / max(len(query_words | f_words), 1)
            scored.append((overlap, f))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:k]]

    def get_successful(self) -> list[ExperienceFragment]:
        return [f for f in self._fragments if f.outcome > 0]

    def get_failures(self) -> list[ExperienceFragment]:
        return [f for f in self._fragments if f.outcome <= 0]

    def __len__(self) -> int:
        return len(self._fragments)

    def __repr__(self) -> str:
        return f"ExperiencePool({len(self._fragments)}/{self.max_size})"
