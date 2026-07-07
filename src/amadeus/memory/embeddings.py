"""Embedding providers.

``OllamaEmbedder`` is the real provider. ``HashingEmbedder`` is a
deterministic, dependency-free fallback used in tests and offline mode —
it produces stable pseudo-embeddings where token overlap correlates with
cosine similarity, which is enough to exercise the retrieval pipeline.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import httpx
import numpy as np


class EmbeddingProvider(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, dim) float32 array of L2-normalized embeddings."""
        ...


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)


class HashingEmbedder:
    """Bag-of-hashed-tokens embedding. Deterministic; offline; test-safe."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                out[i, idx] += sign
        return _normalize(out)


class OllamaEmbedder:
    """Embeddings via a local Ollama server (e.g. nomic-embed-text)."""

    def __init__(self, base_url: str, model: str, dim: int = 768) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.dim = dim
        self._client = httpx.Client(timeout=60.0)

    def embed(self, texts: list[str]) -> np.ndarray:
        response = self._client.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
        )
        response.raise_for_status()
        vectors = np.asarray(response.json()["embeddings"], dtype=np.float32)
        return _normalize(vectors)


def create_embedder(provider: str, *, base_url: str = "", model: str = "", dim: int = 768):
    if provider == "ollama":
        return OllamaEmbedder(base_url=base_url, model=model, dim=dim)
    if provider == "hashing":
        return HashingEmbedder(dim=dim)
    raise ValueError(f"unknown embedding provider: {provider!r}")
