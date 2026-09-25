"""
embed.py — Multilingual sentence-embedding wrapper for business names/addresses.

Model: paraphrase-multilingual-MiniLM-L12-v2 (118M params, Apache-2.0)
Chosen for: cross-script semantic similarity (Latin / Devanagari / Bengali / etc.),
small enough to satisfy the challenge's <=8B parameter constraint, and no
external API calls (runs fully local — satisfies "no external APIs/data").

Design goals (per ML-2 job spec):
  - Runs standalone, no local-path dependencies -> works in Colab/Kaggle as-is.
  - Caches embeddings to disk so repeated experiments don't recompute vectors.
  - Deterministic cache key from normalized text, so cache is reusable across runs.

Usage:
    from embed import Embedder

    embedder = Embedder(cache_dir="data/embeddings_cache")
    vecs = embedder.encode(["ABC Traders", "एबीसी ट्रेडर्स"])
    embedder.save_cache()  # optional: also autosaves periodically
"""

from __future__ import annotations
import hashlib
import json
import os
import pickle
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # fixed output dim for this model — used for sanity checks


def _normalize_for_key(text: str) -> str:
    """Normalize text before hashing so trivial whitespace/case diffs share a cache entry.

    NOTE: this is a cache-key normalization only. It intentionally does NOT
    replace src/preprocessing/normalize.py — that pipeline should still run
    upstream to actually clean the text before it ever reaches the embedder.
    """
    return " ".join(text.strip().lower().split())


def _hash_text(text: str) -> str:
    return hashlib.sha1(_normalize_for_key(text).encode("utf-8")).hexdigest()


class Embedder:
    """Wraps a SentenceTransformer model with a persistent on-disk cache.

    The cache is a dict[str hash -> np.ndarray] pickled to
    `<cache_dir>/embedding_cache.pkl`, plus a small json manifest recording
    the model name / dim so a stale cache from a different model is never
    silently reused.
    """

    def __init__(
        self,
        cache_dir: str = "data/embeddings_cache",
        model_name: str = MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 64,
        autosave_every: int = 500,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = self.cache_dir / "embedding_cache.pkl"
        self.manifest_path = self.cache_dir / "embedding_cache_manifest.json"
        self.batch_size = batch_size
        self.autosave_every = autosave_every

        self._model = None  # lazy-loaded — see _load_model()
        self._device = device
        self._lock = threading.Lock()

        self._cache: dict[str, np.ndarray] = {}
        self._dirty_count = 0
        self._load_cache()

    # ------------------------------------------------------------------ #
    # Model loading (lazy — so importing this module never requires a GPU
    # or even the sentence_transformers package to be installed, e.g. if
    # you're only reading cached embeddings).
    # ------------------------------------------------------------------ #
    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence_transformers is required to compute new embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e

        device = self._device
        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        self._model = SentenceTransformer(self.model_name, device=device)
        self._device = device

    # ------------------------------------------------------------------ #
    # Cache I/O
    # ------------------------------------------------------------------ #
    def _load_cache(self) -> None:
        if self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self._cache = pickle.load(f)
        else:
            self._cache = {}

        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            if manifest.get("model_name") != self.model_name:
                raise ValueError(
                    f"Cache at {self.cache_dir} was built with model "
                    f"'{manifest.get('model_name')}', not '{self.model_name}'. "
                    "Use a different cache_dir per model to avoid mixing vectors."
                )

    def save_cache(self) -> None:
        with self._lock:
            with open(self.cache_path, "wb") as f:
                pickle.dump(self._cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.manifest_path.write_text(
                json.dumps(
                    {"model_name": self.model_name, "embedding_dim": EMBEDDING_DIM, "count": len(self._cache)},
                    indent=2,
                )
            )
            self._dirty_count = 0

    def _maybe_autosave(self) -> None:
        if self._dirty_count >= self.autosave_every:
            self.save_cache()

    def cache_stats(self) -> dict:
        return {"cached_vectors": len(self._cache), "cache_path": str(self.cache_path)}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def encode(
        self,
        texts: Sequence[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """Return an (N, EMBEDDING_DIM) float32 array, one row per input string.

        Cache-hit texts never touch the model. Cache-miss texts are batched
        and encoded together, then written back into the cache.
        """
        if len(texts) == 0:
            return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        keys = [_hash_text(t) for t in texts]
        result: List[Optional[np.ndarray]] = [self._cache.get(k) for k in keys]

        miss_idx = [i for i, v in enumerate(result) if v is None]
        if miss_idx:
            self._load_model()
            miss_texts = [texts[i] for i in miss_idx]

            new_vecs = self._model.encode(
                miss_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True,  # so cosine similarity == dot product downstream
            )

            with self._lock:
                for i, vec in zip(miss_idx, new_vecs):
                    vec = vec.astype(np.float32)
                    result[i] = vec
                    self._cache[keys[i]] = vec
                    self._dirty_count += 1

            self._maybe_autosave()

        return np.stack(result).astype(np.float32)

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]

    def __del__(self):
        # Best-effort final flush so a crash mid-experiment doesn't lose
        # newly-computed vectors. Swallow errors — __del__ must not raise.
        try:
            if getattr(self, "_dirty_count", 0) > 0:
                self.save_cache()
        except Exception:
            pass


if __name__ == "__main__":
    # Smoke test — runs even without a real dataset. Verifies:
    #   1. Cross-script variants of the same business embed similarly.
    #   2. Cache actually avoids recomputation on a second call.
    import time

    embedder = Embedder(cache_dir="data/embeddings_cache_smoketest")

    same_business = [
        "ABC Traders",
        "एबीसी ट्रेडर्स",       # Devanagari transliteration of "ABC Traders"
        "এবিসি ট্রেডার্স",       # Bengali transliteration of "ABC Traders"
    ]
    different_business = "XYZ Pharmaceuticals Pvt Ltd"

    t0 = time.time()
    vecs = embedder.encode(same_business + [different_business])
    t1 = time.time()
    print(f"First encode (cache miss): {t1 - t0:.3f}s for {len(same_business) + 1} texts")

    # cosine sim since vectors are normalized -> dot product
    sims_to_first = vecs @ vecs[0]
    print("\nCosine similarity to 'ABC Traders':")
    for text, sim in zip(same_business + [different_business], sims_to_first):
        print(f"  {sim:.4f}  {text}")

    t2 = time.time()
    _ = embedder.encode(same_business + [different_business])
    t3 = time.time()
    print(f"\nSecond encode (should be cache hit, near-instant): {t3 - t2:.4f}s")
    print(embedder.cache_stats())