"""Voyage embeddings wrapper with disk cache and rate-limit retry."""
from __future__ import annotations

import hashlib
import logging
import pickle
import time
from pathlib import Path

import voyageai

from .config import DEFAULT_EMBEDDING_MODEL, EMBEDDINGS_CACHE

logger = logging.getLogger(__name__)


class Embedder:
    """Embeds text via Voyage with per-text disk caching.

    Cache key is sha256(text + model + input_type). Each cached vector
    is a single pickle file under cache_dir. Calls support both single
    strings (returns one vector) and lists (returns a list of vectors),
    matching the shape convention from the user's notebook.
    """

    def __init__(
        self,
        client: voyageai.Client | None = None,
        model: str = DEFAULT_EMBEDDING_MODEL,
        cache_dir: Path = EMBEDDINGS_CACHE,
    ):
        self._client = client or voyageai.Client()
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, text: str, input_type: str) -> str:
        h = hashlib.sha256()
        h.update(text.encode("utf-8"))
        h.update(b"|")
        h.update(self.model.encode("utf-8"))
        h.update(b"|")
        h.update(input_type.encode("utf-8"))
        return h.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.pkl"

    def _read_cache(self, key: str) -> list[float] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with path.open("rb") as f:
                return pickle.load(f)
        except (pickle.UnpicklingError, EOFError, AttributeError) as e:
            logger.warning("Corrupt embedding cache %s; removing. %s", path, e)
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, key: str, vector: list[float]) -> None:
        with self._cache_path(key).open("wb") as f:
            pickle.dump(vector, f)

    def _embed_batch_with_retry(
        self,
        texts: list[str],
        input_type: str,
        max_retries: int = 5,
    ) -> list[list[float]]:
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                result = self._client.embed(
                    texts, model=self.model, input_type=input_type
                )
                return result.embeddings
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                # Voyage doesn't expose a stable rate-limit exception type;
                # match on message content so real errors propagate immediately.
                if "rate" in msg or "limit" in msg or "429" in msg:
                    if attempt == max_retries - 1:
                        break
                    logger.warning(
                        "Voyage rate limit; sleeping %.1fs (attempt %d)",
                        delay,
                        attempt + 1,
                    )
                    time.sleep(delay)
                    delay = min(delay * 2, 32)
                else:
                    raise
        assert last_exc is not None
        raise last_exc

    def embed(
        self,
        texts: str | list[str],
        input_type: str = "document",
    ) -> list[float] | list[list[float]]:
        """Embed one string (returns one vector) or a list (returns a list)."""
        is_list = isinstance(texts, list)
        items: list[str] = list(texts) if is_list else [texts]  # type: ignore[arg-type]

        results: list[list[float] | None] = [None] * len(items)
        to_compute: list[tuple[int, str]] = []

        for i, t in enumerate(items):
            key = self._cache_key(t, input_type)
            hit = self._read_cache(key)
            if hit is not None:
                results[i] = hit
            else:
                to_compute.append((i, t))

        if to_compute:
            indexes = [i for i, _ in to_compute]
            new_texts = [t for _, t in to_compute]
            new_vectors = self._embed_batch_with_retry(new_texts, input_type)
            for idx, vec in zip(indexes, new_vectors):
                results[idx] = vec
                self._write_cache(self._cache_key(items[idx], input_type), vec)

        final = [r for r in results if r is not None]
        if len(final) != len(items):
            raise RuntimeError("Embedding result count mismatch")
        return final if is_list else final[0]

    def __call__(
        self,
        texts: str | list[str],
        input_type: str = "document",
    ) -> list[float] | list[list[float]]:
        return self.embed(texts, input_type=input_type)
