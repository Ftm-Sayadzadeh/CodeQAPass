"""Build the A1 index of a SWE-QA v4 repository through the official indexing CLI, with a persistent embedding cache.

The official coordinator keeps embeddings in memory and activates an index only when every batch succeeded, so one rate-limit failure
discards all earlier (paid) batches. This driver runs the unchanged `codecompass.indexing.cli.main` and only wraps the embedding provider
with a disk cache, so a rerun sends just the texts that have no vector yet. Nothing in `src/` is changed and the final index is still built and
activated by the official coordinator.

    python scripts/evaluation/swe_qa_v4_index_a1.py --repo conan

Cache key = sha256(namespace + text), namespace = {cache version, provider, endpoint identity, model, dimensions}. Only complete, valid batches
(right count, right dimension, finite values) are stored, written atomically. The cache lives under data/benchmarks/swe_qa_v4/embedding_cache and
must never be committed or mixed with the final index. Pacing between real calls is done here (the CLI delay is set to 0), so cached batches
do not wait.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import swe_qa_v4_runner as v4  # noqa: E402
from codecompass.embeddings import embedding_identity  # noqa: E402
from codecompass.embeddings.base import EmbeddingProviderError, EmbeddingResult  # noqa: E402
from codecompass.indexing import cli  # noqa: E402
from codecompass.providers import ProviderConfig, create_embedding_provider  # noqa: E402

CACHE_VERSION = 1
DEFAULT_PACE_SECONDS = 2.0
real_factory = create_embedding_provider  # replaced in tests


def namespace_hash(provider: str, base_url: str, model: str, dimensions: int | None) -> str:
    identity = embedding_identity(provider, base_url, model, dimensions)
    payload = {"cache_version": CACHE_VERSION, "provider": identity.provider, "endpoint_sha256": identity.endpoint_sha256,
               "model": identity.model, "dimensions": identity.dimensions}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


class CachingEmbeddingProvider:
    """Disk-cached wrapper around an embedding provider; unknown attributes are delegated to the wrapped provider."""

    def __init__(self, inner, cache_dir: Path, namespace: str, dimensions: int | None, pace_seconds: float = 0.0,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic) -> None:
        self._inner = inner
        self._namespace = namespace
        self._dimensions = dimensions
        self._pace = pace_seconds
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None
        self._dir = cache_dir / namespace
        self._dir.mkdir(parents=True, exist_ok=True)
        self._vectors: dict[str, tuple[np.ndarray, str]] = {}
        for path in sorted(self._dir.glob("batch_*.npz")):
            with np.load(path, allow_pickle=False) as data:
                model = str(data["model"])
                for key, vector in zip(data["keys"], data["vectors"]):
                    self._vectors[str(key)] = (vector, model)
        self.stats = {"hits": 0, "misses": 0, "inner_calls": 0, "failed_inner_calls": 0, "cached_vectors_at_start": len(self._vectors)}

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def _key(self, text: str) -> str:
        return hashlib.sha256((self._namespace + "\0" + text).encode("utf-8")).hexdigest()

    def embed_text(self, text: str) -> EmbeddingResult:
        return self.embed_texts((text,))[0]

    def embed_texts(self, texts: Sequence[str]) -> tuple[EmbeddingResult, ...]:
        if not texts:
            return ()
        keys = [self._key(text) for text in texts]
        first_missing: dict[str, int] = {}
        for index, key in enumerate(keys):
            if key not in self._vectors and key not in first_missing:
                first_missing[key] = index
        self.stats["hits"] += sum(1 for key in keys if key in self._vectors)
        if first_missing:
            self.stats["misses"] += len(first_missing)
            self._fetch(first_missing, texts)
        results = []
        for key in keys:
            vector, model = self._vectors[key]
            results.append(EmbeddingResult(vector=vector.tolist(), model=model, dimensions=len(vector)))
        return tuple(results)

    def _fetch(self, first_missing: dict[str, int], texts: Sequence[str]) -> None:
        if self._pace and self._last_call is not None:
            wait = self._pace - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self.stats["inner_calls"] += 1
        miss_keys = list(first_missing)
        try:
            raw = self._inner.embed_texts([texts[first_missing[key]] for key in miss_keys])
        except Exception:
            self.stats["failed_inner_calls"] += 1
            self._last_call = self._clock()
            raise
        self._last_call = self._clock()
        matrix, model = self._validated(raw, len(miss_keys))
        name = hashlib.sha256("".join(miss_keys).encode("ascii")).hexdigest()[:24]
        target = self._dir / f"batch_{name}.npz"
        temporary = self._dir / f"batch_{name}.tmp.npz"
        np.savez(temporary, keys=np.array(miss_keys), vectors=matrix, model=np.array(model))
        os.replace(temporary, target)
        for key, vector in zip(miss_keys, matrix):
            self._vectors[key] = (vector, model)

    def _validated(self, raw: Sequence[EmbeddingResult], expected: int) -> tuple[np.ndarray, str]:
        def invalid(message: str) -> EmbeddingProviderError:
            return EmbeddingProviderError("embedding_cache", None, "InvalidResponse", message)

        if len(raw) != expected:
            raise invalid(f"expected {expected} vectors, received {len(raw)}")
        matrix = np.array([item.vector for item in raw], dtype=np.float64)
        if matrix.ndim != 2 or (self._dimensions is not None and matrix.shape[1] != self._dimensions):
            raise invalid("vector dimension differs from the configured dimension")
        if not np.isfinite(matrix).all():
            raise invalid("non-finite values in the embedding response")
        return matrix, raw[0].model


def build_argv(repo: str, env: dict[str, str], pace: float, artifact_root: Path | None = None,
               batch_size: int = 32) -> list[str]:
    root = v4.repo_root(repo).resolve()
    index = (v4.DATA if artifact_root is None else artifact_root) / "index"
    index.mkdir(parents=True, exist_ok=True)
    argv = [
        "--repository", str(root), "--expected-commit", v4.COMMITS[repo],
        "--database", str(index / f"{repo}.sqlite"), "--chroma", str(index / f"{repo}_chroma"),
        "--collection", f"swe-qa-v4-{repo}", "--project-name", repo,
        "--provider", "openai_compatible", "--base-url", env["CODECOMPASS_GEMINI2_EMBEDDING_BASE_URL"],
        "--embedding-model", env.get("CODECOMPASS_GEMINI2_EMBEDDING_MODEL", "gemini-embedding-2"),
        "--embedding-dimensions", env.get("CODECOMPASS_GEMINI2_EMBEDDING_DIMENSIONS", "3072"),
        "--batch-size", str(batch_size),
        "--embedding-delay-seconds", "0",
    ]
    allowed = Path(__file__).resolve().parents[2] / "reports/evaluation/swe_qa_v4/pylint_allowed_structural_errors.txt"
    if repo == "pylint" and allowed.exists():
        argv.extend(["--allowed-structural-errors-file", str(allowed)])
    return argv


def run(argv: list[str], cache_dir: Path, pace: float, sleep: Callable[[float], None] = time.sleep) -> tuple[int, list[CachingEmbeddingProvider]]:
    """Run the official indexing CLI with the embedding provider wrapped by the cache."""
    wrappers: list[CachingEmbeddingProvider] = []

    def factory(config: ProviderConfig):
        inner = real_factory(config)
        model = config.embedding_model or ""
        namespace = namespace_hash(config.provider, config.base_url or "", model, config.embedding_dimensions)
        wrapper = CachingEmbeddingProvider(inner, cache_dir, namespace, config.embedding_dimensions, pace, sleep)
        wrappers.append(wrapper)
        return wrapper

    original = cli.create_embedding_provider
    cli.create_embedding_provider = factory
    try:
        return cli.main(argv), wrappers
    finally:
        cli.create_embedding_provider = original


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default=v4.PILOT_REPO, choices=sorted(v4.COMMITS))
    parser.add_argument("--artifact-root", type=Path, default=v4.DATA,
                        help="Provider-specific root for the final index and embedding cache")
    parser.add_argument("--cache-dir", type=Path,
                        help="Override the cache path; defaults to ARTIFACT_ROOT/embedding_cache")
    parser.add_argument("--pace-seconds", type=float, default=DEFAULT_PACE_SECONDS)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    v4.require_frozen_settings()
    env = dict(os.environ)
    v4.ensure_embedding_run_identity(args.artifact_root, env)
    os.environ.setdefault("CODECOMPASS_API_KEY", env["CODECOMPASS_GEMINI2_EMBEDDING_API_KEY"])
    cache_dir = args.cache_dir or args.artifact_root / "embedding_cache"
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    code, wrappers = run(build_argv(args.repo, env, args.pace_seconds, args.artifact_root, args.batch_size), cache_dir, args.pace_seconds)
    for wrapper in wrappers:
        print("EMBEDDING_CACHE_STATS " + json.dumps(wrapper.stats))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
