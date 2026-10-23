"""Index one pinned Python repository into SQLite and Chroma."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from codecompass.embeddings import embedding_identity
from codecompass.indexing.coordinator import IndexingCoordinatorError, RepositoryIndexCoordinator
from codecompass.indexing.repository import RepositoryValidationError, validate_pinned_repository
from codecompass.providers import OPENAI_COMPATIBLE, OLLAMA, ProviderConfig, create_embedding_provider
from codecompass.storage import SQLiteMetadataStore
from codecompass.vector_index import ChromaVectorIndex


def main(argv: Sequence[str] | None = None) -> int:
    """Run complete structural and vector indexing with reproducible output."""
    parser = _parser()
    args = parser.parse_args(argv)
    commit = _validate_repository(args.repository, args.expected_commit, parser)
    try:
        config = ProviderConfig.from_environment(
            provider=args.provider,
            base_url=args.base_url or (args.ollama_url if args.provider == OLLAMA else None),
            embedding_model=args.embedding_model,
            timeout_seconds=args.timeout_seconds,
            embedding_dimensions=args.embedding_dimensions,
        )
        embedding_provider = create_embedding_provider(config)
    except ValueError as error:
        parser.error(str(error))

    store = SQLiteMetadataStore(args.database)
    model = config.embedding_model or "nomic-embed-text-local:latest"
    identity = embedding_identity(
        config.provider,
        config.base_url or "http://localhost:11434",
        model,
        config.embedding_dimensions,
    )
    coordinator = RepositoryIndexCoordinator(
        store,
        embedding_provider,
        identity,
        lambda project_id: ChromaVectorIndex(
            args.chroma,
            args.collection,
            managed=True,
            project_id=project_id,
        ),
        batch_size=args.batch_size,
        inter_batch_delay_seconds=args.embedding_delay_seconds,
        allowed_structural_error_paths=_load_allowed_structural_error_paths(args.allowed_structural_errors_file, parser),
    )
    try:
        result = coordinator.index_repository(args.repository, args.project_name)
    except IndexingCoordinatorError as error:
        failures = [asdict(item) for item in error.failures]
        print(
            json.dumps(
                {
                    "repository": str(args.repository.resolve()),
                    "commit": commit,
                    "complete": False,
                    "structural_stats": asdict(error.structural_stats) if error.structural_stats else {},
                    "structural_errors": [item for item in failures if item["stage"] in {"scan", "parse", "chunk", "storage"}],
                    "errors": failures,
                    "truncated_details": [asdict(item) for item in error.truncated],
                },
                indent=2,
            )
        )
        return 1

    output = {
        "repository": str(args.repository.resolve()),
        "commit": commit,
        "python_files": result.structural_stats.files_discovered,
        "strategy": result.strategy,
        "no_changes": result.no_changes,
        "file_changes": result.file_changes,
        "symbols": result.structural_stats.symbols_extracted,
        "canonical_chunks": result.chunks_expected,
        "embeddings": result.embeddings_generated,
        "vectors": result.vectors_stored,
        "chunks_reused": result.chunks_reused,
        "vectors_reused": result.vectors_reused,
        "vectors_deleted": result.vectors_deleted,
        "truncated_embeddings": len(result.truncated),
        "embedding_retries": result.embedding_retries,
        "embedding_failures": 0,
        "vector_failures": 0,
        "id_set_equal": set(result.expected_ids) == set(result.vector_ids),
        "complete": True,
        "truncated_details": [asdict(item) for item in result.truncated],
        "errors": [],
        "allowed_structural_error_paths": sorted(
            _load_allowed_structural_error_paths(args.allowed_structural_errors_file, parser)
        ),
    }
    print(json.dumps(output, indent=2))
    return 0


def _validate_repository(repository: Path, expected_commit: str, parser: argparse.ArgumentParser) -> str:
    try:
        return validate_pinned_repository(repository, expected_commit)
    except RepositoryValidationError as error:
        parser.error(str(error))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--chroma", required=True, type=Path)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--project-name")
    parser.add_argument("--provider", choices=(OLLAMA, OPENAI_COMPATIBLE), default=OLLAMA)
    parser.add_argument("--base-url", help="Provider base URL; required for openai_compatible")
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-dimensions", type=int)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--embedding-delay-seconds",
        type=float,
        default=1.0,
        help="Delay between embedding batches to avoid provider rate-limit bursts",
    )
    parser.add_argument(
        "--allowed-structural-errors-file",
        type=Path,
        help="Evaluation-only newline-delimited relative paths whose parse/chunk errors are expected fixtures",
    )
    return parser


def _load_allowed_structural_error_paths(path: Path | None, parser: argparse.ArgumentParser) -> frozenset[str]:
    if path is None:
        return frozenset()
    try:
        values = {
            line.strip().replace("\\", "/")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        }
    except OSError as error:
        parser.error(f"cannot read allowed structural errors file: {error}")
    return frozenset(values)


if __name__ == "__main__":
    raise SystemExit(main())
