"""Embedding provider for OpenAI-compatible HTTP endpoints."""

from __future__ import annotations

from typing import Any, Sequence

from codecompass._openai_compatible_http import (
    OpenAICompatibleHTTPError,
    post_json,
    validate_base_url,
)
from codecompass.embeddings.base import EmbeddingProviderError, EmbeddingResult


class OpenAICompatibleEmbeddingProvider:
    """Generate ordered single or batch embeddings through `/embeddings`."""

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 30.0,
        expected_dimensions: int | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAI-compatible embedding model is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if expected_dimensions is not None and expected_dimensions < 1:
            raise ValueError("expected_dimensions must be positive or None")
        self.model = model.strip()
        self.base_url = validate_base_url(base_url)
        self.api_key = api_key or None
        self.timeout_seconds = timeout_seconds
        self.expected_dimensions = expected_dimensions

    def embed_text(self, text: str) -> EmbeddingResult:
        """Embed one text input."""
        return self.embed_texts((text,))[0]

    def embed_texts(self, texts: Sequence[str]) -> tuple[EmbeddingResult, ...]:
        """Embed text inputs in their original order."""
        for text in texts:
            if not isinstance(text, str) or not text:
                raise self._error("InvalidInput", "Embedding text must be a non-empty string")
        if not texts:
            return ()
        response = self._post_json({"model": self.model, "input": list(texts)})
        vectors = self._ordered_vectors(response, len(texts))
        model = response.get("model") if isinstance(response.get("model"), str) else self.model
        return tuple(
            EmbeddingResult(vector=vector, model=model, dimensions=len(vector)) for vector in vectors
        )

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return post_json(
                self.base_url,
                "embeddings",
                payload,
                api_key=self.api_key,
                timeout_seconds=self.timeout_seconds,
            )
        except OpenAICompatibleHTTPError as error:
            error_type = (
                "InvalidResponse"
                if error.error_type.startswith("invalid_response_")
                else error.error_type
            )
            raise self._error(error_type, error.message) from None

    def _ordered_vectors(self, response: dict[str, Any], expected_count: int) -> list[list[float]]:
        data = response.get("data")
        if not isinstance(data, list) or len(data) != expected_count:
            raise self._error("InvalidResponse", f"Expected {expected_count} embedding records")
        indexed: dict[int, list[float]] = {}
        missing: list[dict[str, Any]] = []
        for record in data:
            if not isinstance(record, dict):
                raise self._error("InvalidResponse", "Embedding record must be an object")
            if "index" not in record:
                missing.append(record)
                continue
            if not isinstance(record.get("index"), int):
                raise self._error("InvalidResponse", "Embedding record must contain an integer index")
            index = record["index"]
            if index in indexed or index < 0 or index >= expected_count:
                raise self._error("InvalidResponse", "Embedding record indexes are invalid")
            indexed[index] = self._vector(record.get("embedding"))
        remaining = [index for index in range(expected_count) if index not in indexed]
        if len(missing) != len(remaining):
            raise self._error("InvalidResponse", "Embedding record indexes are incomplete")
        # Some compatible endpoints omit indices on only part of a batch; fill the remaining positions in response order.
        for index, record in zip(remaining, missing):
            indexed[index] = self._vector(record.get("embedding"))
        vectors = [indexed[index] for index in range(expected_count)]
        return self._validate_dimensions(vectors)

    def _validate_dimensions(self, vectors: list[list[float]]) -> list[list[float]]:
        dimensions = {len(vector) for vector in vectors}
        if len(dimensions) != 1:
            raise self._error("InvalidResponse", "Embedding dimensions are inconsistent")
        dimensions_value = next(iter(dimensions))
        if self.expected_dimensions is not None and dimensions_value != self.expected_dimensions:
            raise self._error(
                "DimensionMismatch",
                f"Expected embedding dimension {self.expected_dimensions}, got {dimensions_value}",
            )
        return vectors

    def _vector(self, value: Any) -> list[float]:
        if not isinstance(value, list) or not value:
            raise self._error("InvalidResponse", "Embedding vector must be a non-empty list")
        if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
            raise self._error("InvalidResponse", "Embedding vector values must be numeric")
        return [float(item) for item in value]

    def _error(self, error_type: str, message: str) -> EmbeddingProviderError:
        return EmbeddingProviderError("openai_compatible", self.model, error_type, message)
