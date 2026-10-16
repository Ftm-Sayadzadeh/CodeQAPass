from __future__ import annotations

import io
import json
import urllib.error

import pytest

from codecompass.embeddings import (
    EmbeddingProviderError,
    EmbeddingResult,
    OllamaEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from codecompass.llm import (
    LLMProviderError,
    LLMRequest,
    LLMResponse,
    OllamaLLMProvider,
    OpenAICompatibleLLMProvider,
)
from codecompass.providers import ProviderConfig, create_embedding_provider, create_llm_provider


class FakeHTTPResponse:
    def __init__(self, payload) -> None:
        self.data = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    def read(self) -> bytes:
        return self.data


def install_response(monkeypatch, payload):
    captured = []

    def respond(request, timeout):
        captured.append((request, timeout))
        return FakeHTTPResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", respond)
    return captured


def test_openai_compatible_embedding_single_and_batch_preserve_index_order(monkeypatch) -> None:
    captured = install_response(
        monkeypatch,
        {
            "model": "embed-served",
            "data": [
                {"index": 1, "embedding": [3, 4]},
                {"index": 0, "embedding": [1, 2]},
            ],
        },
    )
    provider = OpenAICompatibleEmbeddingProvider(
        "embed-requested", "https://compatible.example/v1", api_key="test-secret", expected_dimensions=2
    )

    results = provider.embed_texts(("first", "second"))

    assert results == (
        EmbeddingResult([1.0, 2.0], "embed-served", 2),
        EmbeddingResult([3.0, 4.0], "embed-served", 2),
    )
    request, timeout = captured[0]
    assert request.full_url == "https://compatible.example/v1/embeddings"
    assert request.get_header("Authorization") == "Bearer test-secret"
    assert json.loads(request.data) == {"model": "embed-requested", "input": ["first", "second"]}
    assert timeout == 30.0


def test_openai_compatible_embedding_single_uses_provider_neutral_result(monkeypatch) -> None:
    install_response(monkeypatch, {"data": [{"index": 0, "embedding": [0.5, 1]}]})

    result = OpenAICompatibleEmbeddingProvider("embed", "http://localhost:8000/v1").embed_text("hello")

    assert result == EmbeddingResult([0.5, 1.0], "embed", 2)


def test_openai_compatible_embedding_accepts_ordered_records_without_indexes(monkeypatch) -> None:
    install_response(monkeypatch, {"model": "embed-served", "data": [
        {"embedding": [1, 2]}, {"embedding": [3, 4]},
    ]})

    results = OpenAICompatibleEmbeddingProvider("embed", "http://localhost:8000/v1", expected_dimensions=2).embed_texts(
        ("first", "second")
    )

    assert results == (
        EmbeddingResult([1.0, 2.0], "embed-served", 2),
        EmbeddingResult([3.0, 4.0], "embed-served", 2),
    )


def test_openai_compatible_embedding_fills_omitted_indexes(monkeypatch) -> None:
    install_response(monkeypatch, {"data": [
        {"embedding": [1, 2]}, {"index": 1, "embedding": [3, 4]},
    ]})

    results = OpenAICompatibleEmbeddingProvider("embed", "http://localhost:8000/v1", expected_dimensions=2).embed_texts(
        ("first", "second")
    )

    assert results == (EmbeddingResult([1.0, 2.0], "embed", 2), EmbeddingResult([3.0, 4.0], "embed", 2))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"data": "invalid"},
        {"data": []},
        {"data": [{"index": 0, "embedding": []}]},
        {"data": [{"index": 0, "embedding": [True]}]},
        {"data": [{"index": 0, "embedding": [1]}, {"index": 0, "embedding": [2]}]},
    ],
)
def test_openai_compatible_embedding_rejects_malformed_response(monkeypatch, payload) -> None:
    install_response(monkeypatch, payload)

    with pytest.raises(EmbeddingProviderError) as raised:
        OpenAICompatibleEmbeddingProvider("embed", "http://localhost:8000/v1").embed_text("hello")

    assert raised.value.provider == "openai_compatible"
    assert raised.value.error_type == "InvalidResponse"


def test_openai_compatible_embedding_dimension_mismatch(monkeypatch) -> None:
    install_response(monkeypatch, {"data": [{"index": 0, "embedding": [1, 2]}]})

    with pytest.raises(EmbeddingProviderError) as raised:
        OpenAICompatibleEmbeddingProvider(
            "embed", "http://localhost:8000/v1", expected_dimensions=3
        ).embed_text("hello")

    assert raised.value.error_type == "DimensionMismatch"


@pytest.mark.parametrize("payload", [b"not-json", b"\xff"])
def test_openai_compatible_embedding_rejects_invalid_json_or_encoding(monkeypatch, payload) -> None:
    install_response(monkeypatch, payload)

    with pytest.raises(EmbeddingProviderError) as raised:
        OpenAICompatibleEmbeddingProvider("embed", "http://localhost:8000/v1").embed_text("hello")

    assert raised.value.error_type == "InvalidResponse"


def test_openai_compatible_llm_success_and_request_options(monkeypatch) -> None:
    captured = install_response(
        monkeypatch,
        {"model": "chat-served", "choices": [{"message": {"content": "grounded answer"}}]},
    )
    provider = OpenAICompatibleLLMProvider("chat-requested", "https://compatible.example/v1")

    result = provider.generate(
        LLMRequest("Question", system_prompt="Use evidence", temperature=0.2, max_tokens=80)
    )

    assert result == LLMResponse("grounded answer", "chat-served", "openai_compatible")
    assert result.finish_reason is None
    request, _ = captured[0]
    assert request.full_url == "https://compatible.example/v1/chat/completions"
    assert request.get_header("Authorization") is None
    assert json.loads(request.data) == {
        "model": "chat-requested",
        "messages": [
            {"role": "system", "content": "Use evidence"},
            {"role": "user", "content": "Question"},
        ],
        "temperature": 0.2,
        "max_tokens": 80,
    }


def test_openai_compatible_llm_preserves_finish_reason(monkeypatch) -> None:
    install_response(
        monkeypatch,
        {
            "model": "chat-served",
            "choices": [
                {"finish_reason": "stop", "message": {"content": "grounded answer"}}
            ],
        },
    )

    result = OpenAICompatibleLLMProvider(
        "chat-requested", "https://compatible.example/v1"
    ).generate(LLMRequest("Question"))

    assert result.finish_reason == "stop"


def test_openai_compatible_llm_maps_json_response_format(monkeypatch) -> None:
    captured = install_response(
        monkeypatch,
        {"choices": [{"message": {"content": "{}"}}]},
    )

    OpenAICompatibleLLMProvider("chat", "https://compatible.example/v1").generate(
        LLMRequest("Question", response_format="json")
    )

    assert json.loads(captured[0][0].data) == {
        "model": "chat",
        "messages": [{"role": "user", "content": "Question"}],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }


def test_openai_compatible_llm_rejects_unknown_response_format(monkeypatch) -> None:
    captured = install_response(
        monkeypatch,
        {"choices": [{"message": {"content": "answer"}}]},
    )

    with pytest.raises(LLMProviderError) as raised:
        OpenAICompatibleLLMProvider("chat", "https://compatible.example/v1").generate(
            LLMRequest("Question", response_format="xml")
        )

    assert raised.value.error_type == "InvalidInput"
    assert captured == []


def test_openai_compatible_llm_sends_authorization_only_when_configured(monkeypatch) -> None:
    captured = install_response(
        monkeypatch, {"choices": [{"message": {"content": "answer"}}]}
    )

    OpenAICompatibleLLMProvider(
        "chat", "https://compatible.example/v1", api_key="test-secret"
    ).generate(LLMRequest("Question"))

    assert captured[0][0].get_header("Authorization") == "Bearer test-secret"


@pytest.mark.parametrize(
    ("payload", "error_type"),
    [
        (b"\xff", "invalid_response_encoding"),
        (b"not-json", "invalid_response_json"),
        ([], "invalid_response_top_level"),
        ({}, "invalid_response_choices"),
        ({"choices": [{}]}, "invalid_response_message"),
        ({"choices": [{"message": {}}]}, "invalid_response_content"),
        (
            {"choices": [{"message": {"content": "   "}}]},
            "invalid_response_empty_content",
        ),
    ],
)
def test_openai_compatible_llm_classifies_invalid_response(
    monkeypatch, payload, error_type: str
) -> None:
    install_response(monkeypatch, payload)

    with pytest.raises(LLMProviderError) as raised:
        OpenAICompatibleLLMProvider("chat", "http://localhost:8000/v1").generate(
            LLMRequest("Question")
        )

    assert raised.value.provider == "openai_compatible"
    assert raised.value.error_type == error_type


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        (TimeoutError("secret-timeout-detail"), "TimeoutError"),
        (urllib.error.URLError("secret-connection-detail"), "ConnectionError"),
        (
            urllib.error.HTTPError(
                "https://compatible.example/v1/embeddings",
                401,
                "unauthorized",
                {},
                io.BytesIO(b'{"error":"test-secret"}'),
            ),
            "HTTPError",
        ),
    ],
)
def test_openai_compatible_embedding_transport_errors_are_secret_safe(
    monkeypatch, failure, error_type
) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(failure))
    provider = OpenAICompatibleEmbeddingProvider(
        "embed", "https://compatible.example/v1", api_key="test-secret"
    )

    with pytest.raises(EmbeddingProviderError) as raised:
        provider.embed_text("hello")

    assert raised.value.error_type == error_type
    assert "test-secret" not in str(raised.value)
    assert "secret-" not in str(raised.value)
    assert raised.value.__cause__ is None


@pytest.mark.parametrize(
    ("failure", "error_type"),
    [
        (TimeoutError("secret-timeout-detail"), "TimeoutError"),
        (urllib.error.URLError("secret-connection-detail"), "ConnectionError"),
        (
            urllib.error.HTTPError(
                "https://compatible.example/v1/chat/completions",
                429,
                "limited",
                {},
                io.BytesIO(b'{"error":"test-secret"}'),
            ),
            "HTTPError",
        ),
    ],
)
def test_openai_compatible_llm_transport_errors_are_secret_safe(monkeypatch, failure, error_type) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(failure))
    provider = OpenAICompatibleLLMProvider(
        "chat", "https://compatible.example/v1", api_key="test-secret"
    )

    with pytest.raises(LLMProviderError) as raised:
        provider.generate(LLMRequest("Question"))

    assert raised.value.error_type == error_type
    assert "test-secret" not in str(raised.value)
    assert "secret-" not in str(raised.value)
    assert raised.value.__cause__ is None


def test_provider_factory_selects_ollama_without_migration() -> None:
    config = ProviderConfig(provider="ollama", llm_model="local-chat")

    embedding = create_embedding_provider(config)
    llm = create_llm_provider(config)

    assert isinstance(embedding, OllamaEmbeddingProvider)
    assert embedding.model == "nomic-embed-text-local:latest"
    assert embedding.base_url == "http://localhost:11434"
    assert isinstance(llm, OllamaLLMProvider)
    assert llm.model == "local-chat"


def test_provider_factory_selects_openai_compatible_and_hides_key() -> None:
    config = ProviderConfig(
        provider="openai_compatible",
        base_url="https://compatible.example/v1",
        api_key="test-secret",
        embedding_model="embed",
        llm_model="chat",
        embedding_dimensions=768,
    )

    assert isinstance(create_embedding_provider(config), OpenAICompatibleEmbeddingProvider)
    assert isinstance(create_llm_provider(config), OpenAICompatibleLLMProvider)
    assert "test-secret" not in repr(config)


def test_provider_config_reads_secret_only_from_environment() -> None:
    config = ProviderConfig.from_environment(
        environment={
            "CODECOMPASS_PROVIDER": "openai_compatible",
            "CODECOMPASS_BASE_URL": "http://localhost:8000/v1",
            "CODECOMPASS_API_KEY": "test-secret",
            "CODECOMPASS_EMBEDDING_MODEL": "embed",
            "CODECOMPASS_LLM_MODEL": "chat",
            "CODECOMPASS_TIMEOUT_SECONDS": "12.5",
            "CODECOMPASS_EMBEDDING_DIMENSIONS": "4",
        }
    )

    assert config.provider == "openai_compatible"
    assert config.api_key == "test-secret"
    assert config.timeout_seconds == 12.5
    assert config.embedding_dimensions == 4
    assert "test-secret" not in repr(config)


@pytest.mark.parametrize(
    "config",
    [
        ProviderConfig(provider="unknown", embedding_model="embed"),
        ProviderConfig(provider="openai_compatible", embedding_model="embed"),
        ProviderConfig(provider="openai_compatible", base_url="http://localhost:8000/v1"),
        ProviderConfig(
            provider="openai_compatible",
            base_url="http://user:password@localhost:8000/v1",
            embedding_model="embed",
        ),
    ],
)
def test_invalid_provider_combinations_fail_early(config) -> None:
    with pytest.raises(ValueError):
        create_embedding_provider(config)
