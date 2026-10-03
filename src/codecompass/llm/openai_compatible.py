"""LLM provider for OpenAI-compatible chat-completion endpoints."""

from __future__ import annotations

import os
from typing import Any

from codecompass._openai_compatible_http import (
    OpenAICompatibleHTTPError,
    post_json,
    validate_base_url,
)
from codecompass.llm.base import LLMProviderError, LLMRequest, LLMResponse


class OpenAICompatibleLLMProvider:
    """Generate non-streaming text through `/chat/completions`."""

    def __init__(
        self,
        model: str,
        base_url: str,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("OpenAI-compatible LLM model is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.model = model.strip()
        self.base_url = validate_base_url(base_url)
        self.api_key = api_key or None
        self.timeout_seconds = timeout_seconds

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate text using the current provider-neutral request model."""
        self._validate_request(request)
        response = self._post_json(self._payload(request))
        text, finish_reason = self._response(response)
        model = response.get("model") if isinstance(response.get("model"), str) else self.model
        return LLMResponse(
            text=text,
            model=model,
            provider="openai_compatible",
            finish_reason=finish_reason,
        )

    def _payload(self, request: LLMRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system_prompt is not None:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": float(request.temperature),
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.response_format == "json":
            payload["response_format"] = {"type": "json_object"}
        reasoning_effort = os.getenv("CODECOMPASS_LLM_REASONING_EFFORT")
        if reasoning_effort in {"low", "high", "max"}:
            payload["reasoning_effort"] = reasoning_effort
        return payload

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return post_json(
                self.base_url,
                "chat/completions",
                payload,
                api_key=self.api_key,
                timeout_seconds=self.timeout_seconds,
            )
        except OpenAICompatibleHTTPError as error:
            raise self._error(error.error_type, error.message) from None

    def _response(self, response: dict[str, Any]) -> tuple[str, str | None]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise self._error(
                "invalid_response_choices", "OpenAI-compatible response missing choices"
            )
        choice = choices[0]
        message = choice.get("message")
        if not isinstance(message, dict):
            raise self._error(
                "invalid_response_message", "OpenAI-compatible response missing message"
            )
        text = message.get("content")
        if not isinstance(text, str):
            raise self._error(
                "invalid_response_content", "OpenAI-compatible response missing message content"
            )
        if not text.strip():
            raise self._error(
                "invalid_response_empty_content",
                "OpenAI-compatible generated text must not be empty",
            )
        finish_reason = choice.get("finish_reason")
        return text, finish_reason if isinstance(finish_reason, str) else None

    def _validate_request(self, request: LLMRequest) -> None:
        if not isinstance(request.prompt, str) or not request.prompt.strip():
            raise self._error("InvalidInput", "Prompt must be a non-empty string")
        if request.system_prompt is not None and not isinstance(request.system_prompt, str):
            raise self._error("InvalidInput", "System prompt must be a string or None")
        if isinstance(request.temperature, bool) or not isinstance(request.temperature, (int, float)) or request.temperature < 0:
            raise self._error("InvalidInput", "Temperature must be a non-negative number")
        if request.max_tokens is not None:
            if isinstance(request.max_tokens, bool) or not isinstance(request.max_tokens, int) or request.max_tokens < 1:
                raise self._error("InvalidInput", "max_tokens must be a positive integer or None")
        if request.response_format not in (None, "json"):
            raise self._error("InvalidInput", "Unsupported response_format")

    def _error(self, error_type: str, message: str) -> LLMProviderError:
        return LLMProviderError("openai_compatible", self.model, error_type, message)
