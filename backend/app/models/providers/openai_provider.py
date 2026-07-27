"""
OpenAI-Compatible LLM Provider

Supports any endpoint that speaks the OpenAI chat completion protocol:
OpenAI, OpenRouter, Azure OpenAI, vLLM, Ollama, and other compatible
services.

Uses ``httpx.AsyncClient`` for non-blocking HTTP with configurable
timeouts, retries, and structured logging.
"""

import asyncio
import time
from typing import Any, Dict, Optional

import httpx
import structlog

from app.models.llm import (
    BaseLLMProvider,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    LLMAuthenticationError,
    LLMProviderConfig,
    LLMProviderError,
    LLMProviderType,
    LLMRateLimitError,
    LLMTimeoutError,
    ModelInfo,
    TokenUsageInfo,
)

logger = structlog.get_logger().bind(component="llm_provider")


class OpenAIProvider(BaseLLMProvider):
    """
    OpenAI-compatible chat completion provider.

    Works with any API that implements the ``/chat/completions`` endpoint
    (OpenAI, OpenRouter, local LLM servers, etc.).

    Parameters
    ----------
    config : LLMProviderConfig
        Provider configuration.  If *not* supplied the provider falls back
        to reading ``AI_AGENT_OPENAI_API_KEY``, ``AI_AGENT_OPENAI_API_BASE``,
        and ``AI_AGENT_DEFAULT_MODEL`` from the environment (via the
        ``Settings`` object).
    """

    def __init__(self, config: Optional[LLMProviderConfig] = None) -> None:
        if config is None:
            config = self._config_from_settings()
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self.logger = logger.bind(
            model=self.config.model,
            provider=self.config.provider_type.value,
        )
        self.logger.info("OpenAIProvider initialised", base_url=self.config.base_url)

    # ── factory helpers ───────────────────────────────────────

    @classmethod
    def _config_from_settings(cls) -> LLMProviderConfig:
        """Build config lazily from the application Settings singleton."""
        from app.config import get_settings

        settings = get_settings()
        return LLMProviderConfig(
            provider_type=LLMProviderType(settings.llm_provider),
            api_key=settings.openai_api_key,
            base_url=settings.openai_api_base,
            model=settings.default_model,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            timeout=settings.llm_timeout,
            max_retries=settings.llm_max_retries,
        )

    # ── HTTP client lifecycle ─────────────────────────────────

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(self.config.timeout),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
            self.logger.debug("HTTP client closed")

    # ── core API ──────────────────────────────────────────────

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """
        Send a chat completion request with automatic retry.

        Retries on transient errors (429, 500-599, timeouts) up to
        ``config.max_retries`` times with exponential back-off.

        Parameters
        ----------
        request : ChatCompletionRequest
            The completion request.

        Returns
        -------
        ChatCompletionResponse
            Parsed response from the provider.

        Raises
        ------
        LLMAuthenticationError
            If the API key is invalid (401).
        LLMRateLimitError
            If retries for 429 are exhausted.
        LLMProviderError
            On other HTTP errors or malformed responses.
        """
        payload = self._build_payload(request)
        client = self._get_client()
        last_error: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            start = time.monotonic()
            self.logger.debug(
                "chat_completion request",
                attempt=attempt,
                model=payload.get("model"),
                message_count=len(request.messages),
            )

            try:
                http_response = await client.post("/chat/completions", json=payload)
                elapsed_ms = (time.monotonic() - start) * 1000

                if http_response.status_code == 401:
                    raise LLMAuthenticationError(
                        "Invalid API key",
                        provider=self.config.provider_type.value,
                        status_code=401,
                    )

                if http_response.status_code == 429:
                    error_body = http_response.text
                    self.logger.warning(
                        "Rate limited",
                        attempt=attempt,
                        body=error_body,
                        ms=round(elapsed_ms, 2),
                    )
                    last_error = LLMRateLimitError(
                        f"Rate limited (429): {error_body}",
                        provider=self.config.provider_type.value,
                        status_code=429,
                    )
                    if attempt < self.config.max_retries:
                        backoff = min(2 ** attempt, 30)
                        await asyncio.sleep(backoff)
                        continue
                    raise last_error

                if http_response.status_code >= 500:
                    error_body = http_response.text
                    self.logger.warning(
                        "Server error",
                        status_code=http_response.status_code,
                        attempt=attempt,
                        body=error_body,
                        ms=round(elapsed_ms, 2),
                    )
                    last_error = LLMProviderError(
                        f"Server error ({http_response.status_code}): {error_body}",
                        provider=self.config.provider_type.value,
                        status_code=http_response.status_code,
                    )
                    if attempt < self.config.max_retries:
                        backoff = min(2 ** attempt, 30)
                        await asyncio.sleep(backoff)
                        continue
                    raise last_error

                if http_response.status_code != 200:
                    raise LLMProviderError(
                        f"Unexpected status {http_response.status_code}: {http_response.text}",
                        provider=self.config.provider_type.value,
                        status_code=http_response.status_code,
                    )

                data = http_response.json()
                response = self._parse_response(data)

                self.logger.info(
                    "chat_completion success",
                    model=response.model,
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    ms=round(elapsed_ms, 2),
                )
                return response

            except (LLMAuthenticationError, LLMProviderError):
                raise
            except httpx.TimeoutException as exc:
                elapsed_ms = (time.monotonic() - start) * 1000
                self.logger.warning(
                    "Request timeout",
                    attempt=attempt,
                    ms=round(elapsed_ms, 2),
                )
                last_error = LLMTimeoutError(
                    f"Request timed out after {self.config.timeout}s",
                    provider=self.config.provider_type.value,
                )
                if attempt < self.config.max_retries:
                    backoff = min(2 ** attempt, 30)
                    await asyncio.sleep(backoff)
                    continue
                raise last_error from exc
            except httpx.HTTPError as exc:
                elapsed_ms = (time.monotonic() - start) * 1000
                self.logger.error(
                    "HTTP error",
                    error=str(exc),
                    attempt=attempt,
                    ms=round(elapsed_ms, 2),
                )
                last_error = LLMProviderError(
                    f"HTTP error: {exc}",
                    provider=self.config.provider_type.value,
                )
                if attempt < self.config.max_retries:
                    backoff = min(2 ** attempt, 30)
                    await asyncio.sleep(backoff)
                    continue
                raise last_error from exc

        raise last_error or LLMProviderError("All retries exhausted")

    async def get_model_info(self) -> ModelInfo:
        """Return metadata about the configured model."""
        return ModelInfo(
            model_name=self.config.model,
            provider_type=self.config.provider_type,
            supports_streaming=True,
            extra={"base_url": self.config.base_url},
        )

    # ── internal helpers ──────────────────────────────────────

    def _build_payload(self, request: ChatCompletionRequest) -> Dict[str, Any]:
        """Build the JSON payload for the OpenAI-compatible endpoint."""
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        if request.model:
            model = request.model
        else:
            model = self.config.model

        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        else:
            payload["temperature"] = self.config.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        else:
            payload["max_tokens"] = self.config.max_tokens
        if request.stream:
            payload["stream"] = True

        payload.update(request.metadata)
        return payload

    @staticmethod
    def _parse_response(data: Dict[str, Any]) -> ChatCompletionResponse:
        """Parse the raw JSON response into a ``ChatCompletionResponse``."""
        choices = []
        for raw_choice in data.get("choices", []):
            raw_msg = raw_choice.get("message", {})
            choices.append(
                ChatCompletionChoice(
                    index=raw_choice.get("index", 0),
                    message=ChatMessage(
                        role=raw_msg.get("role", "assistant"),
                        content=raw_msg.get("content", ""),
                    ),
                    finish_reason=raw_choice.get("finish_reason"),
                )
            )

        raw_usage = data.get("usage", {})
        usage = TokenUsageInfo(
            prompt_tokens=raw_usage.get("prompt_tokens", 0),
            completion_tokens=raw_usage.get("completion_tokens", 0),
            total_tokens=raw_usage.get("total_tokens", 0),
        )

        return ChatCompletionResponse(
            id=data.get("id", ""),
            model=data.get("model", ""),
            choices=choices,
            usage=usage,
            metadata={
                k: v
                for k, v in data.items()
                if k not in ("id", "model", "choices", "usage", "created")
            },
        )
