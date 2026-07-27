"""
Unit Tests for LLM Provider Layer

Tests cover:
- BaseLLMProvider interface and data models
- OpenAI provider construction and config
- OpenAI provider chat completion (with mocked HTTP)
- Retry logic on transient errors
- Error mapping (auth, rate-limit, timeout)
- Model info and health check
"""

import pytest
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

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
from app.models.providers.openai_provider import OpenAIProvider


# ── Data model tests ─────────────────────────────────────────────


class TestChatMessage:
    def test_create(self) -> None:
        msg = ChatMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.name is None

    def test_with_name(self) -> None:
        msg = ChatMessage(role="assistant", content="hi", name="bot")
        assert msg.name == "bot"


class TestChatCompletionRequest:
    def test_minimal(self) -> None:
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        assert len(req.messages) == 1
        assert req.model is None
        assert req.stream is False

    def test_full(self) -> None:
        req = ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hi")],
            model="gpt-4o",
            temperature=0.5,
            max_tokens=100,
            stream=True,
            metadata={"top_p": 0.9},
        )
        assert req.model == "gpt-4o"
        assert req.temperature == 0.5
        assert req.max_tokens == 100
        assert req.stream is True
        assert req.metadata["top_p"] == 0.9


class TestTokenUsageInfo:
    def test_defaults(self) -> None:
        usage = TokenUsageInfo()
        assert usage.prompt_tokens == 0
        assert usage.completion_tokens == 0
        assert usage.total_tokens == 0

    def test_custom(self) -> None:
        usage = TokenUsageInfo(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        assert usage.total_tokens == 30


class TestChatCompletionResponse:
    def test_empty(self) -> None:
        resp = ChatCompletionResponse()
        assert resp.content == ""
        assert resp.choices == []

    def test_content_property(self) -> None:
        resp = ChatCompletionResponse(
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content="Hello!")
                )
            ]
        )
        assert resp.content == "Hello!"


class TestLLMProviderConfig:
    def test_defaults(self) -> None:
        cfg = LLMProviderConfig()
        assert cfg.provider_type == LLMProviderType.OPENAI
        assert cfg.model == "gpt-4"
        assert cfg.temperature == 0.7
        assert cfg.timeout == 60.0
        assert cfg.max_retries == 3

    def test_custom(self) -> None:
        cfg = LLMProviderConfig(
            provider_type=LLMProviderType.OPENROUTER,
            api_key="sk-test",
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3",
            timeout=30.0,
            max_retries=5,
        )
        assert cfg.provider_type == LLMProviderType.OPENROUTER
        assert cfg.api_key == "sk-test"
        assert cfg.max_retries == 5


class TestModelInfo:
    def test_defaults(self) -> None:
        info = ModelInfo()
        assert info.model_name == ""
        assert info.supports_streaming is False


class TestExceptions:
    def test_provider_error(self) -> None:
        err = LLMProviderError("boom", provider="openai", status_code=500)
        assert str(err) == "boom"
        assert err.provider == "openai"
        assert err.status_code == 500

    def test_rate_limit_error(self) -> None:
        err = LLMRateLimitError("rate limited", provider="openai", status_code=429)
        assert isinstance(err, LLMProviderError)

    def test_auth_error(self) -> None:
        err = LLMAuthenticationError("bad key", status_code=401)
        assert isinstance(err, LLMProviderError)

    def test_timeout_error(self) -> None:
        err = LLMTimeoutError("timed out")
        assert isinstance(err, LLMProviderError)


# ── Concrete provider subclass for interface testing ─────────────


class StubProvider(BaseLLMProvider):
    """Minimal concrete implementation for testing the ABC."""

    async def chat_completion(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        return ChatCompletionResponse(
            id="stub-1",
            model=self.config.model,
            choices=[
                ChatCompletionChoice(
                    message=ChatMessage(role="assistant", content="stub response")
                )
            ],
            usage=TokenUsageInfo(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        )

    async def get_model_info(self) -> ModelInfo:
        return ModelInfo(model_name=self.config.model)


class TestBaseLLMProvider:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseLLMProvider(LLMProviderConfig())  # type: ignore

    def test_stub_provider(self) -> None:
        provider = StubProvider(LLMProviderConfig(model="test-model"))
        assert provider.config.model == "test-model"

    @pytest.mark.asyncio
    async def test_stub_chat_completion(self) -> None:
        provider = StubProvider(LLMProviderConfig())
        req = ChatCompletionRequest(messages=[ChatMessage(role="user", content="hi")])
        resp = await provider.chat_completion(req)
        assert resp.content == "stub response"
        assert resp.usage.total_tokens == 10

    @pytest.mark.asyncio
    async def test_stub_model_info(self) -> None:
        provider = StubProvider(LLMProviderConfig(model="my-model"))
        info = await provider.get_model_info()
        assert info.model_name == "my-model"

    @pytest.mark.asyncio
    async def test_health_check_success(self) -> None:
        provider = StubProvider(LLMProviderConfig())
        assert await provider.health_check() is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self) -> None:
        class FailingProvider(BaseLLMProvider):
            async def chat_completion(self, request):
                raise LLMProviderError("unavailable")

            async def get_model_info(self):
                return ModelInfo()

        provider = FailingProvider(LLMProviderConfig())
        assert await provider.health_check() is False

    @pytest.mark.asyncio
    async def test_close_is_noop(self) -> None:
        provider = StubProvider(LLMProviderConfig())
        await provider.close()

    def test_repr(self) -> None:
        provider = StubProvider(LLMProviderConfig(model="gpt-4"))
        r = repr(provider)
        assert "StubProvider" in r
        assert "gpt-4" in r


# ── OpenAI provider tests ────────────────────────────────────────


class TestOpenAIProviderInit:
    def test_init_with_config(self) -> None:
        cfg = LLMProviderConfig(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model="my-model",
        )
        provider = OpenAIProvider(cfg)
        assert provider.config.api_key == "sk-test"
        assert provider.config.model == "my-model"

    @patch("app.config.get_settings")
    def test_init_from_settings(self, mock_settings: MagicMock) -> None:
        mock_settings.return_value = MagicMock(
            llm_provider="openai",
            openai_api_key="sk-settings",
            openai_api_base="https://api.openai.com/v1",
            default_model="gpt-4",
            temperature=0.5,
            max_tokens=2048,
            llm_timeout=30.0,
            llm_max_retries=2,
        )
        provider = OpenAIProvider()
        assert provider.config.api_key == "sk-settings"
        assert provider.config.model == "gpt-4"
        assert provider.config.max_retries == 2


class TestOpenAIProviderChatCompletion:
    """Tests using a mocked httpx client."""

    def _make_provider(self) -> OpenAIProvider:
        cfg = LLMProviderConfig(
            api_key="sk-test",
            base_url="https://api.example.com/v1",
            model="test-model",
            max_retries=2,
            timeout=5.0,
        )
        return OpenAIProvider(cfg)

    def _mock_response(
        self,
        status_code: int = 200,
        json_data: Dict[str, Any] | None = None,
        text: str = "",
    ) -> httpx.Response:
        import json as _json

        if json_data is None:
            json_data = {
                "id": "chatcmpl-123",
                "model": "test-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello there!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                    "total_tokens": 15,
                },
            }
        if text:
            content = text.encode()
        else:
            content = _json.dumps(json_data).encode()
        response = httpx.Response(
            status_code=status_code,
            content=content,
            headers={"content-type": "application/json"} if not text else {},
            request=httpx.Request("POST", "https://api.example.com/v1/chat/completions"),
        )
        return response

    @pytest.mark.asyncio
    async def test_successful_completion(self) -> None:
        provider = self._make_provider()
        mock_response = self._mock_response()

        with patch.object(provider, "_get_client") as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = client

            req = ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")]
            )
            resp = await provider.chat_completion(req)

            assert resp.content == "Hello there!"
            assert resp.model == "test-model"
            assert resp.usage.prompt_tokens == 10
            assert resp.usage.completion_tokens == 5
            assert resp.id == "chatcmpl-123"

    @pytest.mark.asyncio
    async def test_auth_error(self) -> None:
        provider = self._make_provider()
        mock_response = self._mock_response(status_code=401, text="Unauthorized")

        with patch.object(provider, "_get_client") as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = client

            req = ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")]
            )
            with pytest.raises(LLMAuthenticationError):
                await provider.chat_completion(req)

    @pytest.mark.asyncio
    async def test_rate_limit_exhausts_retries(self) -> None:
        provider = self._make_provider()
        provider.config.max_retries = 2
        mock_response = self._mock_response(status_code=429, text="Rate limited")

        with patch.object(provider, "_get_client") as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = client

            req = ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")]
            )
            with pytest.raises(LLMRateLimitError):
                await provider.chat_completion(req)
            assert client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_server_error_retries(self) -> None:
        provider = self._make_provider()
        provider.config.max_retries = 3

        error_response = self._mock_response(status_code=500, text="Internal error")
        success_response = self._mock_response()

        call_count = 0

        async def side_effect(*args: Any, **kwargs: Any) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return error_response
            return success_response

        with patch.object(provider, "_get_client") as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(side_effect=side_effect)
            mock_get.return_value = client

            req = ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")]
            )
            resp = await provider.chat_completion(req)
            assert resp.content == "Hello there!"
            assert call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_retries(self) -> None:
        provider = self._make_provider()
        provider.config.max_retries = 2

        with patch.object(provider, "_get_client") as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_get.return_value = client

            req = ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")]
            )
            with pytest.raises(LLMTimeoutError):
                await provider.chat_completion(req)
            assert client.post.call_count == 2

    @pytest.mark.asyncio
    async def test_custom_model_override(self) -> None:
        provider = self._make_provider()
        mock_response = self._mock_response()

        with patch.object(provider, "_get_client") as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = client

            req = ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")],
                model="custom-model",
            )
            await provider.chat_completion(req)

            call_args = client.post.call_args
            payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
            assert payload["model"] == "custom-model"

    @pytest.mark.asyncio
    async def test_temperature_and_max_tokens_defaults(self) -> None:
        provider = self._make_provider()
        mock_response = self._mock_response()

        with patch.object(provider, "_get_client") as mock_get:
            client = AsyncMock()
            client.post = AsyncMock(return_value=mock_response)
            mock_get.return_value = client

            req = ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="Hi")]
            )
            await provider.chat_completion(req)

            call_args = client.post.call_args
            payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
            assert payload["temperature"] == 0.7
            assert payload["max_tokens"] == 4096


class TestOpenAIProviderModelInfo:
    @pytest.mark.asyncio
    async def test_get_model_info(self) -> None:
        cfg = LLMProviderConfig(
            model="claude-3",
            provider_type=LLMProviderType.OPENROUTER,
            base_url="https://openrouter.ai/api/v1",
        )
        provider = OpenAIProvider(cfg)
        info = await provider.get_model_info()
        assert info.model_name == "claude-3"
        assert info.provider_type == LLMProviderType.OPENROUTER
        assert info.supports_streaming is True


class TestOpenAIProviderClose:
    @pytest.mark.asyncio
    async def test_close(self) -> None:
        provider = OpenAIProvider(LLMProviderConfig(api_key="sk-test"))
        client = AsyncMock()
        client.is_closed = False
        client.aclose = AsyncMock()
        provider._client = client

        await provider.close()
        client.aclose.assert_called_once()
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_close_when_not_initialised(self) -> None:
        provider = OpenAIProvider(LLMProviderConfig(api_key="sk-test"))
        provider._client = None
        await provider.close()


class TestOpenAIProviderParseResponse:
    def test_parse_minimal(self) -> None:
        data = {
            "id": "cmpl-1",
            "model": "gpt-4",
            "choices": [],
            "usage": {},
        }
        resp = OpenAIProvider._parse_response(data)
        assert resp.id == "cmpl-1"
        assert resp.choices == []
        assert resp.usage.prompt_tokens == 0

    def test_parse_full(self) -> None:
        data = {
            "id": "cmpl-2",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "test"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            "created": 1234567890,
            "object": "chat.completion",
        }
        resp = OpenAIProvider._parse_response(data)
        assert resp.id == "cmpl-2"
        assert resp.model == "gpt-4o"
        assert len(resp.choices) == 1
        assert resp.choices[0].message.content == "test"
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.total_tokens == 150
        assert "created" not in resp.metadata
        assert resp.metadata.get("object") == "chat.completion"
