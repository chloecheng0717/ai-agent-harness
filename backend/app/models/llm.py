"""
LLM Provider Models

This module defines the base interface and data structures for LLM provider
abstraction.  It allows the Orchestrator to communicate with any LLM backend
through a uniform async API while tracking token usage and model metadata.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enums ────────────────────────────────────────────────────────


class LLMProviderType(str, Enum):
    """Supported LLM provider types."""
    OPENAI = "openai"
    OPENROUTER = "openrouter"
    ANTHROPIC = "anthropic"
    CUSTOM = "custom"


# ── Request / Response DTOs ──────────────────────────────────────


class ChatMessage(BaseModel):
    """A single message in the chat completion request."""
    role: str = Field(..., description="Message role: 'system', 'user', 'assistant'")
    content: str = Field(..., description="Message content")
    name: Optional[str] = Field(default=None, description="Optional sender name")


class ChatCompletionRequest(BaseModel):
    """Request payload for chat completion."""
    messages: List[ChatMessage] = Field(..., description="Conversation messages")
    model: Optional[str] = Field(default=None, description="Override model name")
    temperature: Optional[float] = Field(default=None, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Max tokens to generate")
    stream: bool = Field(default=False, description="Whether to stream responses")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra provider-specific params")


class TokenUsageInfo(BaseModel):
    """Token usage returned by the provider."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionChoice(BaseModel):
    """A single completion choice."""
    index: int = 0
    message: ChatMessage = Field(default_factory=lambda: ChatMessage(role="assistant", content=""))
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """Response from a chat completion call."""
    id: str = ""
    model: str = ""
    choices: List[ChatCompletionChoice] = Field(default_factory=list)
    usage: TokenUsageInfo = Field(default_factory=TokenUsageInfo)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def content(self) -> str:
        """Convenience accessor for the first choice's text."""
        if self.choices:
            return self.choices[0].message.content
        return ""


class LLMProviderConfig(BaseModel):
    """Configuration for an LLM provider instance."""
    provider_type: LLMProviderType = Field(default=LLMProviderType.OPENAI)
    api_key: str = Field(default="", description="API key for the provider")
    base_url: str = Field(default="", description="Base URL for the API endpoint")
    model: str = Field(default="gpt-4", description="Default model name")
    temperature: float = Field(default=0.7, description="Default sampling temperature")
    max_tokens: int = Field(default=4096, description="Default max tokens")
    timeout: float = Field(default=60.0, description="Request timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")


class ModelInfo(BaseModel):
    """Metadata about the model served by the provider."""
    model_name: str = ""
    provider_type: LLMProviderType = LLMProviderType.OPENAI
    max_context_length: Optional[int] = None
    supports_streaming: bool = False
    extra: Dict[str, Any] = Field(default_factory=dict)


# ── Exceptions ───────────────────────────────────────────────────


class LLMProviderError(Exception):
    """Base exception for all LLM provider errors."""

    def __init__(self, message: str, provider: str = "", status_code: int = 0) -> None:
        self.provider = provider
        self.status_code = status_code
        super().__init__(message)


class LLMRateLimitError(LLMProviderError):
    """Raised when the provider returns a rate-limit (429) response."""


class LLMAuthenticationError(LLMProviderError):
    """Raised when the provider returns an authentication (401) response."""


class LLMTimeoutError(LLMProviderError):
    """Raised when a request to the provider times out."""


# ── Abstract Provider Interface ──────────────────────────────────


class BaseLLMProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Subclasses must implement :meth:`chat_completion` and :meth:`get_model_info`.
    The interface is intentionally async-first to integrate cleanly with the
    Orchestrator's lifecycle steps.
    """

    def __init__(self, config: LLMProviderConfig) -> None:
        self.config = config

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """
        Send a chat completion request to the LLM provider.

        Parameters
        ----------
        request : ChatCompletionRequest
            The completion request payload.

        Returns
        -------
        ChatCompletionResponse
            The provider's response.

        Raises
        ------
        LLMProviderError
            On provider-level errors (auth, rate-limit, timeout, etc.).
        """

    @abstractmethod
    async def get_model_info(self) -> ModelInfo:
        """
        Return metadata about the currently configured model.

        Returns
        -------
        ModelInfo
            Model metadata including name, context length, etc.
        """

    async def health_check(self) -> bool:
        """
        Lightweight connectivity check.

        The default implementation sends a minimal request; providers may
        override for more efficient probes.

        Returns
        -------
        bool
            ``True`` if the provider is reachable and healthy.
        """
        try:
            request = ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="ping")],
                max_tokens=1,
            )
            await self.chat_completion(request)
            return True
        except Exception:
            return False

    async def close(self) -> None:
        """Release any held resources (e.g. HTTP clients)."""

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}"
            f" model={self.config.model!r}"
            f" provider={self.config.provider_type.value}>"
        )
