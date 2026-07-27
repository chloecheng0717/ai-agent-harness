"""
Configuration Management

This module provides centralized configuration management using Pydantic Settings.
It supports environment variables, .env files, and provides validation for all settings.
"""

from functools import lru_cache
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Uses pydantic-settings for automatic environment variable parsing
    and validation. All settings can be overridden via environment variables
    with the prefix 'AI_AGENT_'.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        env_prefix="AI_AGENT_",
    )
    
    # Application
    app_name: str = Field(default="AI Agent Harness", description="Application name")
    app_version: str = Field(default="0.1.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    environment: str = Field(default="development", description="Environment name")
    
    # Server
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=1, description="Number of workers")
    
    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/ai_agent_harness",
        description="Database connection URL"
    )
    database_echo: bool = Field(default=False, description="Enable SQL logging")
    
    # Redis
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis URL")
    
    # LLM Configuration
    llm_provider: str = Field(default="openai", description="LLM provider type (openai, openrouter, custom)")
    openai_api_key: str = Field(default="", description="OpenAI-compatible API key")
    openai_api_base: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI-compatible API base URL"
    )
    default_model: str = Field(default="gpt-4", description="Default LLM model")
    llm_timeout: float = Field(default=60.0, description="LLM request timeout in seconds")
    llm_max_retries: int = Field(default=3, description="Maximum LLM retry attempts")
    
    # Vector Database
    chroma_host: str = Field(default="localhost", description="ChromaDB host")
    chroma_port: int = Field(default=8000, description="ChromaDB port")
    chroma_collection: str = Field(
        default="ai_agent_harness",
        description="ChromaDB collection name"
    )
    
    # Agent Configuration
    max_iterations: int = Field(default=10, description="Maximum agent iterations")
    max_tokens: int = Field(default=4096, description="Maximum tokens for LLM responses")
    temperature: float = Field(default=0.7, description="LLM temperature")
    
    # Security
    secret_key: str = Field(default="dev-secret-key-change-in-production", description="Secret key")
    allowed_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="CORS allowed origins"
    )
    
    # Logging
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(default="json", description="Logging format (json or text)")
    
    # Memory
    memory_enabled: bool = Field(default=True, description="Enable memory system")
    memory_short_term_max_size: int = Field(default=100, description="Short-term memory max entries")
    memory_long_term_collection: str = Field(default="ai_agent_harness", description="Long-term memory collection name")
    memory_episodic_database_url: str = Field(
        default="",
        description="Episodic memory PostgreSQL URL (empty for in-memory)",
    )
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment name."""
        allowed = ["development", "staging", "testing", "production"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v
    
    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, v: float) -> float:
        """Validate temperature is between 0 and 2."""
        if not 0 <= v <= 2:
            raise ValueError("Temperature must be between 0 and 2")
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def database_url_sync(self) -> str:
        """Get synchronous database URL for Alembic migrations."""
        return self.database_url.replace("+asyncpg", "")


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses lru_cache to ensure settings are loaded only once.
    This is important for performance and consistency.
    
    Returns:
        Settings: Application settings instance
    """
    return Settings()