"""Configuration settings for the dashboard backend."""
import os
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""

    # API Settings
    api_title: str = "Agentic AI Toolkit Dashboard API"
    api_version: str = "1.0.0"
    api_prefix: str = "/api/v1"
    debug: bool = True

    # CORS
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost"]

    # Ollama Settings
    ollama_host: str = "host.docker.internal"
    ollama_port: int = 11434
    default_model: str = "gemma2:2b"

    # Cost Rates ($ per token) - 4-Component Model
    input_token_rate: float = 0.000001  # C_inference input
    output_token_rate: float = 0.000002  # C_inference output
    latency_rate: float = 0.0001  # C_latency per second
    tool_rate: float = 0.0001  # C_tools per call
    human_rate: float = 0.50  # C_human per minute

    # Model-specific token rates
    token_rates: dict = {
        "gemma2:2b": 0.00001,
        "phi3:latest": 0.00001,
        "llama3.2:latest": 0.00002,
        "mistral:latest": 0.00003,
        "llama3.1:8b": 0.00005,
        "qwen2:7b": 0.0001,
    }

    # Default cost rates (legacy)
    default_tool_cost: float = 0.01
    default_latency_rate: float = 0.001  # per second
    default_human_cost: float = 5.0  # per intervention

    # Redis (for WebSocket pub/sub)
    redis_url: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()
