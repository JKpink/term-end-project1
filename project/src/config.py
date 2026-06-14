"""Pydantic Settings for AgentNet-DA."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Unified settings loaded from environment variables and .env file."""

    model_config = {"env_file": ".env", "extra": "ignore"}

    # LLM Backend: "dashscope" or "ollama"
    llm_backend: str = "dashscope"
    llm_model: str = "qwen-plus"
    llm_max_tokens: int = 4096
    llm_temperature: float = 0.0

    # DashScope API
    dashscope_api_key: str = ""
    dashscope_api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # Ollama Local
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3.5:9b"

    # Embedding
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # Agent
    agent_count: int = 4
    agent_max_iterations: int = 20
    agent_forward_path_max_length: int = 3

    # Experience Pool
    pool_size: int = 50
    pool_retrieval_count: int = 3
    pool_embedding_cache_limit: int = 1000
    pool_decay_rate: float = 0.1
    pool_decay_interval: int = 10

    # Paths
    output_dir: str = "./outputs"
    data_dir: str = "./data"

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)


_config: Optional[Settings] = None


def get_config() -> Settings:
    global _config
    if _config is None:
        _config = Settings()
    return _config
