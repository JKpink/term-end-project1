"""LLM client supporting DashScope API and local Ollama."""
from __future__ import annotations

import json
import re
from typing import Any, Optional

from openai import OpenAI

from .config import get_config


class LLMClient:
    """Unified LLM client. Switch backend via LLM_BACKEND env var."""

    def __init__(self):
        cfg = get_config()
        backend = cfg.llm_backend

        if backend == "ollama":
            self.client = OpenAI(
                base_url=cfg.ollama_base_url,
                api_key="ollama",
            )
            self.model = cfg.ollama_model
        else:
            self.client = OpenAI(
                api_key=cfg.dashscope_api_key,
                base_url=cfg.dashscope_api_base,
            )
            self.model = cfg.llm_model

        self.temperature = cfg.llm_temperature
        self.max_tokens = cfg.llm_max_tokens

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[dict] = None,
    ) -> str:
        """Send a chat completion request, return text response."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        resp = self.client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def chat_json(
        self,
        messages: list[dict[str, str]],
        temperature: Optional[float] = None,
    ) -> dict:
        """Send a chat request expecting JSON output, parse the result."""
        raw = self.chat(messages, temperature=temperature)
        return extract_json(raw)


def extract_json(text: str) -> dict:
    """Extract a JSON object from LLM output (handles markdown code fences)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {}
