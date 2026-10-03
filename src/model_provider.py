from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class ProviderConfig:
    """Provider configuration shared by the agents.

    Supported providers:
    - openai
    - custom (OpenAI-compatible base URL)
    - gemini
    - anthropic
    - ollama
    - openrouter
    """

    provider: str
    model_name: str
    temperature: float = 0.0
    api_key: str | None = None
    base_url: str | None = None


def normalize_provider(value: str) -> str:
    """Map aliases and normalize provider names."""
    if not value:
        return "openai"
    val = value.strip().lower()
    mapping = {
        "anthorpic": "anthropic",
        "google": "gemini",
        "google-genai": "gemini",
        "gemini-genai": "gemini",
        "ollama_ai": "ollama",
        "open_router": "openrouter",
    }
    return mapping.get(val, val)


def build_chat_model(config: ProviderConfig):
    """Instantiate the real chat model for the selected provider."""
    provider = normalize_provider(config.provider)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatOpenAI(**kwargs)

    elif provider == "custom":
        from langchain_openai import ChatOpenAI
        kwargs = {
            "model": config.model_name,
            "temperature": config.temperature,
            "api_key": config.api_key or "custom-key",
        }
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOpenAI(**kwargs)

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["google_api_key"] = config.api_key
        return ChatGoogleGenerativeAI(**kwargs)

    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        kwargs = {"model_name": config.model_name, "temperature": config.temperature}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        return ChatAnthropic(**kwargs)

    elif provider == "ollama":
        from langchain_ollama import ChatOllama
        kwargs = {"model": config.model_name, "temperature": config.temperature}
        if config.base_url:
            kwargs["base_url"] = config.base_url
        return ChatOllama(**kwargs)

    elif provider == "openrouter":
        from langchain_openai import ChatOpenAI
        api_key = config.api_key or os.getenv("OPENROUTER_API_KEY")
        base_url = config.base_url or "https://openrouter.ai/api/v1"
        return ChatOpenAI(
            model=config.model_name,
            temperature=config.temperature,
            api_key=api_key or "openrouter-key",
            base_url=base_url,
        )

    else:
        raise ValueError(f"Unsupported provider: {config.provider}")

