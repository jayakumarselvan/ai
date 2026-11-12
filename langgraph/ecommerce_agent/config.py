"""Runtime configuration, read from environment variables (see .env.example)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    llm_model: str
    llm_api_base: str | None
    llm_temperature: float
    refund_approval_threshold: float
    return_window_days: int


def get_settings() -> Settings:
    return Settings(
        # Any LiteLLM model string, e.g. "gpt-4o-mini", "ollama/llama3.1", "watsonx/ibm/granite-3-8b-instruct"
        llm_model=os.environ.get("LLM_MODEL", "ollama/llama3.1"),
        llm_api_base=os.environ.get("LLM_API_BASE") or None,
        llm_temperature=float(os.environ.get("LLM_TEMPERATURE", "0")),
        # Refunds strictly greater than this amount need human (support) approval.
        refund_approval_threshold=float(os.environ.get("REFUND_APPROVAL_THRESHOLD", "100")),
        return_window_days=int(os.environ.get("RETURN_WINDOW_DAYS", "30")),
    )
