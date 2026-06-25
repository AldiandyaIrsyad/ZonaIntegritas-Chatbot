"""Configuration for the dataset generation pipeline.

Skripsi §3.2.1c defines:
    - Generator: DeepSeek V4
    - Panel: GLM 5.2, DeepSeek V4, Gemini 3.1 Pro, Llama 3, Mistral
    - Temperature: 0.0 (deterministic)
    - Acceptance threshold: ≥4/5 majority vote
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetGenSettings(BaseSettings):
    """Settings for the dataset generation pipeline.

    All values can be overridden via environment variables with prefix
    ``DATAGEN_`` (e.g., ``DATAGEN_OPENROUTER_API_KEY``).
    """

    model_config = SettingsConfigDict(env_file=".env", env_prefix="DATAGEN_", extra="ignore")

    # --- API ---
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key for LLM access",
    )
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )

    # --- Generator ---
    generator_model: str = Field(
        default="deepseek/deepseek-chat",
        description="Model for draft generation (DeepSeek V4)",
    )
    generator_temperature: float = Field(
        default=0.0,
        description="Temperature for generation (0.0 = deterministic)",
    )

    # --- Evaluator Panel ---
    panel_models: str = Field(
        default="google/gemini-3.1-pro,meta-llama/llama-3-70b-instruct,mistralai/mistral-large,deepseek/deepseek-chat,zai-org/glm-5.2",
        description="Comma-separated list of 5 panel models for majority voting",
    )
    panel_temperature: float = Field(
        default=0.0,
        description="Temperature for panel evaluation (0.0 = deterministic)",
    )
    acceptance_threshold: int = Field(
        default=4,
        description="Minimum votes out of 5 to accept an item (≥4/5 = accept)",
    )

    # --- Output ---
    output_dir: str = Field(
        default="data",
        description="Directory for generated CSV files",
    )

    # --- Limits ---
    max_samples: int = Field(
        default=0,
        description="Maximum samples to generate (0 = no limit). For cost control.",
    )

    @property
    def panel_model_list(self) -> List[str]:
        """Parse panel_models into a list.

        Returns:
            List of model identifiers.
        """
        return [m.strip() for m in self.panel_models.split(",") if m.strip()]


@lru_cache
def get_dataset_gen_settings() -> DatasetGenSettings:
    """Return the cached DatasetGenSettings singleton.

    Returns:
        DatasetGenSettings instance.
    """
    return DatasetGenSettings()
