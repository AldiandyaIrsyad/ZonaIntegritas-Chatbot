"""Configuration for the dataset generation pipeline.

Skripsi §3.2.1c defines a Generator–Evaluator architecture with a cost-conscious
model selection (binary YES/NO or single-label judging does not need frontier
reasoning models):
    - Generator: DeepSeek (deepseek-v4-pro) — cheap, strong instruction-following
    - Panel (5 distinct labs, generator excluded to avoid self-eval bias), all
      verified live on OpenRouter: Gemini 2.5 Flash, Llama 3.3 70B, Qwen3.7
      Plus, Mistral Large 2512, and GPT-5.6-Luna as a premium tie-breaker
      voice (the other 4 are cheap; panel votes are short YES/NO/label
      outputs so the tie-breaker's higher per-token rate adds negligible
      absolute cost).
    - Temperature: 0.0 (deterministic)
    - Acceptance threshold: ≥4/5 majority vote

Model slugs must resolve on OpenRouter; an invalid slug fails closed (panel
votes NO) and silently rejects every candidate. Verify with a live /models
check before a full run.

NOTE: the "flash-lite"/"flash" (smallest) tier of a model family is not
always reliable enough for judging tasks — during real generation runs,
gemini-2.5-flash-lite gave unexplained bare "NO" votes and glm-4.7-flash
sometimes exhausted its token budget on hidden reasoning and returned empty
content (which fails closed as NO). Both were swapped out after empirical
verification; mistral-small-3.2-24b (smallest Mistral tier) was similarly
upgraded to mistral-large-2512 as a precaution.
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
        description="Model for draft generation (DeepSeek). Kept out of the panel.",
    )
    generator_temperature: float = Field(
        default=0.0,
        description="Temperature for generation (0.0 = deterministic)",
    )

    # --- Evaluator Panel ---
    # 5 distinct-lab models (verified OpenRouter list prices, 2026-07-22):
    #   google/gemini-2.5-flash               $0.30  / $2.50  per 1M tokens
    #   meta-llama/llama-3.3-70b-instruct     $0.10  / $0.32  per 1M tokens
    #   qwen/qwen3.7-plus                     $0.32  / $1.28  per 1M tokens
    #   mistralai/mistral-small-3.2-24b-instruct $0.075/$0.20 per 1M tokens
    #   openai/gpt-oss-20b                    $0.029 / $0.14  per 1M tokens
    # Keeps the original panel size/quality (5 members, majority-of-4) —
    # a 3-model panel of smaller checkpoints was tried and reverted: fewer,
    # weaker raters isn't a good trade against the actual cost driver, which
    # was specific expensive members, not panel size. Two swaps vs. the
    # original default, both same-lab (keeps 5 distinct labs) and both real
    # mid/large checkpoints, not downgrades to tiny ones:
    #   - openai/gpt-5.6-luna ($1.00/$6.00, ~46% of panel cost alone)
    #     → openai/gpt-oss-20b
    #   - mistralai/mistral-large-2512 ($0.50/$1.50)
    #     → mistralai/mistral-small-3.2-24b-instruct (not a reasoning/
    #       thinking-toggle model, so no risk from this panel's
    #       ``reasoning: {enabled: false}`` request being unsupported)
    # Since each panel call's output is a single label word, INPUT price
    # dominates cost, not output price — the panel's average input price
    # drops from ~$0.44/M to ~$0.17/M (~63% lower) from these two swaps
    # alone. Generator (deepseek) remains intentionally excluded to avoid
    # self-evaluation bias.
    #
    # Caveat carried over from the original panel selection: avoid gemini
    # flash-lite specifically for judging (empirically unreliable — bare
    # unexplained "NO" votes / empty responses from exhausted reasoning
    # budget; plain "flash" has not shown this). gpt-oss uses OpenAI's own
    # reasoning-effort semantics, which may not honor this panel's
    # ``reasoning: {enabled: false}`` the same way Qwen3/Llama do — if it
    # produces unreliable votes, the existing majority-of-4 threshold
    # degrades gracefully (worst case, that member's vote just doesn't
    # count) rather than corrupting results, but swap it for
    # qwen/qwen3-32b (same reasoning-disable path already proven via
    # qwen3.7-plus) if it turns out to misbehave.
    panel_models: str = Field(
        default="google/gemini-2.5-flash,meta-llama/llama-3.3-70b-instruct,qwen/qwen3.7-plus,mistralai/mistral-small-3.2-24b-instruct,openai/gpt-oss-20b",
        description=(
            "Comma-separated list of 5 distinct-lab panel models for majority "
            "voting. Generator (deepseek) intentionally excluded."
        ),
    )
    panel_temperature: float = Field(
        default=0.0,
        description="Temperature for panel evaluation (0.0 = deterministic)",
    )
    acceptance_threshold: int = Field(
        default=4,
        description=(
            "Minimum votes out of 5 to accept an item (≥4/5 = accept) — "
            "MUST move in lockstep with panel size. A threshold at or above "
            "the panel size means no label can ever be accepted."
        ),
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
