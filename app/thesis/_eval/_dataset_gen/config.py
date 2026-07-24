"""Configuration for the dataset generation pipeline.

Skripsi §3.2.1c defines a Generator–Evaluator architecture:
    - Generator: DeepSeek — cheap, strong instruction-following
    - Panel (5 distinct labs, generator excluded to avoid self-eval bias), all
      verified live on OpenRouter: Gemini 2.5 Flash, Llama 3.3 70B, Qwen3.7
      Plus, Nemotron 3 Super, and GPT-5.6-Luna.
    - Temperature: 0.0 (deterministic), with upstream provider pinned — see
      ``panel_allow_fallbacks``, since temperature alone is not enough for
      reproducibility when a slug is served by several providers.
    - Acceptance threshold: ≥4/5 majority vote

Model slugs must resolve on OpenRouter; an invalid slug fails closed (panel
votes NO) and silently rejects every candidate. Run
``python -m app.thesis._eval._dataset_gen.preflight`` before a full run — it
resolves slugs, checks each model votes correctly in both directions, and
reports which upstream provider served each call.

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
    # 5 distinct-lab models (slugs and prices verified live against
    # OpenRouter /models on 2026-07-22 via the preflight script):
    #   google/gemini-2.5-flash              $0.30  / $2.50  per 1M tokens
    #   meta-llama/llama-3.3-70b-instruct    $0.10  / $0.32  per 1M tokens
    #   qwen/qwen3.7-plus                    $0.32  / $1.28  per 1M tokens
    #   nvidia/nemotron-3-super-120b-a12b    $0.08  / $0.45  per 1M tokens
    #   openai/gpt-5.6-luna                  $1.00  / $6.00  per 1M tokens
    # Panel size stays 5 with a majority-of-4 threshold: a 3-model panel was
    # tried and reverted as too volatile, and fewer/weaker raters is a poor
    # trade against cost that is negligible at this scale anyway (the whole
    # A-C regeneration is well under $1).
    #
    # This restores gpt-5.6-luna, which an earlier cost pass had swapped out
    # for openai/gpt-oss-20b, and replaces the Mistral slot with Nemotron 3
    # Super — a 120B MoE (12B active) at essentially the same price as the
    # mistral-small-24b it replaces, so it is a straight capability gain.
    # Blended input price is ~$0.36/M. Since each panel call's output is a
    # single label word, INPUT price is what dominates cost.
    #
    # NOTE ON ORDER: position carries no meaning. EvaluatorPanel.evaluate
    # gathers all five votes concurrently and counts them equally; there is
    # no tie-breaker role in the code, whatever earlier revisions of this
    # comment implied by calling a premium member a "tie-breaker voice".
    #
    # Generator (deepseek) remains intentionally excluded to avoid
    # self-evaluation bias.
    #
    # Caveats carried over: avoid gemini flash-**lite** for judging
    # (empirically unreliable — bare unexplained "NO" votes / empty responses
    # from an exhausted reasoning budget; plain "flash" has not shown this),
    # and avoid ':free' variants, which are separately rate-limited. Any
    # model that rejects this panel's ``reasoning: {enabled: false}`` request
    # errors into a fail-closed NO vote on every call, so run
    # ``python -m app.thesis._eval._dataset_gen.preflight`` before a full
    # generation — it verifies each slug resolves and votes correctly in both
    # directions, which is exactly the failure this comment keeps warning about.
    panel_models: str = Field(
        default="google/gemini-2.5-flash,meta-llama/llama-3.3-70b-instruct,qwen/qwen3.7-plus,nvidia/nemotron-3-super-120b-a12b,openai/gpt-5.6-luna",
        description=(
            "Comma-separated list of 5 distinct-lab panel models for majority "
            "voting. Generator (deepseek) intentionally excluded."
        ),
    )
    panel_provider_order: str = Field(
        default="",
        description=(
            "Optional comma-separated OpenRouter provider preference order for "
            "panel calls (e.g. 'DeepInfra,Together'). Empty = let OpenRouter "
            "choose, but still pin the choice per run via allow_fallbacks."
        ),
    )
    panel_allow_fallbacks: bool = Field(
        default=False,
        description=(
            "Whether OpenRouter may fall back to another upstream provider. "
            "False by default: one slug can be served by several providers at "
            "different quantizations, so temperature=0.0 alone does NOT make "
            "the panel reproducible — a mid-run reroute silently changes the "
            "rater. Set True only if pinning causes availability failures."
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
