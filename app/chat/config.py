from functools import lru_cache
from typing import Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.thesis.prompts import DEFAULT_SYSTEM_PROMPT_ID
from app.thesis.ivm.judge import DEFAULT_RELEVANCE_JUDGE_PROMPT

class ChatConfig(BaseSettings):
    """Configuration for the Chat module.

    LLM settings read the ``LLM_*`` environment variables defined in ``.env``
    (e.g. ``LLM_API_KEY``, ``LLM_OPENROUTER_BASE_URL``, ``LLM_OPENROUTER_MODEL``).
    The ``validation_alias`` for each field maps the canonical env var name to
    the config attribute so that ``get_chat_config()`` picks up the OpenRouter
    credentials instead of falling back to the dead Ollama defaults.
    """

    # LLM Settings
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible base URL",
        validation_alias="LLM_OPENROUTER_BASE_URL",
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr("dummy"),
        description="API key for LLM",
        validation_alias="LLM_API_KEY",
    )
    llm_model: str = Field(
        default="llama3.1:8b",
        description="Model name to use for generation",
        validation_alias="LLM_OPENROUTER_MODEL",
    )
    llm_temperature: float = Field(
        default=0.0,
        description="Sampling temperature for generation (0.0 = deterministic, per skripsi §3.2.1c)",
    )
    system_prompt: str = Field(
        default=DEFAULT_SYSTEM_PROMPT_ID,
        description="Default system prompt (Bahasa Indonesia)",
        validation_alias="CHAT_SYSTEM_PROMPT",
    )

    # Safety/Relevance (IVM)
    infinity_url: str = Field(default="http://localhost:7997", description="Infinity server URL", validation_alias="INFINITY_BASE_URL")
    prompt_guard_model: str = Field(default="meta-llama/Llama-Prompt-Guard-2-86M", validation_alias="INFINITY_PROMPT_GUARD_MODEL")
    nli_model: str = Field(default="StevenLimcorn/indo-roberta-indonli", validation_alias="INFINITY_NLI_MODEL")
    security_threshold: float = Field(default=0.75, description="Threshold for prompt injection detection")
    relevance_judge_prompt: str = Field(
        default=DEFAULT_RELEVANCE_JUDGE_PROMPT,
        description="System prompt for the IVM LLM-as-judge relevance checker",
        validation_alias="CHAT_RELEVANCE_JUDGE_PROMPT",
    )
    ood_method: Literal["llm_judge", "similarity_threshold", "nli_entailment"] = Field(
        default="llm_judge",
        description="IVM relevance/OOD backend (see app/thesis/ivm/checkers.py)",
        validation_alias="CHAT_OOD_METHOD",
    )
    ood_similarity_threshold: float = Field(
        default=0.02,
        description=(
            "Min top-1 retrieval (RRF fusion) score for the 'similarity_threshold' "
            "OOD method — placeholder, calibrate empirically against this KB's own "
            "score distribution"
        ),
        validation_alias="CHAT_OOD_SIMILARITY_THRESHOLD",
    )
    ood_nli_entailment_threshold: float = Field(
        default=0.5,
        description="Min NLI entailment_score for the 'nli_entailment' OOD method",
        validation_alias="CHAT_OOD_NLI_THRESHOLD",
    )

    # HyDE (Hypothetical Document Embeddings) Settings
    hyde_enabled: bool = Field(
        default=True,
        description="Enable HyDE: generate a hypothetical answer doc and embed that instead of the raw query",
        validation_alias="CHAT_HYDE_ENABLED",
    )
    hyde_max_tokens: int = Field(
        default=256,
        description="Max tokens for the hypothetical document generation",
        validation_alias="CHAT_HYDE_MAX_TOKENS",
    )
    hyde_temperature: float = Field(
        default=0.0,
        description="Temperature for HyDE generation (0.0 = deterministic)",
        validation_alias="CHAT_HYDE_TEMPERATURE",
    )
    hyde_system_prompt: str = Field(
        default=(
            "Anda menghasilkan draf jawaban hipotetis untuk mendukung pencarian "
            "semantik pada basis pengetahuan dokumen resmi institusi berbahasa "
            "Indonesia (SOP, peraturan, kontrak, keputusan) yang terstruktur "
            "dalam BAB, Pasal, dan Ayat. Tulis paragraf singkat yang meniru gaya "
            "dan istilah dokumen resmi tersebut — walau pertanyaannya singkat, "
            "tidak lazim, atau tampak di luar topik, anggaplah pertanyaan itu "
            "mungkin merujuk pada pihak, entitas, atau ketentuan yang disebutkan "
            "di salah satu dokumen. Jangan khawatir soal akurasi faktual; "
            "tujuannya murni kecocokan semantik untuk pencarian."
        ),
        description="System prompt for HyDE hypothetical document generation — sets the domain/register the model should imitate.",
        validation_alias="CHAT_HYDE_SYSTEM_PROMPT",
    )
    hyde_prompt_template: str = Field(
        default=(
            "Pertanyaan: {query}\n\n"
            "Tulis draf jawaban hipotetis (2-4 kalimat) dengan gaya dokumen "
            "resmi seperti dijelaskan di atas."
        ),
        description="User-turn template for HyDE hypothetical document generation. {query} is replaced with the user query.",
        validation_alias="CHAT_HYDE_PROMPT_TEMPLATE",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

@lru_cache
def get_chat_config() -> ChatConfig:
    """Returns the cached ChatConfig singleton.

    Returns:
        ChatConfig: The cached settings instance.
    """
    return ChatConfig()
