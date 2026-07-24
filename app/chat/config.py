from functools import lru_cache
from typing import Literal
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.thesis.prompts import DEFAULT_SYSTEM_PROMPT_ID
from app.thesis.ivm.judge import (
    DEFAULT_RELEVANCE_JUDGE_PROMPT,
    DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE,
)

class ChatConfig(BaseSettings):
    """Configuration for the Chat module.

    LLM settings read the ``CHAT_LLM_*`` environment variables defined in
    ``.env`` (e.g. ``CHAT_LLM_API_KEY``, ``CHAT_LLM_BASE_URL``,
    ``CHAT_LLM_MODEL``) — matching the ``CHAT_`` prefix every other field in
    this class uses. The ``validation_alias`` for each field maps the
    canonical env var name to the config attribute so that
    ``get_chat_config()`` picks up the OpenRouter credentials instead of
    falling back to the dead Ollama defaults.

    NOTE: prior to this fix, these three fields used a stale ``LLM_*``
    alias (``LLM_API_KEY``, ``LLM_OPENROUTER_BASE_URL``,
    ``LLM_OPENROUTER_MODEL``) left over from before the rest of the class
    was migrated to the ``CHAT_`` prefix — ``.env.example`` and
    ``docs/11-deployment.md`` had already documented ``CHAT_LLM_*`` as the
    real variable names, so any ``.env`` written against the docs silently
    fell back to the dead Ollama defaults instead of erroring. If you have
    an existing ``.env`` with ``LLM_API_KEY``/``LLM_OPENROUTER_BASE_URL``/
    ``LLM_OPENROUTER_MODEL``, rename them to ``CHAT_LLM_API_KEY``/
    ``CHAT_LLM_BASE_URL``/``CHAT_LLM_MODEL``.
    """

    # LLM Settings
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="OpenAI-compatible base URL",
        validation_alias="CHAT_LLM_BASE_URL",
    )
    llm_api_key: SecretStr = Field(
        default=SecretStr("dummy"),
        description="API key for LLM",
        validation_alias="CHAT_LLM_API_KEY",
    )
    llm_model: str = Field(
        default="llama3.1:8b",
        description="Model name to use for generation",
        validation_alias="CHAT_LLM_MODEL",
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

    # --- Safety backend selection (IVM) ---
    # Mirrors ``ood_method``: one env var swaps the ISafetyModel adapter so the
    # off-the-shelf classifier, its Indonesian fine-tune, and a hosted
    # generative guard can all be compared without a code change.
    safety_backend: Literal["prompt_guard", "qwen3guard"] = Field(
        default="prompt_guard",
        description=(
            "IVM safety backend. 'prompt_guard' = local sequence classifier "
            "(base or fine-tuned, chosen via prompt_guard_model); 'qwen3guard' = "
            "hosted generative guard over an OpenAI-compatible API."
        ),
        validation_alias="CHAT_SAFETY_BACKEND",
    )
    prompt_guard_url: str = Field(
        default="http://localhost:7998",
        description=(
            "Base URL of the dedicated prompt-guard server. Separate from "
            "``infinity_url`` so the guard can be swapped or restarted without "
            "reloading the reranker and NLI models."
        ),
        validation_alias="PROMPT_GUARD_BASE_URL",
    )
    safety_api_base_url: str = Field(
        default="https://router.huggingface.co/v1",
        description="OpenAI-compatible base URL for the hosted safety backend",
        validation_alias="CHAT_SAFETY_API_BASE_URL",
    )
    safety_api_key: str = Field(
        default="",
        description="API key for the hosted safety backend",
        validation_alias="CHAT_SAFETY_API_KEY",
    )
    safety_api_model: str = Field(
        default="Qwen/Qwen3Guard-Gen-0.6B:featherless-ai",
        description="Model identifier for the hosted safety backend",
        validation_alias="CHAT_SAFETY_API_MODEL",
    )
    safety_controversial_is_unsafe: bool = Field(
        default=True,
        description=(
            "Qwen3Guard emits three tiers (Safe/Controversial/Unsafe). True maps "
            "Controversial to unsafe, matching the fail-closed posture of the rest "
            "of the IVM. Fix this before running an experiment — changing it "
            "afterwards turns the label boundary into a tuned parameter."
        ),
        validation_alias="CHAT_SAFETY_CONTROVERSIAL_IS_UNSAFE",
    )
    relevance_judge_prompt: str = Field(
        default=DEFAULT_RELEVANCE_JUDGE_PROMPT,
        description="System prompt for the IVM LLM-as-judge relevance checker",
        validation_alias="CHAT_RELEVANCE_JUDGE_PROMPT",
    )
    relevance_judge_user_template: str = Field(
        default=DEFAULT_RELEVANCE_JUDGE_USER_TEMPLATE,
        description="User-turn template for the IVM LLM-as-judge relevance checker. {context} and {query} are replaced.",
        validation_alias="CHAT_RELEVANCE_JUDGE_USER_TEMPLATE",
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

    # Chat PDF Attachments
    attachment_max_file_size_mb: int = Field(
        default=15,
        description="Max upload size (MB) for a chat-attached PDF",
        validation_alias="CHAT_ATTACHMENT_MAX_FILE_SIZE_MB",
    )
    attachment_max_pages: int = Field(
        default=40,
        description="Max page count for a chat-attached PDF",
        validation_alias="CHAT_ATTACHMENT_MAX_PAGES",
    )
    attachment_max_chars: int = Field(
        default=50_000,
        description="Max extracted character count kept from a chat-attached PDF (truncated beyond this)",
        validation_alias="CHAT_ATTACHMENT_MAX_CHARS",
    )
    attachment_search_excerpt_chars: int = Field(
        default=4_000,
        description=(
            "Max characters of attachment text folded into the KB relevance/"
            "retrieval search query (kept short so it doesn't degrade "
            "embedding/HyDE quality — full text still goes to the LLM prompt)"
        ),
        validation_alias="CHAT_ATTACHMENT_SEARCH_EXCERPT_CHARS",
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
            "dalam BAB, Pasal, dan Ayat. Berikut daftar dokumen yang tersedia "
            "dalam basis pengetahuan ini:\n{kb_context}\n\n"
            "Tulis paragraf singkat yang meniru gaya dan istilah dokumen resmi "
            "tersebut, dengan mengacu pada dokumen-dokumen di atas jika relevan "
            "— walau pertanyaannya singkat atau tidak lazim, anggaplah "
            "pertanyaan itu mungkin merujuk pada pihak, entitas, atau ketentuan "
            "yang disebutkan di salah satu dokumen di atas. Namun, jika "
            "pertanyaan jelas tidak berkaitan dengan topik dokumen manapun di "
            "atas, balas HANYA dengan string kosong — jangan mengarang jawaban. "
            "Jangan khawatir soal akurasi faktual untuk jawaban yang tetap "
            "ditulis; tujuannya murni kecocokan semantik untuk pencarian."
        ),
        description="System prompt for HyDE hypothetical document generation — sets the domain/register the model should imitate. May contain a {kb_context} placeholder, filled with the active KB document titles/descriptions.",
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
    hyde_context_enabled: bool = Field(
        default=True,
        description="Ground HyDE with the KB's actual active document titles/descriptions ({kb_context} in hyde_system_prompt), so short/OOD queries don't get a fabricated hypothetical document.",
        validation_alias="CHAT_HYDE_CONTEXT_ENABLED",
    )
    hyde_context_max_docs: int = Field(
        default=20,
        description="Max number of active KB documents listed in the {kb_context} grounding block.",
        validation_alias="CHAT_HYDE_CONTEXT_MAX_DOCS",
    )
    hyde_context_refresh_seconds: int = Field(
        default=300,
        description="TTL (seconds) for the cached {kb_context} grounding block before it's refetched from the KB.",
        validation_alias="CHAT_HYDE_CONTEXT_REFRESH_SECONDS",
    )

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

@lru_cache
def get_chat_config() -> ChatConfig:
    """Returns the cached ChatConfig singleton.

    Returns:
        ChatConfig: The cached settings instance.
    """
    return ChatConfig()
