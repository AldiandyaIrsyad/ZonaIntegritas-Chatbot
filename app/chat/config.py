from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    system_prompt: str = Field(default="You are a helpful AI assistant answering questions based on provided knowledge base context.", description="Default system prompt")

    # Safety/Relevance (IVM)
    infinity_url: str = Field(default="http://localhost:7997", description="Infinity server URL", validation_alias="INFINITY_BASE_URL")
    prompt_guard_model: str = Field(default="meta-llama/Llama-Prompt-Guard-2-86M", validation_alias="INFINITY_PROMPT_GUARD_MODEL")
    nli_model: str = Field(default="StevenLimcorn/indo-roberta-indonli", validation_alias="INFINITY_NLI_MODEL")
    security_threshold: float = Field(default=0.75, description="Threshold for prompt injection detection")
    similarity_threshold: float = Field(default=0.4, description="Threshold for KB search relevance")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

@lru_cache
def get_chat_config() -> ChatConfig:
    """Returns the cached ChatConfig singleton.

    Returns:
        ChatConfig: The cached settings instance.
    """
    return ChatConfig()
