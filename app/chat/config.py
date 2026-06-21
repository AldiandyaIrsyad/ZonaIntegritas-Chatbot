from functools import lru_cache
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class ChatConfig(BaseSettings):
    """Configuration for the Chat module."""
    
    # LLM Settings
    llm_base_url: str = Field(default="http://localhost:11434/v1", description="OpenAI-compatible base URL")
    llm_api_key: SecretStr = Field(default=SecretStr("dummy"), description="API key for LLM")
    llm_model: str = Field(default="llama3.1:8b", description="Model name to use for generation")
    system_prompt: str = Field(default="You are a helpful AI assistant answering questions based on provided knowledge base context.", description="Default system prompt")

    # Safety/Relevance (IVM)
    infinity_url: str = Field(default="http://localhost:7997", description="Infinity server URL")
    prompt_guard_model: str = Field(default="meta-llama/Llama-Prompt-Guard-2-86M")
    nli_model: str = Field(default="StevenLimcorn/indo-roberta-indonli")
    security_threshold: float = Field(default=0.75, description="Threshold for prompt injection detection")
    similarity_threshold: float = Field(default=0.4, description="Threshold for KB search relevance")

    model_config = SettingsConfigDict(env_prefix="CHAT_", env_file=".env", extra="ignore")

@lru_cache
def get_chat_config() -> ChatConfig:
    """Returns the cached ChatConfig singleton.

    Returns:
        ChatConfig: The cached settings instance.
    """
    return ChatConfig()
