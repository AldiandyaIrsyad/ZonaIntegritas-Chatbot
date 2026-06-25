from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class QdrantSettings(BaseSettings):
    host: str = Field(default="127.0.0.1")
    port: int = Field(default=6333)
    grpc_port: int = Field(default=6334)
    collection_name: str = Field(default="knowledge_base")
    model_config = SettingsConfigDict(env_file=".env", env_prefix="QDRANT_", extra="ignore")

@lru_cache
def get_qdrant_settings() -> QdrantSettings: return QdrantSettings()


class UnstructuredSettings(BaseSettings):
    base_url: str = Field(default="http://localhost:8001")
    port: int = Field(default=8001)
    extract_images: bool = Field(
        default=True,
        description="Whether to extract image/figure elements during PDF parsing.",
    )
    model_config = SettingsConfigDict(env_file=".env", env_prefix="UNSTRUCTURED_", extra="ignore")

@lru_cache
def get_unstructured_settings() -> UnstructuredSettings: return UnstructuredSettings()

class KBStorageSettings(BaseSettings):
    upload_dir: str = Field(default="./uploads/knowledge_base")
    image_dir: str = Field(
        default="./uploads/knowledge_base/images",
        description="Directory for extracted page/region images.",
    )
    model_config = SettingsConfigDict(env_file=".env", env_prefix="STORAGE_KB_", extra="ignore")

@lru_cache
def get_storage_settings() -> KBStorageSettings: return KBStorageSettings()

# KB only cares about Infinity's embedding and reranking features
class KBInfinitySettings(BaseSettings):
    base_url: str = Field(default="http://127.0.0.1:7997")
    embedding_model: str = Field(default="BAAI/bge-m3")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3")
    reranker_enabled: bool = Field(default=True, description="Toggle reranking of retrieved documents.")
    model_config = SettingsConfigDict(env_file=".env", env_prefix="INFINITY_", extra="ignore")

@lru_cache
def get_infinity_settings() -> KBInfinitySettings: return KBInfinitySettings()


class VLMSettings(BaseSettings):
    """Configuration for Vision-Language Model enrichment of visual elements.

    Supports three modes:
    - ``cloud``: Use a cloud VLM API (OpenRouter: Gemini, GPT-4o).
    - ``local``: Use a local VLM via Ollama (LLaVA, Qwen-VL).
    - ``fallback``: No VLM — use PyMuPDF drawing analysis for text-only
      heuristic descriptions of figures.

    When VLM enrichment fails or is disabled, figure elements are either
    described heuristically (fallback mode) or skipped (text remains empty
    and the element is filtered out by the chunker).
    """

    mode: str = Field(
        default="fallback",
        description="VLM provider mode: 'cloud', 'local', or 'fallback'.",
    )
    # Cloud (OpenRouter) settings
    cloud_api_key: str = Field(default="", description="OpenRouter API key for cloud VLM.")
    cloud_model: str = Field(
        default="google/gemini-2.0-flash-001",
        description="Cloud VLM model identifier.",
    )
    cloud_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL.",
    )
    # Local (Ollama) settings
    local_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL for local VLM.",
    )
    local_model: str = Field(
        default="llava:13b",
        description="Local VLM model name in Ollama.",
    )
    # Behaviour
    timeout: float = Field(
        default=120.0,
        description="Timeout in seconds for VLM API calls.",
    )
    enabled: bool = Field(
        default=True,
        description="Master toggle. If False, figures are skipped (no enrichment).",
    )

    model_config = SettingsConfigDict(env_file=".env", env_prefix="VLM_", extra="ignore")


@lru_cache
def get_vlm_settings() -> VLMSettings:
    """Returns the cached VLMSettings singleton.

    Returns:
        VLMSettings: The cached settings instance.
    """
    return VLMSettings()
