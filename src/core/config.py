"""
Configuration management for the application.

Defines Pydantic settings models for all components and provides cached getters.
"""
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    """Settings for the LLM integration via OpenRouter."""
    api_key: SecretStr | None = None
    use_local: bool = False
    base_url: str = "https://openrouter.ai/api/v1"
    model: str = "google/gemini-2.5-flash"
    max_tokens: int = 24000
    max_completion_tokens: int = 12000
    default_headers: dict = {
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Local-Dev-App"
    }

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="LLM_", 
        extra="ignore"
    )

class DatabaseSettings(BaseSettings):
    """Settings for the PostgreSQL database."""
    user: str = "postgres"
    password: str
    db: str = "postgres"
    host: str = "localhost"
    port: str = "5432"

    @property
    def database_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="POSTGRES_", 
        extra="ignore"
    )

class StorageSettings(BaseSettings):
    """Settings for local file storage."""
    admin_upload_dir: str = "admin_upload"
    user_upload_dir: str = "user_upload"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="STORAGE_", 
        extra="ignore"
    )

class QdrantSettings(BaseSettings):
    """Settings for Qdrant vector database."""
    host: str = "localhost"
    port: int = 6333
    collection_name: str = "knowledge_base"
    session_collection_name: str = "session_documents"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="QDRANT_",
        extra="ignore"
    )

class InfinitySettings(BaseSettings):
    """Settings for the Infinity embedding and reranking service."""
    base_url: str = "http://localhost:7997"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Classify models — both are loaded by the same Infinity container.
    nli_model: str = "StevenLimcorn/indo-roberta-indonli"
    prompt_guard_model: str = "ProtectAI/deberta-v3-base-prompt-injection-v2"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="INFINITY_",
        extra="ignore"
    )

class UnstructuredSettings(BaseSettings):
    """Settings for the Unstructured document parsing service."""
    base_url: str = "http://localhost:8001"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="UNSTRUCTURED_",
        extra="ignore"
    )

# Observability timberio/vector:0.47.0-alpine
class VectorSettings(BaseSettings):
    """Settings for the Vector log forwarder."""
    host: str = "localhost"
    port: int = 9000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VECTOR_",
        extra="ignore"
    )

class IVMSettings(BaseSettings):
    """Settings for the Input Validation Module (IVM)."""
    security_threshold: float = 0.75
    similarity_threshold: float = 0.3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="IVM_",
        extra="ignore"
    )

class RAMSettings(BaseSettings):
    """Response Assessment Module (RAM) configuration.

    nli_model: Hugging Face model ID for the NLI pipeline.
        Currently: StevenLimcorn/indo-roberta-indonli
        Future:    LazarusNLP/indobert-lite-base-p1-indonli-distil-mdeberta
    nli_device: -1 = CPU, 0 = first GPU.
    nli_max_length: Token truncation limit for premise + hypothesis.
    nli_enabled: Kill-switch — set RAM_NLI_ENABLED=false to disable without code changes.
    """
    nli_model: str = "StevenLimcorn/indo-roberta-indonli"
    nli_device: int = -1
    nli_max_length: int = 512
    nli_enabled: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAM_",
        extra="ignore"
    )
    
@lru_cache
def get_settings() -> LLMSettings:
    """Get cached LLMSettings.
    
    Returns:
        LLMSettings: The cached LLM settings instance.
    """
    return LLMSettings()

@lru_cache
def get_db_settings() -> DatabaseSettings:
    """Get cached DatabaseSettings.
    
    Returns:
        DatabaseSettings: The cached database settings instance.
    """
    return DatabaseSettings()

@lru_cache
def get_storage_settings() -> StorageSettings:
    """Get cached StorageSettings.
    
    Returns:
        StorageSettings: The cached storage settings instance.
    """
    return StorageSettings()

@lru_cache
def get_qdrant_settings() -> QdrantSettings:
    """Get cached QdrantSettings.
    
    Returns:
        QdrantSettings: The cached Qdrant settings instance.
    """
    return QdrantSettings()

@lru_cache
def get_infinity_settings() -> InfinitySettings:
    """Get cached InfinitySettings.
    
    Returns:
        InfinitySettings: The cached Infinity settings instance.
    """
    return InfinitySettings()

@lru_cache
def get_unstructured_settings() -> UnstructuredSettings:
    """Get cached UnstructuredSettings.
    
    Returns:
        UnstructuredSettings: The cached unstructured settings instance.
    """
    return UnstructuredSettings()

@lru_cache
def get_vector_settings() -> VectorSettings:
    """Get cached VectorSettings.
    
    Returns:
        VectorSettings: The cached vector settings instance.
    """
    return VectorSettings()

@lru_cache
def get_ivm_settings() -> IVMSettings:
    """Get cached IVMSettings.
    
    Returns:
        IVMSettings: The cached IVM settings instance.
    """
    return IVMSettings()

@lru_cache
def get_ram_settings() -> RAMSettings:
    """Get cached RAMSettings.
    
    Returns:
        RAMSettings: The cached RAM settings instance.
    """
    return RAMSettings()