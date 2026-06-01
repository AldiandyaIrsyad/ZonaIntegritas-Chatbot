"""
Configuration management for the application.

Defines Pydantic settings models for all components and provides cached getters.
"""
from functools import lru_cache

from pydantic import SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from dotenv import load_dotenv

# This allow .env to have variables
load_dotenv(override=True) 

class AppSettings(BaseSettings):
    """General application configuration."""
    title: str = "Chat Application with PDF Knowledge Base"
    description: str = "An intelligent chat system that leverages Large Language Models (LLMs) and PDF document management to provide context-aware responses. Users can upload PDF documents which are processed and indexed for semantic search, enabling the LLM to reference relevant content in its responses."
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        extra="ignore"
    )
    
@lru_cache
def get_app_settings() -> AppSettings:
    """Get cached AppSettings.
    
    Returns:
        AppSettings: The cached app settings instance.
    """
    return AppSettings()

class LLMSettings(BaseSettings):
    """Settings for the LLM integration via OpenRouter."""
    use_local: bool = False
    api_key: SecretStr | None = None

    # Generated after
    base_url: str | None = None
    model: str | None = None 
    
    # LLM_OPENROUTER
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_model: str = "google/gemini-2.5-flash"

    # LLM_OLLAMA
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:latest"


    # common
    max_tokens: int = 240_000
    max_completion_tokens: int = 120_000
    default_headers: dict = {
        "HTTP-Referer": "http://localhost:3000",
        "X-Title": "Local-Dev-App"
    }


    # validating schema
    @model_validator(mode="after")
    def validate_settings(self) -> "LLMSettings":
        """Validate the LLM settings."""
        
        # fallback to defaults based on use_local if not explicitly set
        if self.base_url is None:
            self.base_url = self.ollama_base_url if self.use_local else self.openrouter_base_url  
            
        if self.model is None:
            self.model = self.ollama_model if self.use_local else self.openrouter_model  

        return self

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="LLM_", 
        extra="ignore"
    )

@lru_cache
def get_llm_settings() -> LLMSettings:
    """Get cached LLMSettings.
    
    Returns:
        LLMSettings: The cached LLM settings instance.
    """
    return LLMSettings()

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

@lru_cache
def get_db_settings() -> DatabaseSettings:
    """Get cached DatabaseSettings.
    
    Returns:
        DatabaseSettings: The cached database settings instance.
    """
    return DatabaseSettings()

class StorageSettings(BaseSettings):
    """Settings for local file storage."""
    admin_upload_dir: str = "admin_upload"
    user_upload_dir: str = "user_upload"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="STORAGE_", 
        extra="ignore"
    )

@lru_cache
def get_storage_settings() -> StorageSettings:
    """Get cached StorageSettings.
    
    Returns:
        StorageSettings: The cached storage settings instance.
    """
    return StorageSettings()

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

@lru_cache
def get_qdrant_settings() -> QdrantSettings:
    """Get cached QdrantSettings.
    
    Returns:
        QdrantSettings: The cached Qdrant settings instance.
    """
    return QdrantSettings()

class InfinitySettings(BaseSettings):
    """Settings for the Infinity embedding and reranking service."""
    base_url: str = "http://localhost:7997"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    # Classify models — both are loaded by the same Infinity container.
    nli_model: str = "StevenLimcorn/indo-roberta-indonli"
    prompt_guard_model: str = "meta-llama/Llama-Prompt-Guard-2-86M"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="INFINITY_",
        extra="ignore"
    )

@lru_cache
def get_infinity_settings() -> InfinitySettings:
    """Get cached InfinitySettings.
    
    Returns:
        InfinitySettings: The cached Infinity settings instance.
    """
    return InfinitySettings()

class UnstructuredSettings(BaseSettings):
    """Settings for the Unstructured document parsing service."""
    base_url: str = "http://localhost:8001"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="UNSTRUCTURED_",
        extra="ignore"
    )

@lru_cache
def get_unstructured_settings() -> UnstructuredSettings:
    """Get cached UnstructuredSettings.
    
    Returns:
        UnstructuredSettings: The cached unstructured settings instance.
    """
    return UnstructuredSettings()

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

@lru_cache
def get_vector_settings() -> VectorSettings:
    """Get cached VectorSettings.
    
    Returns:
        VectorSettings: The cached vector settings instance.
    """
    return VectorSettings()

class IVMSettings(BaseSettings):
    """Settings for the Input Validation Module (IVM)."""
    security_threshold: float = 0.75
    similarity_threshold: float = 0.3

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="IVM_",
        extra="ignore"
    )


@lru_cache
def get_ivm_settings() -> IVMSettings:
    """Get cached IVMSettings.
    
    Returns:
        IVMSettings: The cached IVM settings instance.
    """
    return IVMSettings()

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
def get_ram_settings() -> RAMSettings:
    """Get cached RAMSettings.
    
    Returns:
        RAMSettings: The cached RAM settings instance.
    """
    return RAMSettings()