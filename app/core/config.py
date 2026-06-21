"""Configuration module for the application."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv
from functools import lru_cache

# This allows variable within .env
load_dotenv(override=True) 


class AppSettings(BaseSettings):
    """Application settings.
    
    Attributes:
        title: The title of the application.
        description: Description of the application.
        version: Application version.
        host: Host address to bind the server to.
        port: Port number to bind the server to.
        reload: Whether to enable auto-reload for development.
    """
    title: str = "Chat Application with PDF Knowledge Base"
    description: str = ""
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = Field(default=8000, ge=1, le=65535, description="Port must be between 1 and 65535")
    reload: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
        extra="ignore"
    )
    
@lru_cache
def get_app_settings() -> AppSettings:
    """Retrieves the cached application settings.
    
    Returns:
        AppSettings: The application configuration instance.
    """
    return AppSettings()


class LoggerSettings(BaseSettings):
    """Settings for the Vector log forwarder.
    
    Attributes:
        host: Host address for the Vector log forwarder.
        port: Port number for the Vector log forwarder.
        log_level: Log level for the Vector log forwarder.
    """
    vector_host: str = "localhost"
    vector_port: int = Field(default=9000, ge=1, le=65535, description="Port must be between 1 and 65535")
    log_level: str = Field(default="INFO", description="Log level")


    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LOGGER_",
        extra="ignore"
    )

@lru_cache 
def get_logger_settings() -> LoggerSettings:
    """Retrieves the cached logger settings.
    
    Returns:
        LoggerSettings: The logger configuration instance.
    """
    return LoggerSettings()


class DatabaseSettings(BaseSettings):
    """Database connection settings.
    
    Attributes:
        user: PostgreSQL user.
        password: PostgreSQL password.
        db: PostgreSQL database name.
        port: PostgreSQL port.
        host: PostgreSQL host.
    """
    user: str = Field(default="postgres", description="Database user")
    password: str = Field(default="postgres", description="Database password")
    db: str = Field(default="postgres", description="Database name")
    port: int = Field(default=5432, ge=1, le=65535, description="Database port")
    host: str = Field(default="localhost", description="Database host")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="POSTGRES_",
        extra="ignore"
    )

    @property
    def async_database_url(self) -> str:
        """Constructs the asyncpg connection string."""
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

@lru_cache
def get_db_settings() -> DatabaseSettings:
    """Retrieves the cached database settings.
    
    Returns:
        DatabaseSettings: The database configuration instance.
    """
    return DatabaseSettings()


class UnstructuredSettings(BaseSettings):
    """Settings for the Unstructured API.
    
    Attributes:
        base_url: Base URL for the Unstructured API.
        port: Port for the Unstructured API.
    """
    base_url: str = Field(default="http://localhost:8001", description="Unstructured API base URL")
    port: int = Field(default=8001, ge=1, le=65535, description="Unstructured API port")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="UNSTRUCTURED_",
        extra="ignore"
    )

@lru_cache
def get_unstructured_settings() -> UnstructuredSettings:
    """Retrieves the cached Unstructured API settings.
    
    Returns:
        UnstructuredSettings: The Unstructured API configuration instance.
    """
    return UnstructuredSettings()


class LLMSettings(BaseSettings):
    """LLM provider settings."""
    api_key: SecretStr = Field(default=SecretStr("your_llm_api_key"), description="API Key for OpenRouter")
    use_local: bool = Field(default=False, description="Whether to use local Ollama instance")
    ollama_port: int = Field(default=11434, description="Ollama API port")
    ollama_base_url: str = Field(default="http://127.0.0.1:11434/v1", description="Ollama API base URL")
    ollama_model: str = Field(default="qwen2.5:0.5b", description="Ollama model name")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1", description="OpenRouter base URL")
    openrouter_model: str = Field(default="google/gemini-2.5-flash", description="OpenRouter model name")
    max_tokens: int = Field(default=8192, description="Maximum context window size")
    max_completion_tokens: int = Field(default=1024, description="Tokens reserved for generation output")
    default_headers: dict[str, str] = Field(default_factory=dict, description="Default headers for LLM connection")

    @property
    def base_url(self) -> str:
        """Dynamically resolve the base URL based on local toggle."""
        return self.ollama_base_url if self.use_local else self.openrouter_base_url

    @property
    def model(self) -> str:
        """Dynamically resolve the model based on local toggle."""
        return self.ollama_model if self.use_local else self.openrouter_model

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="LLM_",
        extra="ignore"
    )

@lru_cache
def get_llm_settings() -> LLMSettings:
    """Retrieves the cached LLM settings."""
    return LLMSettings()


class InfinitySettings(BaseSettings):
    """Infinity embeddings and NLI server settings."""
    base_url: str = Field(default="http://127.0.0.1:7997", description="Infinity API base URL")
    port: int = Field(default=7997, description="Infinity API port")
    embedding_model: str = Field(default="BAAI/bge-m3", description="Embedding model name")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", description="Reranker model name")
    prompt_guard_model: str = Field(default="meta-llama/Llama-Prompt-Guard-2-86M", description="Prompt guard model name")
    nli_model: str = Field(default="StevenLimcorn/indo-roberta-indonli", description="NLI model name")
    hf_api_key: SecretStr = Field(default=SecretStr("your_hf_api_key"), description="HuggingFace API Key")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="INFINITY_",
        extra="ignore"
    )

@lru_cache
def get_infinity_settings() -> InfinitySettings:
    """Retrieves the cached Infinity settings."""
    return InfinitySettings()


class QdrantSettings(BaseSettings):
    """Qdrant Vector Database settings."""
    host: str = Field(default="127.0.0.1", description="Qdrant host")
    port: int = Field(default=6333, description="Qdrant REST port")
    grpc_port: int = Field(default=6334, description="Qdrant gRPC port")
    collection_name: str = Field(default="knowledge_base", description="Qdrant collection name")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="QDRANT_",
        extra="ignore"
    )

@lru_cache
def get_qdrant_settings() -> QdrantSettings:
    """Retrieves the cached Qdrant settings."""
    return QdrantSettings()


class StorageSettings(BaseSettings):
    """Storage directory settings."""
    kb_upload_dir: str = Field(default="./uploads/knowledge_base", description="Directory for admin uploaded knowledge base PDFs")
    user_upload_dir: str = Field(default="./uploads/user_chat", description="Directory for user uploaded chat PDFs")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="STORAGE_",
        extra="ignore"
    )

@lru_cache
def get_storage_settings() -> StorageSettings:
    """Retrieves the cached Storage settings."""
    return StorageSettings()


class IVMSettings(BaseSettings):
    """Input Validation Module settings."""
    security_threshold: float = Field(default=0.5, description="Score threshold for injection detection")
    similarity_threshold: float = Field(default=0.5, description="Score threshold for semantic relevance")
    top_k: int = Field(default=3, description="Top K results for k-NN voting in relevance checks")
    relevance_strategy: str = Field(default="silhouette_knn", description="Strategy for relevance evaluation (top_one or silhouette_knn)")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="IVM_",
        extra="ignore"
    )

@lru_cache
def get_ivm_settings() -> IVMSettings:
    """Retrieves the cached IVM settings."""
    return IVMSettings()


class RAMSettings(BaseSettings):
    """Settings for the Response Assessment Module."""
    nli_enabled: bool = Field(default=True, description="Whether to enable NLI checks")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="RAM_",
        extra="ignore"
    )

@lru_cache
def get_ram_settings() -> RAMSettings:
    """Retrieves the cached RAM settings."""
    return RAMSettings()

