from functools import lru_cache
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class LLMSettings(BaseSettings):
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
    admin_upload_dir: str = "admin_upload"
    user_upload_dir: str = "user_upload"

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_prefix="STORAGE_", 
        extra="ignore"
    )

class QdrantSettings(BaseSettings):
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
    base_url: str = "http://localhost:7997"
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="INFINITY_",
        extra="ignore"
    )

class UnstructuredSettings(BaseSettings):
    base_url: str = "http://localhost:8001"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="UNSTRUCTURED_",
        extra="ignore"
    )

# Observability timberio/vector:0.47.0-alpine
class VectorSettings(BaseSettings):
    host: str = "localhost"
    port: int = 9000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="VECTOR_",
        extra="ignore"
    )
    
@lru_cache
def get_settings() -> LLMSettings:
    return LLMSettings()

@lru_cache
def get_db_settings() -> DatabaseSettings:
    return DatabaseSettings()

@lru_cache
def get_storage_settings() -> StorageSettings:
    return StorageSettings()

@lru_cache
def get_qdrant_settings() -> QdrantSettings:
    return QdrantSettings()

@lru_cache
def get_infinity_settings() -> InfinitySettings:
    return InfinitySettings()

@lru_cache
def get_unstructured_settings() -> UnstructuredSettings:
    return UnstructuredSettings()

@lru_cache
def get_vector_settings() -> VectorSettings:
    return VectorSettings()