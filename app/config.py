from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── App ──────────────────────────────────────────────────────────
    app_env: str = "development"
    app_port: int = 8000
    debug: bool = False
    api_key: str = "neuromem-dev-key-change-me"

    # ── LLM (Provider-Agnostic) ──────────────────────────────────────
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024

    # ── Redis ────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    working_memory_ttl_seconds: int = 7200
    working_memory_max_turns: int = 12

    # ── PostgreSQL ───────────────────────────────────────────────────
    postgres_dsn: str = "postgresql://neuromem:neuromem@localhost:5432/neuromem"
    episodic_memory_ttl_days: int = 30

    # ── Qdrant ───────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "neuromem_semantic"
    qdrant_api_key: str = ""

    # ── Neo4j (Knowledge Graph) ──────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neuromem123"

    # ── Embeddings ───────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024
    embedding_device: str = "cpu"

    # ── Re-Ranking ───────────────────────────────────────────────────
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ── Cognitive Scoring ────────────────────────────────────────────
    consolidation_decay_threshold: float = 0.2
    surprise_threshold: float = 0.15
    memory_token_budget: int = 2000

    # ── Logging ──────────────────────────────────────────────────────
    log_level: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
