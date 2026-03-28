"""Global configuration via pydantic-settings."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """SCA configuration. All fields can be overridden via SCA_* env vars."""

    default_model: str = "claude-sonnet-4-6"
    default_n: int = 20
    default_temperature: float = 0.9
    embedding_model: str = "all-MiniLM-L6-v2"
    min_cluster_size: int = 3
    convergence_window: int = 3
    convergence_threshold: float = 0.01

    model_config = SettingsConfigDict(env_prefix="SCA_")


settings = Settings()
