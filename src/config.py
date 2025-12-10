from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # LLM Configuration
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    llm_provider: Literal["openai", "anthropic", "ollama"] = "anthropic"
    llm_model: str = "claude-3-5-sonnet-20241022"

    # Server
    host: str = "0.0.0.0"
    port: int = 3025

    # Tailscale
    tailscale_ip: str = ""

    # Chess AI Service
    chess_ai_url: str = "http://localhost:3020"

    # Whisper
    whisper_model: str = "base"

    # Video Processing
    frame_sample_rate: int = 1  # frames per second

    # Logging
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
