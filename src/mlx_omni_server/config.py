import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class Settings(BaseSettings):
    """Central configuration for MLX Omni Server"""

    # Cache settings
    cache_directory: str = "./caches"

    # Server settings
    host: str = "0.0.0.0"
    port: int = 10240
    workers: int = 1
    log_level: str = "info"

    # Model settings
    default_model: Optional[str] = None
    model_cache_dir: Optional[str] = None

    class Config:
        env_prefix = "MLX_OMNI_"
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Create global settings instance
settings = Settings()

# Ensure cache directory exists
cache_path = Path(settings.cache_directory)
cache_path.mkdir(parents=True, exist_ok=True)
