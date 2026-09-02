"""
Application Configuration
Handles environment-based settings using Pydantic BaseSettings.
Follows Design Principle: Separation of Concerns
"""

from pydantic import ConfigDict
from pydantic_settings import BaseSettings
from typing import List
from functools import lru_cache
import logging


class Settings(BaseSettings):
    model_config = ConfigDict(env_file=".env", case_sensitive=False)
    """
    Application settings loaded from environment variables.
    All configuration is externalized for portability and security.
    """

    # ===== APP INFO =====
    app_name: str = "ClarityAI"
    app_version: str = "1.0.0"
    environment: str = "development"  # development, staging, production

    # ===== DATABASE =====
    database_url: str = "sqlite:///./clarityai.db"
    db_user: str = "clarityai"
    db_password: str = "change_me_in_production"
    db_name: str = "clarityai_db"
    db_host: str = "localhost"
    db_port: int = 5432

    # ===== API =====
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    api_version: str = "v1"
    cors_origins: str = "http://localhost:3000,http://localhost"

    # ===== MODEL & ML =====
    model_path: str = "./models/model_v1.pt"
    classical_model_path: str = "./models/classical_model.joblib"

    # ===== LOGGING =====
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"

    # ===== FILE UPLOAD =====
    max_file_size: int = 52428800  # 50MB
    allowed_formats: str = "image/jpeg,image/png,image/webp"
    upload_dir: str = "./temp/uploads"

    # ===== SECURITY =====
    secret_key: str = "your-secret-key-here"

    @property
    def resolved_database_url(self) -> str:
        """Return the active database URL.

        Default to the app's configured database_url, which remains SQLite for local dev
        and tests. In production, set DATABASE_URL explicitly to PostgreSQL.
        """
        if self.database_url:
            return self.database_url

        return (
            f"postgresql+psycopg2://{self.db_user}:{self.db_password}@"
            f"{self.db_host}:{self.db_port}/{self.db_name}"
        )

    @property
    def cors_origins_list(self) -> List[str]:
        """Convert comma-separated origins to list"""
        return [origin.strip() for origin in self.cors_origins.split(",")]

    @property
    def allowed_formats_list(self) -> List[str]:
        """Convert comma-separated formats to list"""
        return [fmt.strip() for fmt in self.allowed_formats.split(",")]

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment.lower() == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment.lower() == "development"


@lru_cache()
def get_settings() -> Settings:
    """
    Singleton pattern for settings.
    Cached to avoid re-loading from environment on every call.
    """
    return Settings()


def setup_logging(settings: Settings) -> logging.Logger:
    """
    Configure logging based on settings.
    Follows Design Principle: Separation of Concerns
    """
    logger = logging.getLogger("clarityai")
    
    # Set level
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger
