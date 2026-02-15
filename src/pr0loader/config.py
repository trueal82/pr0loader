"""Configuration management using Pydantic Settings."""

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Authentication
    pp: str = Field(..., description="PP cookie for authentication")
    me: str = Field(..., description="ME cookie for authentication")

    # Storage
    filesystem_prefix: Path = Field(
        default=Path("./media"),
        description="Directory to store downloaded media files"
    )
    db_path: Path = Field(
        default=Path("./pr0loader.db"),
        description="Path to SQLite database"
    )

    # API settings
    content_flags: int = Field(default=15, description="Content flags filter")
    api_base_url: str = Field(
        default="https://pr0gramm.com/api",
        description="Base URL for pr0gramm API"
    )
    media_base_url: str = Field(
        default="https://img.pr0gramm.com",
        description="Base URL for media files"
    )

    # Rate limiting
    max_retries: int = Field(default=100, description="Maximum retry attempts")
    request_timeout: int = Field(default=30, description="Request timeout in seconds")
    max_backoff_seconds: int = Field(default=300, description="Maximum backoff time")
    request_delay: float = Field(default=1.0, description="Delay between requests")

    # Processing
    full_update: bool = Field(default=False, description="Perform full update")
    start_from: Optional[int] = Field(default=None, description="Start from specific ID")

    # Dataset preparation
    min_valid_tags: int = Field(default=5, description="Minimum valid tags per item")
    output_dir: Path = Field(default=Path("./output"), description="Output directory for datasets")

    # Training
    batch_size: int = Field(default=128, description="Training batch size")
    num_epochs: int = Field(default=5, description="Number of training epochs")
    learning_rate: float = Field(default=0.0001, description="Learning rate")
    image_size: tuple[int, int] = Field(default=(224, 224), description="Input image size")
    model_path: Path = Field(default=Path("./models/trained_model.keras"), description="Model save path")
    checkpoint_dir: Path = Field(default=Path("./checkpoints"), description="Checkpoint directory")

    # Development
    dev_mode: bool = Field(default=False, description="Development mode (limits data)")
    dev_limit: int = Field(default=100, description="Limit items in dev mode")

    @field_validator("filesystem_prefix", "db_path", "output_dir", "model_path", "checkpoint_dir", mode="before")
    @classmethod
    def expand_path(cls, v):
        """Expand user home directory in paths."""
        if isinstance(v, str):
            return Path(v).expanduser()
        return v


def load_settings() -> Settings:
    """Load settings from environment and .env file."""
    return Settings()

