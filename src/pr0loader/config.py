"""Configuration management using Pydantic Settings."""

import os
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_default_data_dir() -> Path:
    """Get the default data directory based on OS conventions."""
    if os.name == "nt":  # Windows
        # Use %LOCALAPPDATA%\pr0loader or fallback to %USERPROFILE%\.pr0loader
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "pr0loader"
        return Path.home() / ".pr0loader"
    else:  # Linux/macOS
        # Follow XDG Base Directory Specification
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        if xdg_data_home:
            return Path(xdg_data_home) / "pr0loader"
        return Path.home() / ".local" / "share" / "pr0loader"


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base data directory - all variable files go under here
    data_dir: Path = Field(
        default_factory=get_default_data_dir,
        description="Base directory for all pr0loader data (db, media, auth, models, etc.)"
    )

    # Authentication (optional - can now use auto-login)
    pp: str = Field(default="", description="PP cookie for authentication (optional if using login)")
    me: str = Field(default="", description="ME cookie for authentication (optional if using login)")

    # Storage - paths relative to data_dir (will be resolved in validator)
    filesystem_prefix: Optional[Path] = Field(
        default=None,
        description="Directory to store downloaded media files (default: {data_dir}/media)"
    )
    db_path: Optional[Path] = Field(
        default=None,
        description="Path to SQLite database (default: {data_dir}/pr0loader.db)"
    )
    output_dir: Optional[Path] = Field(
        default=None,
        description="Output directory for datasets (default: {data_dir}/output)"
    )
    model_path: Optional[Path] = Field(
        default=None,
        description="Model save path (default: {data_dir}/models/trained_model.keras)"
    )
    checkpoint_dir: Optional[Path] = Field(
        default=None,
        description="Checkpoint directory (default: {data_dir}/checkpoints)"
    )
    auth_dir: Optional[Path] = Field(
        default=None,
        description="Auth/credentials directory (default: {data_dir}/auth)"
    )

    # API settings
    content_flags: int = Field(default=15, description="Content flags filter (15=all)")
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

    # Training
    batch_size: int = Field(default=128, description="Training batch size")
    num_epochs: int = Field(default=5, description="Number of training epochs")
    learning_rate: float = Field(default=0.0001, description="Learning rate")
    image_size: tuple[int, int] = Field(default=(224, 224), description="Input image size")

    # Development
    dev_mode: bool = Field(default=False, description="Development mode (limits data)")
    dev_limit: int = Field(default=100, description="Limit items in dev mode")

    @field_validator("data_dir", mode="before")
    @classmethod
    def expand_data_dir(cls, v):
        """Expand user home directory in data_dir."""
        if isinstance(v, str):
            return Path(v).expanduser()
        return v

    @model_validator(mode="after")
    def resolve_paths(self):
        """Resolve all paths relative to data_dir if not explicitly set."""
        data_dir = self.data_dir

        # Ensure data_dir is expanded
        if isinstance(data_dir, str):
            data_dir = Path(data_dir).expanduser()

        # Set default paths relative to data_dir
        if self.filesystem_prefix is None:
            self.filesystem_prefix = data_dir / "media"
        elif isinstance(self.filesystem_prefix, str):
            self.filesystem_prefix = Path(self.filesystem_prefix).expanduser()

        if self.db_path is None:
            self.db_path = data_dir / "pr0loader.db"
        elif isinstance(self.db_path, str):
            self.db_path = Path(self.db_path).expanduser()

        if self.output_dir is None:
            self.output_dir = data_dir / "output"
        elif isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir).expanduser()

        if self.model_path is None:
            self.model_path = data_dir / "models" / "trained_model.keras"
        elif isinstance(self.model_path, str):
            self.model_path = Path(self.model_path).expanduser()

        if self.checkpoint_dir is None:
            self.checkpoint_dir = data_dir / "checkpoints"
        elif isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir).expanduser()

        if self.auth_dir is None:
            self.auth_dir = data_dir / "auth"
        elif isinstance(self.auth_dir, str):
            self.auth_dir = Path(self.auth_dir).expanduser()

        return self

    def ensure_directories(self):
        """Create all required directories if they don't exist."""
        dirs = [
            self.data_dir,
            self.filesystem_prefix,
            self.output_dir,
            self.checkpoint_dir,
            self.auth_dir,
            self.model_path.parent,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Load settings from environment and .env file."""
    settings = Settings()
    return settings

