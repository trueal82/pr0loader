"""Setup wizard for first-time configuration."""

import os
import re
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich import box

console = Console()


def get_default_data_dir() -> Path:
    """Get the default data directory based on OS conventions."""
    if os.name == "nt":  # Windows
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data) / "pr0loader"
        return Path.home() / ".pr0loader"
    else:  # Linux/macOS
        xdg_data_home = os.environ.get("XDG_DATA_HOME")
        if xdg_data_home:
            return Path(xdg_data_home) / "pr0loader"
        return Path.home() / ".local" / "share" / "pr0loader"


def env_file_exists() -> bool:
    """Check if .env file exists in current directory or parent directories."""
    # Check current directory
    if Path(".env").exists():
        return True

    # Check if running from within the project
    current = Path.cwd()
    for _ in range(3):  # Check up to 3 levels up
        env_path = current / ".env"
        if env_path.exists():
            return True
        current = current.parent

    return False


def get_env_path() -> Path:
    """Get the path where .env should be created."""
    # If we're in a directory with pyproject.toml, use that directory
    if Path("pyproject.toml").exists():
        return Path(".env")

    # Otherwise use the default data directory
    data_dir = get_default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / ".env"


def load_existing_env() -> Dict[str, str]:
    """Load existing .env file values as defaults."""
    env_values = {}

    env_path = get_env_path()
    if not env_path.exists():
        return env_values

    try:
        content = env_path.read_text(encoding="utf-8")
        for line in content.splitlines():
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            # Parse KEY = value
            match = re.match(r'^([A-Z_]+)\s*=\s*(.*)$', line)
            if match:
                key, value = match.groups()
                # Remove quotes if present
                value = value.strip().strip('"').strip("'")
                env_values[key] = value
    except Exception:
        pass

    return env_values


class SetupWizard:
    """Interactive setup wizard for pr0loader configuration."""

    def __init__(self):
        self.console = console
        self.config = {}
        self.existing_config = load_existing_env()
        self.is_reconfigure = bool(self.existing_config)
        self.run_init = False
        self.run_login = False

    def _get_default(self, key: str, fallback: Any) -> str:
        """Get default value from existing config or fallback."""
        if key in self.existing_config:
            return self.existing_config[key]
        return str(fallback)

    def show_welcome(self):
        """Show welcome message."""
        self.console.print()

        if self.is_reconfigure:
            self.console.print(Panel(
                "[bold cyan]pr0loader Setup[/bold cyan]\n\n"
                "Reconfiguring pr0loader. Current values will be shown as defaults.\n"
                "Press Enter to keep existing values, or type new ones.",
                title="⚙️ Reconfigure",
                border_style="blue",
            ))
        else:
            self.console.print(Panel(
                "[bold cyan]Welcome to pr0loader![/bold cyan]\n\n"
                "This wizard will help you set up your configuration.\n"
                "You can change these settings later by editing the .env file.",
                title="🚀 First-Time Setup",
                border_style="blue",
            ))
        self.console.print()

    def ask_data_directory(self) -> Path:
        """Ask for the data directory location."""
        default = self._get_default("DATA_DIR", get_default_data_dir())

        self.console.print("[bold]📁 Data Directory[/bold]")
        self.console.print("[dim]All pr0loader data (database, media, models) will be stored here.[/dim]")
        self.console.print()

        # Show what will be created
        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("", style="dim")
        table.add_column("")
        table.add_row("Database:", "{DATA_DIR}/pr0loader.db")
        table.add_row("Media:", "{DATA_DIR}/media/")
        table.add_row("Models:", "{DATA_DIR}/models/")
        table.add_row("Output:", "{DATA_DIR}/output/")
        self.console.print(table)
        self.console.print()

        path_str = Prompt.ask(
            "Data directory",
            default=default
        )

        path = Path(path_str).expanduser()
        self.config["DATA_DIR"] = path
        return path

    def ask_authentication(self) -> dict:
        """Ask about authentication preferences."""
        self.console.print()
        self.console.print("[bold]🔐 Authentication[/bold]")
        self.console.print("[dim]pr0loader needs to authenticate with pr0gramm to access content.[/dim]")
        self.console.print()

        # Check if cookies already exist
        existing_pp = self.existing_config.get("PP", "")
        existing_me = self.existing_config.get("ME", "")
        has_existing = bool(existing_pp and existing_me)

        if has_existing:
            self.console.print("[green]✓ Cookies already configured[/green]")
            if not Confirm.ask("Reconfigure authentication?", default=False):
                self.config["PP"] = existing_pp
                self.config["ME"] = existing_me
                return {"PP": existing_pp, "ME": existing_me}

        self.console.print("Options:")
        self.console.print("  1. [cyan]Auto-login[/cyan] - Extract cookies from browser after setup")
        self.console.print("  2. [cyan]Manual[/cyan] - Enter cookies from browser developer tools")
        self.console.print("  3. [cyan]Later[/cyan] - Skip for now, use 'pr0loader login' later")
        self.console.print()

        choice = Prompt.ask(
            "Authentication method",
            choices=["1", "2", "3"],
            default="1"
        )

        auth_config = {}

        if choice == "1":
            # Will run login --auto after setup
            self.run_login = True
        elif choice == "2":
            self.console.print()
            self.console.print("[dim]To get cookies: Open pr0gramm.com → F12 → Application → Cookies[/dim]")
            self.console.print()

            pp = Prompt.ask("PP cookie value", default=existing_pp or "")
            me = Prompt.ask("ME cookie value", default=existing_me or "")

            if pp:
                auth_config["PP"] = pp
            if me:
                auth_config["ME"] = me

        self.config.update(auth_config)
        return auth_config

    def ask_content_flags(self) -> int:
        """Ask about content preferences."""
        self.console.print()
        self.console.print("[bold]🎭 Content Preferences[/bold]")
        self.console.print("[dim]What type of content do you want to download?[/dim]")
        self.console.print()

        self.console.print("Content flags are a bitmask:")
        self.console.print("  [green]1[/green]  - SFW (Safe for Work)")
        self.console.print("  [yellow]2[/yellow]  - NSFW (Not Safe for Work)")
        self.console.print("  [red]4[/red]  - NSFL (Not Safe for Life)")
        self.console.print("  [magenta]8[/magenta]  - NSFP (Not Safe for Public / Political)")
        self.console.print("  [bold]15[/bold] - All content")
        self.console.print()

        default_flags = self._get_default("CONTENT_FLAGS", "15")

        flags_str = Prompt.ask(
            "Content flags",
            default=default_flags
        )

        try:
            flags = int(flags_str)
            if not 1 <= flags <= 15:
                flags = 15
        except ValueError:
            flags = 15

        self.config["CONTENT_FLAGS"] = flags
        return flags

    def ask_advanced_settings(self) -> dict:
        """Ask about advanced settings (optional)."""
        self.console.print()

        if not Confirm.ask("Configure advanced settings?", default=False):
            # Use existing or defaults
            self.config["BATCH_SIZE"] = int(self._get_default("BATCH_SIZE", 128))
            self.config["NUM_EPOCHS"] = int(self._get_default("NUM_EPOCHS", 5))
            self.config["MIN_VALID_TAGS"] = int(self._get_default("MIN_VALID_TAGS", 5))
            return {}

        advanced = {}

        self.console.print()
        self.console.print("[bold]⚙️ Advanced Settings[/bold]")
        self.console.print()

        # Training settings
        batch_size = Prompt.ask(
            "Training batch size",
            default=self._get_default("BATCH_SIZE", "128")
        )
        try:
            advanced["BATCH_SIZE"] = int(batch_size)
        except ValueError:
            advanced["BATCH_SIZE"] = 128

        epochs = Prompt.ask(
            "Training epochs",
            default=self._get_default("NUM_EPOCHS", "5")
        )
        try:
            advanced["NUM_EPOCHS"] = int(epochs)
        except ValueError:
            advanced["NUM_EPOCHS"] = 5

        min_tags = Prompt.ask(
            "Minimum tags per item for dataset",
            default=self._get_default("MIN_VALID_TAGS", "5")
        )
        try:
            advanced["MIN_VALID_TAGS"] = int(min_tags)
        except ValueError:
            advanced["MIN_VALID_TAGS"] = 5

        # Database batch size
        self.console.print()
        self.console.print("[dim]Database batch size: number of items to batch before committing.[/dim]")
        self.console.print("[dim]Higher values = faster (especially on HDDs), but use more RAM.[/dim]")
        db_batch = Prompt.ask(
            "Database batch size",
            default=self._get_default("DB_BATCH_SIZE", "200")
        )
        try:
            advanced["DB_BATCH_SIZE"] = int(db_batch)
        except ValueError:
            advanced["DB_BATCH_SIZE"] = 200

        self.config.update(advanced)
        return advanced

    def ask_post_setup_actions(self):
        """Ask about running init and login after setup."""
        self.console.print()
        self.console.print("[bold]🚀 Post-Setup Actions[/bold]")
        self.console.print()

        # Ask about init
        self.run_init = Confirm.ask(
            "Create data directories now? (pr0loader init)",
            default=True
        )

        # Ask about login if not already set
        if not self.run_login and "PP" not in self.config:
            self.run_login = Confirm.ask(
                "Attempt auto-login from browser? (pr0loader login --auto)",
                default=True
            )

    def generate_env_content(self) -> str:
        """Generate the .env file content."""
        lines = [
            "# pr0loader Configuration",
            "# Generated by setup wizard",
            "#",
            "# Edit this file to change settings, or run 'pr0loader setup' again.",
            "",
            "# ===========================================",
            "# DATA DIRECTORY",
            f"DATA_DIR = {self.config.get('DATA_DIR', get_default_data_dir())}",
            "",
        ]

        # Authentication
        if "PP" in self.config or "ME" in self.config:
            lines.extend([
                "# ===========================================",
                "# AUTHENTICATION",
            ])
            if "PP" in self.config:
                lines.append(f"PP = {self.config['PP']}")
            if "ME" in self.config:
                lines.append(f"ME = {self.config['ME']}")
            lines.append("")

        # Content flags
        lines.extend([
            "# ===========================================",
            "# CONTENT SETTINGS",
            f"CONTENT_FLAGS = {self.config.get('CONTENT_FLAGS', 15)}",
            "",
        ])

        # Training settings
        lines.extend([
            "# ===========================================",
            "# TRAINING SETTINGS",
            f"BATCH_SIZE = {self.config.get('BATCH_SIZE', 128)}",
            f"NUM_EPOCHS = {self.config.get('NUM_EPOCHS', 5)}",
            f"MIN_VALID_TAGS = {self.config.get('MIN_VALID_TAGS', 5)}",
            "",
        ])

        # Performance settings
        lines.extend([
            "# ===========================================",
            "# PERFORMANCE SETTINGS",
            f"DB_BATCH_SIZE = {self.config.get('DB_BATCH_SIZE', 200)}  # Items to batch before DB commit (higher = faster)",
            "",
        ])

        # Development mode
        lines.extend([
            "# ===========================================",
            "# DEVELOPMENT MODE",
            f"DEV_MODE = {self._get_default('DEV_MODE', 'false').lower()}",
            f"DEV_LIMIT = {self._get_default('DEV_LIMIT', '100')}",
            "",
        ])

        return "\n".join(lines)

    def save_env_file(self, path: Path) -> bool:
        """Save the .env file."""
        try:
            content = self.generate_env_content()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return True
        except Exception as e:
            self.console.print(f"[red]Error saving .env file: {e}[/red]")
            return False

    def show_summary(self, env_path: Path):
        """Show configuration summary."""
        self.console.print()
        self.console.print(Panel(
            "[bold green]Configuration saved![/bold green]\n\n"
            f"Settings file: [cyan]{env_path}[/cyan]\n"
            f"Data directory: [cyan]{self.config.get('DATA_DIR', get_default_data_dir())}[/cyan]",
            title="✅ Setup Complete",
            border_style="green",
        ))

    def run_post_actions(self) -> bool:
        """Run post-setup actions (init, login)."""
        success = True

        if self.run_init:
            self.console.print()
            self.console.print("[cyan]Creating directories...[/cyan]")
            try:
                from pr0loader.config import load_settings
                settings = load_settings()
                settings.ensure_directories()
                self.console.print("[green]✓ Directories created[/green]")
            except Exception as e:
                self.console.print(f"[red]✗ Failed to create directories: {e}[/red]")
                success = False

        if self.run_login:
            self.console.print()
            self.console.print("[cyan]Attempting browser login...[/cyan]")
            try:
                from pr0loader.auth import get_auth_manager
                auth = get_auth_manager()
                creds = auth.get_credentials(auto_login=True)
                if creds:
                    self.console.print(f"[green]✓ Logged in as: {creds.username}[/green]")
                else:
                    self.console.print("[yellow]✗ Could not extract credentials from browser[/yellow]")
                    self.console.print("[dim]  Run 'pr0loader login --interactive' to login manually[/dim]")
            except Exception as e:
                self.console.print(f"[yellow]✗ Login failed: {e}[/yellow]")
                self.console.print("[dim]  Run 'pr0loader login' to try again[/dim]")

        return success

    def run(self) -> Optional[Path]:
        """Run the setup wizard."""
        self.show_welcome()

        # Ask for settings
        self.ask_data_directory()
        self.ask_authentication()
        self.ask_content_flags()
        self.ask_advanced_settings()
        self.ask_post_setup_actions()

        # Confirm
        self.console.print()
        if not Confirm.ask("Save configuration?", default=True):
            self.console.print("[yellow]Setup cancelled.[/yellow]")
            return None

        # Save
        env_path = get_env_path()
        if self.save_env_file(env_path):
            self.show_summary(env_path)
            self.run_post_actions()

            # Final message
            self.console.print()
            self.console.print("[bold]Ready to go![/bold] Run [cyan]pr0loader sync[/cyan] to start downloading.")
            self.console.print()

            return env_path

        return None


def run_setup_wizard() -> Optional[Path]:
    """Run the setup wizard and return the path to the created .env file."""
    wizard = SetupWizard()
    return wizard.run()


def check_and_run_setup() -> bool:
    """
    Check if setup is needed and run wizard if so.

    Returns:
        True if setup was run (or not needed), False if user cancelled.
    """
    if env_file_exists():
        return True

    console.print()
    console.print("[yellow]No configuration file found.[/yellow]")

    if Confirm.ask("Run setup wizard?", default=True):
        result = run_setup_wizard()
        return result is not None
    else:
        console.print()
        console.print("[dim]You can run setup later with: pr0loader setup[/dim]")
        console.print("[dim]Or create a .env file manually from template.env[/dim]")
        return False

