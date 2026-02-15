"""Unified UI components for pr0loader interactive mode."""

import os
import sys
from typing import Optional, List, Tuple, Any, Callable
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich import box
from rich.live import Live
from rich.layout import Layout
from rich.rule import Rule


# Global console instance
console = Console()


def clear_screen():
    """Clear the terminal screen."""
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")
    # Also use ANSI escape as fallback
    console.print("\033[2J\033[H", end="")


def get_status_info() -> dict:
    """Gather system status information."""
    status = {}

    try:
        from pr0loader.config import load_settings
        settings = load_settings()

        status["dev_mode"] = settings.dev_mode
        status["dev_limit"] = settings.dev_limit
        status["content_flags"] = settings.content_flags
        status["data_dir"] = str(settings.data_dir)

        # Database stats
        if settings.db_path.exists():
            try:
                from pr0loader.storage import SQLiteStorage
                with SQLiteStorage(settings.db_path) as storage:
                    status["db_items"] = storage.get_item_count()
            except Exception:
                status["db_items"] = 0
        else:
            status["db_items"] = 0

        # Model status
        status["model_ready"] = settings.model_path.exists()

        # Auth status
        try:
            from pr0loader.auth import get_auth_manager
            auth = get_auth_manager()
            creds = auth.store.load()
            if creds and creds.is_valid():
                status["auth_status"] = "Logged in"
                status["auth_user"] = creds.username or ""
            else:
                status["auth_status"] = "Not logged in"
        except Exception:
            status["auth_status"] = "Unknown"

        # GPU status
        try:
            import torch
            if torch.cuda.is_available():
                status["gpu_available"] = True
                status["gpu_name"] = torch.cuda.get_device_name(0)
                if len(status["gpu_name"]) > 20:
                    status["gpu_name"] = status["gpu_name"][:17] + "..."
            else:
                status["gpu_available"] = False
        except ImportError:
            status["gpu_available"] = None
        except Exception:
            status["gpu_available"] = False

    except Exception:
        pass

    return status


def render_status_panel(status_info: dict, width: int = 28) -> Panel:
    """Render the status info panel."""
    lines = []

    # Auth status
    auth_user = status_info.get("auth_user", "")
    if auth_user:
        lines.append(Text(f"🔐 {auth_user}", style="green"))
    else:
        auth_status = status_info.get("auth_status", "Unknown")
        lines.append(Text(f"🔐 {auth_status}", style="yellow" if auth_status == "Not logged in" else "green"))

    # Dev mode
    dev_mode = status_info.get("dev_mode", False)
    if dev_mode:
        dev_limit = status_info.get("dev_limit", 1000)
        lines.append(Text(f"🧪 Dev Mode: {dev_limit}", style="yellow"))
    else:
        lines.append(Text("🧪 Dev Mode: OFF", style="dim"))

    # Content flags
    content_flags = status_info.get("content_flags", 15)
    flags_desc = []
    if content_flags & 1: flags_desc.append("SFW")
    if content_flags & 2: flags_desc.append("NSFW")
    if content_flags & 4: flags_desc.append("NSFL")
    if content_flags & 8: flags_desc.append("POL")
    lines.append(Text(f"🎭 {'+'.join(flags_desc)}", style="cyan"))

    # Database stats
    db_items = status_info.get("db_items", 0)
    if db_items > 0:
        lines.append(Text(f"📊 {db_items:,} items", style="green"))
    else:
        lines.append(Text("📊 No database", style="dim"))

    # GPU status
    gpu_available = status_info.get("gpu_available", None)
    if gpu_available is True:
        gpu_name = status_info.get("gpu_name", "GPU")
        lines.append(Text(f"🎮 {gpu_name}", style="green"))
    elif gpu_available is False:
        lines.append(Text("🎮 CPU only", style="dim"))

    # Model status
    model_ready = status_info.get("model_ready", False)
    if model_ready:
        lines.append(Text("🤖 Model ready", style="green"))
    else:
        lines.append(Text("🤖 No model", style="dim"))

    content = "\n".join(str(line) for line in lines)
    return Panel(
        content,
        title="[bold]Status[/bold]",
        border_style="dim",
        width=width,
    )


def render_header(title: str, subtitle: str = "") -> Panel:
    """Render a page header."""
    content = f"[bold cyan]{title}[/bold cyan]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"

    return Panel(
        Align.center(content),
        box=box.DOUBLE,
        border_style="blue",
        padding=(0, 2),
    )


def render_page(
    title: str,
    content: Any,
    subtitle: str = "",
    show_status: bool = True,
    footer: str = "",
) -> None:
    """Render a full page with header, content, optional status sidebar, and footer."""
    clear_screen()

    # Header
    console.print(render_header(title, subtitle))
    console.print()

    # Main content with optional status sidebar
    if show_status:
        status_info = get_status_info()
        status_panel = render_status_panel(status_info)

        # Create side-by-side layout
        layout_table = Table.grid(padding=(0, 2))
        layout_table.add_column("content", ratio=3)
        layout_table.add_column("status", ratio=1)
        layout_table.add_row(content, status_panel)
        console.print(layout_table)
    else:
        console.print(content)

    # Footer
    if footer:
        console.print()
        console.print(f"[dim]{footer}[/dim]")


class DialogBuilder:
    """Builder for creating consistent dialog forms."""

    def __init__(self, title: str, subtitle: str = ""):
        self.title = title
        self.subtitle = subtitle
        self.fields: List[dict] = []
        self.values: dict = {}

    def add_confirm(
        self,
        key: str,
        label: str,
        default: bool = False,
        description: str = "",
    ) -> "DialogBuilder":
        """Add a yes/no confirmation field."""
        self.fields.append({
            "type": "confirm",
            "key": key,
            "label": label,
            "default": default,
            "description": description,
        })
        return self

    def add_text(
        self,
        key: str,
        label: str,
        default: str = "",
        description: str = "",
        choices: Optional[List[str]] = None,
    ) -> "DialogBuilder":
        """Add a text input field."""
        self.fields.append({
            "type": "text",
            "key": key,
            "label": label,
            "default": default,
            "description": description,
            "choices": choices,
        })
        return self

    def add_number(
        self,
        key: str,
        label: str,
        default: int = 0,
        description: str = "",
        min_value: Optional[int] = None,
        max_value: Optional[int] = None,
    ) -> "DialogBuilder":
        """Add a number input field."""
        self.fields.append({
            "type": "number",
            "key": key,
            "label": label,
            "default": default,
            "description": description,
            "min_value": min_value,
            "max_value": max_value,
        })
        return self

    def add_path(
        self,
        key: str,
        label: str,
        default: str = "",
        description: str = "",
        must_exist: bool = False,
    ) -> "DialogBuilder":
        """Add a file/directory path input field."""
        self.fields.append({
            "type": "path",
            "key": key,
            "label": label,
            "default": default,
            "description": description,
            "must_exist": must_exist,
        })
        return self

    def add_separator(self, label: str = "") -> "DialogBuilder":
        """Add a visual separator."""
        self.fields.append({
            "type": "separator",
            "label": label,
        })
        return self

    def add_info(self, text: str, style: str = "dim") -> "DialogBuilder":
        """Add informational text."""
        self.fields.append({
            "type": "info",
            "text": text,
            "style": style,
        })
        return self

    def run(self, show_summary: bool = True) -> Optional[dict]:
        """Run the dialog and return collected values, or None if cancelled."""
        clear_screen()

        # Render header
        console.print(render_header(self.title, self.subtitle))
        console.print()

        # Create form panel
        form_content = []

        for field in self.fields:
            if field["type"] == "separator":
                if field.get("label"):
                    console.print(Rule(field["label"], style="dim"))
                else:
                    console.print()
                continue

            if field["type"] == "info":
                console.print(f"[{field['style']}]{field['text']}[/{field['style']}]")
                continue

            # Show description if present
            if field.get("description"):
                console.print(f"[dim]{field['description']}[/dim]")

            try:
                if field["type"] == "confirm":
                    value = Confirm.ask(
                        f"[cyan]{field['label']}[/cyan]",
                        default=field["default"]
                    )
                    self.values[field["key"]] = value

                elif field["type"] == "text":
                    if field.get("choices"):
                        value = Prompt.ask(
                            f"[cyan]{field['label']}[/cyan]",
                            choices=field["choices"],
                            default=field["default"] or field["choices"][0]
                        )
                    else:
                        value = Prompt.ask(
                            f"[cyan]{field['label']}[/cyan]",
                            default=field["default"]
                        )
                    self.values[field["key"]] = value

                elif field["type"] == "number":
                    value = IntPrompt.ask(
                        f"[cyan]{field['label']}[/cyan]",
                        default=field["default"]
                    )
                    # Validate range
                    if field.get("min_value") is not None and value < field["min_value"]:
                        value = field["min_value"]
                    if field.get("max_value") is not None and value > field["max_value"]:
                        value = field["max_value"]
                    self.values[field["key"]] = value

                elif field["type"] == "path":
                    value = Prompt.ask(
                        f"[cyan]{field['label']}[/cyan]",
                        default=field["default"]
                    )
                    path = Path(value).expanduser()
                    if field.get("must_exist") and not path.exists():
                        console.print(f"[yellow]Warning: Path does not exist: {path}[/yellow]")
                    self.values[field["key"]] = str(path)

            except KeyboardInterrupt:
                console.print("\n[yellow]Cancelled[/yellow]")
                return None

        # Show summary
        if show_summary and self.values:
            console.print()
            self._show_summary()

            if not Confirm.ask("\n[bold]Proceed with these settings?[/bold]", default=True):
                return None

        return self.values

    def _show_summary(self):
        """Show a summary of collected values."""
        table = Table(title="Selected Options", box=box.ROUNDED)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        for key, value in self.values.items():
            if isinstance(value, bool):
                display = "✓ Yes" if value else "✗ No"
            elif isinstance(value, (list, tuple)):
                display = ", ".join(str(v) for v in value)
            else:
                display = str(value)

            # Pretty-print the key
            label = key.replace("_", " ").title()
            table.add_row(label, display)

        console.print(table)


class OptionSelector:
    """Multi-select option dialog."""

    def __init__(self, title: str, options: List[Tuple[str, str, bool]]):
        """
        Args:
            title: Dialog title
            options: List of (key, label, default_selected)
        """
        self.title = title
        self.options = options

    def run(self) -> Optional[List[str]]:
        """Run the selector and return list of selected keys."""
        clear_screen()
        console.print(render_header(self.title, "Select options to enable"))
        console.print()

        selected = {}
        for key, label, default in self.options:
            selected[key] = Confirm.ask(f"[cyan]{label}[/cyan]", default=default)

        return [k for k, v in selected.items() if v]


class ActionConfirm:
    """Confirmation dialog for actions."""

    def __init__(
        self,
        title: str,
        message: str,
        details: Optional[List[str]] = None,
        confirm_text: str = "Proceed?",
        warning: bool = False,
    ):
        self.title = title
        self.message = message
        self.details = details or []
        self.confirm_text = confirm_text
        self.warning = warning

    def run(self) -> bool:
        """Show confirmation and return True if confirmed."""
        clear_screen()

        style = "yellow" if self.warning else "cyan"
        console.print(render_header(self.title))
        console.print()

        console.print(f"[{style}]{self.message}[/{style}]")

        if self.details:
            console.print()
            for detail in self.details:
                console.print(f"  • {detail}")

        console.print()
        return Confirm.ask(f"[bold]{self.confirm_text}[/bold]", default=not self.warning)


class ProgressPage:
    """Page for showing progress of long-running operations."""

    def __init__(self, title: str, steps: List[str]):
        self.title = title
        self.steps = steps
        self.current_step = 0

    def start(self):
        """Start the progress display."""
        clear_screen()
        console.print(render_header(self.title, f"Step 0/{len(self.steps)}"))
        console.print()

    def next_step(self, message: str = ""):
        """Move to the next step."""
        self.current_step += 1
        step_name = self.steps[self.current_step - 1] if self.current_step <= len(self.steps) else "Done"

        console.print(f"\n[bold cyan]═══ Step {self.current_step}/{len(self.steps)}: {step_name} ═══[/bold cyan]")
        if message:
            console.print(f"[dim]{message}[/dim]")

    def complete(self, message: str = "All steps completed!"):
        """Mark progress as complete."""
        console.print()
        console.print(Panel(
            f"[bold green]✓ {message}[/bold green]",
            border_style="green",
        ))


def show_message(
    title: str,
    message: str,
    style: str = "info",
    wait_for_key: bool = True,
) -> None:
    """Show a simple message dialog."""
    clear_screen()

    if style == "error":
        border_style = "red"
        icon = "✗"
    elif style == "warning":
        border_style = "yellow"
        icon = "⚠"
    elif style == "success":
        border_style = "green"
        icon = "✓"
    else:
        border_style = "blue"
        icon = "ℹ"

    console.print(Panel(
        f"[bold]{icon} {title}[/bold]\n\n{message}",
        border_style=border_style,
        padding=(1, 2),
    ))

    if wait_for_key:
        console.print()
        Prompt.ask("[dim]Press Enter to continue[/dim]", default="")


def wait_for_key(message: str = "Press Enter to continue"):
    """Wait for user to press a key."""
    console.print()
    Prompt.ask(f"[dim]{message}[/dim]", default="")

