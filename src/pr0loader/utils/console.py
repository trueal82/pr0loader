"""Rich-based console output and progress tracking."""

from contextlib import contextmanager
from typing import Optional, Generator

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    MofNCompleteColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich import box

# Global console instance
console = Console()

# Headless mode flag
_headless = False


def set_headless(headless: bool):
    """Set headless mode (disables rich output)."""
    global _headless
    _headless = headless


def is_headless() -> bool:
    """Check if running in headless mode."""
    return _headless


def create_progress(description: str = "Processing") -> Progress:
    """Create a Rich progress bar with consistent styling."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        disable=_headless,
    )


def create_download_progress() -> Progress:
    """Create a progress bar optimized for downloads."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(bar_width=30),
        TaskProgressColumn(),
        TextColumn("[green]{task.fields[size]}"),
        TimeElapsedColumn(),
        console=console,
        disable=_headless,
    )


@contextmanager
def progress_context(description: str = "Processing", total: Optional[int] = None) -> Generator:
    """Context manager for progress tracking."""
    progress = create_progress(description)
    with progress:
        task = progress.add_task(description, total=total)
        yield progress, task


def print_header(title: str, subtitle: Optional[str] = None):
    """Print a styled header."""
    if _headless:
        console.print(f"\n=== {title} ===")
        if subtitle:
            console.print(f"    {subtitle}")
        return

    content = f"[bold magenta]{title}[/bold magenta]"
    if subtitle:
        content += f"\n[dim]{subtitle}[/dim]"

    console.print(Panel(content, box=box.DOUBLE_EDGE, padding=(1, 2)))


def print_stats_table(title: str, stats: dict):
    """Print a statistics table."""
    table = Table(title=title, box=box.ROUNDED if not _headless else None)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    for key, value in stats.items():
        table.add_row(key, str(value))

    console.print(table)


def print_success(message: str):
    """Print a success message."""
    console.print(f"[bold green]✓[/bold green] {message}")


def print_error(message: str):
    """Print an error message."""
    console.print(f"[bold red]✗[/bold red] {message}")


def print_warning(message: str):
    """Print a warning message."""
    console.print(f"[bold yellow]⚠[/bold yellow] {message}")


def print_info(message: str):
    """Print an info message."""
    console.print(f"[bold blue]ℹ[/bold blue] {message}")


def print_step(step: int, total: int, description: str):
    """Print a step indicator."""
    console.print(f"\n[bold cyan]Step {step}/{total}:[/bold cyan] {description}")

