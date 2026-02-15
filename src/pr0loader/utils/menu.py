"""Interactive menu for pr0loader CLI."""

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich import box

console = Console()


def is_interactive() -> bool:
    """Check if we're running in an interactive terminal."""
    return sys.stdin.isatty() and sys.stdout.isatty()


def detect_terminal_capabilities() -> dict:
    """Detect terminal capabilities for screen/tmux compatibility."""
    import os

    term = os.environ.get("TERM", "")
    is_screen = "screen" in term or os.environ.get("STY") is not None
    is_tmux = "tmux" in term or os.environ.get("TMUX") is not None
    is_dumb = term == "dumb" or term == ""

    # Force ASCII-safe output for limited terminals
    force_ascii = is_dumb or os.environ.get("PR0LOADER_ASCII", "").lower() in ("1", "true")

    return {
        "is_screen": is_screen,
        "is_tmux": is_tmux,
        "is_dumb": is_dumb,
        "force_ascii": force_ascii,
        "term": term,
    }


def create_menu_console() -> Console:
    """Create a console configured for the current terminal."""
    caps = detect_terminal_capabilities()

    # For screen/tmux, we still support colors but may need to be careful with some features
    # Rich handles this well, but we can force simpler output if needed
    return Console(
        force_terminal=True if (caps["is_screen"] or caps["is_tmux"]) else None,
        no_color=caps["is_dumb"],
        legacy_windows=False,
    )


class InteractiveMenu:
    """Interactive menu for selecting pipeline options."""

    def __init__(self):
        self.console = create_menu_console()
        self.caps = detect_terminal_capabilities()

    def show_header(self):
        """Show the menu header."""
        header = """
[bold magenta]              ___  _                 _           
 _ __  _ __  / _ \\| | ___   __ _  __| | ___ _ __ 
| '_ \\| '__|/ | | | |/ _ \\ / _` |/ _` |/ _ \\ '__|
| |_) | |  / |_| | | (_) | (_| | (_| |  __/ |   
| .__/|_|  \\___/|_|\\___/ \\__,_|\\__,_|\\___|_|   
|_|[/bold magenta]
"""
        if not self.caps["force_ascii"]:
            self.console.print(header)
        else:
            self.console.print("\n=== pr0loader ===\n", style="bold")

        self.console.print("[dim]Interactive Mode - Use arrow keys or type to select[/dim]\n")

        # Show terminal info in screen/tmux
        if self.caps["is_screen"] or self.caps["is_tmux"]:
            session_type = "screen" if self.caps["is_screen"] else "tmux"
            self.console.print(f"[dim]Running in {session_type} session[/dim]\n")

    def show_main_menu(self) -> Optional[str]:
        """Show the main menu and return selected action."""

        menu_items = [
            ("1", "sync", "🔄 Sync", "Fetch metadata and download assets"),
            ("2", "fetch", "📥 Fetch", "Fetch metadata only"),
            ("3", "download", "📁 Download", "Download media files"),
            ("4", "prepare", "📊 Prepare", "Prepare dataset for training"),
            ("5", "train", "🧠 Train", "Train tag prediction model"),
            ("6", "predict", "🔮 Predict", "Predict tags for images"),
            ("7", "info", "📊 Info", "Show system information"),
            ("8", "run-all", "🚀 Run All", "Run complete pipeline"),
            ("q", "quit", "❌ Quit", "Exit pr0loader"),
        ]

        table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
        table.add_column("Key", style="bold cyan", width=4)
        table.add_column("Action", style="bold")
        table.add_column("Description", style="dim")

        for key, _, name, desc in menu_items:
            table.add_row(f"[{key}]", name, desc)

        self.console.print(Panel(table, title="[bold]Main Menu[/bold]", border_style="blue"))

        choice = Prompt.ask(
            "\n[bold cyan]Select an option[/bold cyan]",
            choices=[item[0] for item in menu_items],
            default="1"
        )

        for key, action, _, _ in menu_items:
            if choice == key:
                return action

        return None

    def configure_sync(self) -> dict:
        """Configure sync options interactively."""
        self.console.print("\n[bold]Configure Sync Options[/bold]\n")

        options = {}

        # Full sync?
        options["full"] = Confirm.ask(
            "Perform full sync (re-fetch all metadata)?",
            default=False
        )

        # Start from specific ID?
        if not options["full"]:
            if Confirm.ask("Start from a specific ID?", default=False):
                options["start_from"] = IntPrompt.ask("Enter starting ID")
            else:
                options["start_from"] = None
        else:
            options["start_from"] = None

        # Include videos?
        options["include_videos"] = Confirm.ask(
            "Include videos? (default: images only)",
            default=False
        )

        # Verify existing files?
        options["verify"] = Confirm.ask(
            "Verify existing files with HEAD request?",
            default=True
        )

        # Metadata only?
        options["metadata_only"] = Confirm.ask(
            "Metadata only (skip downloads)?",
            default=False
        )

        # Show summary
        self._show_options_summary("Sync", options)

        if not Confirm.ask("\nProceed with these options?", default=True):
            return self.configure_sync()

        return options

    def configure_download(self) -> dict:
        """Configure download options interactively."""
        self.console.print("\n[bold]Configure Download Options[/bold]\n")

        options = {}

        options["include_videos"] = Confirm.ask(
            "Include videos? (default: images only)",
            default=False
        )

        self._show_options_summary("Download", options)

        if not Confirm.ask("\nProceed with these options?", default=True):
            return self.configure_download()

        return options

    def configure_fetch(self) -> dict:
        """Configure fetch options interactively."""
        self.console.print("\n[bold]Configure Fetch Options[/bold]\n")

        options = {}

        options["full"] = Confirm.ask(
            "Perform full update (re-fetch all)?",
            default=False
        )

        if not options["full"]:
            if Confirm.ask("Start from a specific ID?", default=False):
                options["start_from"] = IntPrompt.ask("Enter starting ID")
            else:
                options["start_from"] = None
        else:
            options["start_from"] = None

        self._show_options_summary("Fetch", options)

        if not Confirm.ask("\nProceed with these options?", default=True):
            return self.configure_fetch()

        return options

    def configure_prepare(self) -> dict:
        """Configure prepare options interactively."""
        self.console.print("\n[bold]Configure Prepare Options[/bold]\n")

        options = {}

        options["min_tags"] = IntPrompt.ask(
            "Minimum valid tags per item",
            default=5
        )

        if Confirm.ask("Specify custom output file?", default=False):
            options["output"] = Prompt.ask("Output file path")
        else:
            options["output"] = None

        self._show_options_summary("Prepare", options)

        if not Confirm.ask("\nProceed with these options?", default=True):
            return self.configure_prepare()

        return options

    def configure_train(self) -> dict:
        """Configure train options interactively."""
        self.console.print("\n[bold]Configure Train Options[/bold]\n")

        options = {}

        # Find available datasets
        from pr0loader.config import load_settings
        settings = load_settings()
        datasets = sorted(settings.output_dir.glob("*_dataset.csv"), reverse=True)

        if datasets:
            self.console.print("[dim]Available datasets:[/dim]")
            for i, ds in enumerate(datasets[:5], 1):
                self.console.print(f"  {i}. {ds.name}")

            if Confirm.ask("\nUse most recent dataset?", default=True):
                options["dataset"] = str(datasets[0])
            else:
                options["dataset"] = Prompt.ask("Enter dataset path")
        else:
            options["dataset"] = Prompt.ask("Enter dataset path")

        options["epochs"] = IntPrompt.ask("Number of epochs", default=5)
        options["batch_size"] = IntPrompt.ask("Batch size", default=128)
        options["dev_mode"] = Confirm.ask("Development mode (use subset)?", default=False)

        if Confirm.ask("Specify custom output model path?", default=False):
            options["output"] = Prompt.ask("Output model path")
        else:
            options["output"] = None

        self._show_options_summary("Train", options)

        if not Confirm.ask("\nProceed with these options?", default=True):
            return self.configure_train()

        return options

    def configure_predict(self) -> dict:
        """Configure predict options interactively."""
        self.console.print("\n[bold]Configure Predict Options[/bold]\n")

        options = {}

        # Get image paths
        image_paths = []
        while True:
            path = Prompt.ask("Enter image path (or 'done' to finish)")
            if path.lower() == 'done':
                break
            if Path(path).exists():
                image_paths.append(path)
                self.console.print(f"  [green]✓[/green] Added: {path}")
            else:
                self.console.print(f"  [red]✗[/red] File not found: {path}")

        if not image_paths:
            self.console.print("[red]No images specified![/red]")
            return self.configure_predict()

        options["images"] = image_paths
        options["top_k"] = IntPrompt.ask("Number of top tags to show", default=5)
        options["json_output"] = Confirm.ask("Output as JSON?", default=False)

        if Confirm.ask("Use custom model?", default=False):
            options["model"] = Prompt.ask("Model path")
        else:
            options["model"] = None

        self._show_options_summary("Predict", options)

        if not Confirm.ask("\nProceed with these options?", default=True):
            return self.configure_predict()

        return options

    def configure_run_all(self) -> dict:
        """Configure run-all options interactively."""
        self.console.print("\n[bold]Configure Run All Options[/bold]\n")

        options = {}

        self.console.print("[dim]Select which steps to run:[/dim]")
        options["skip_fetch"] = not Confirm.ask("  Run fetch step?", default=True)
        options["skip_download"] = not Confirm.ask("  Run download step?", default=True)
        options["skip_prepare"] = not Confirm.ask("  Run prepare step?", default=True)
        options["skip_train"] = not Confirm.ask("  Run train step?", default=True)

        options["include_videos"] = Confirm.ask(
            "Include videos in download?",
            default=False
        )

        options["dev_mode"] = Confirm.ask(
            "Development mode (use subset)?",
            default=False
        )

        self._show_options_summary("Run All", options)

        if not Confirm.ask("\nProceed with these options?", default=True):
            return self.configure_run_all()

        return options

    def _show_options_summary(self, action: str, options: dict):
        """Show a summary of selected options."""
        self.console.print(f"\n[bold]Selected {action} Options:[/bold]")

        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Option", style="cyan")
        table.add_column("Value", style="green")

        for key, value in options.items():
            if value is None:
                display = "[dim]default[/dim]"
            elif isinstance(value, bool):
                display = "Yes" if value else "No"
            elif isinstance(value, list):
                display = f"{len(value)} item(s)"
            else:
                display = str(value)

            table.add_row(key.replace("_", " ").title(), display)

        self.console.print(table)


def run_interactive_menu() -> tuple[Optional[str], dict]:
    """
    Run the interactive menu.

    Returns:
        Tuple of (action, options) or (None, {}) if quit.
    """
    if not is_interactive():
        return None, {}

    menu = InteractiveMenu()
    menu.show_header()

    action = menu.show_main_menu()

    if action == "quit" or action is None:
        return None, {}

    # Configure options based on action
    options = {}

    if action == "sync":
        options = menu.configure_sync()
    elif action == "download":
        options = menu.configure_download()
    elif action == "fetch":
        options = menu.configure_fetch()
    elif action == "prepare":
        options = menu.configure_prepare()
    elif action == "train":
        options = menu.configure_train()
    elif action == "predict":
        options = menu.configure_predict()
    elif action == "run-all":
        options = menu.configure_run_all()
    # info doesn't need configuration

    return action, options

