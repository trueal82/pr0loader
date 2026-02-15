"""Interactive menu for pr0loader CLI."""

import sys
from pathlib import Path
from typing import Optional, List, Tuple

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.live import Live
from rich.text import Text
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

    # On Windows, TERM is usually empty but the terminal is not "dumb"
    # Check for Windows Terminal, ConEmu, or standard Windows console
    is_windows = sys.platform == "win32"
    is_windows_terminal = os.environ.get("WT_SESSION") is not None  # Windows Terminal
    is_conemu = os.environ.get("ConEmuANSI") == "ON"

    # Only consider truly dumb terminals (explicit dumb, or empty TERM on non-Windows)
    is_dumb = term == "dumb" or (term == "" and not is_windows)

    # Force ASCII-safe output only if explicitly requested
    force_ascii = os.environ.get("PR0LOADER_ASCII", "").lower() in ("1", "true")

    return {
        "is_screen": is_screen,
        "is_tmux": is_tmux,
        "is_dumb": is_dumb,
        "is_windows": is_windows,
        "is_windows_terminal": is_windows_terminal,
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


def get_key() -> str:
    """
    Get a single keypress from the user.
    Returns the key pressed, or special strings for arrow keys.
    Works with both native Windows input and ANSI escape sequences.
    """
    if sys.platform == "win32":
        import msvcrt
        import time

        key = msvcrt.getch()

        # Native Windows special keys (works in cmd.exe, PowerShell, some terminals)
        if key == b'\xe0' or key == b'\x00':
            key2 = msvcrt.getch()
            if key2 == b'H':
                return 'UP'
            elif key2 == b'P':
                return 'DOWN'
            elif key2 == b'K':
                return 'LEFT'
            elif key2 == b'M':
                return 'RIGHT'
            return ''

        # ESC key or ANSI escape sequence (Windows Terminal, ConEmu, etc.)
        elif key == b'\x1b':
            # Wait a tiny bit for the rest of the escape sequence
            # Windows Terminal sends ANSI sequences but timing can vary
            time.sleep(0.01)  # 10ms wait

            if msvcrt.kbhit():
                key2 = msvcrt.getch()
                if key2 == b'[':
                    # Wait for the final character
                    time.sleep(0.01)
                    if msvcrt.kbhit():
                        key3 = msvcrt.getch()
                        if key3 == b'A':
                            return 'UP'
                        elif key3 == b'B':
                            return 'DOWN'
                        elif key3 == b'C':
                            return 'RIGHT'
                        elif key3 == b'D':
                            return 'LEFT'
                    return ''  # Incomplete sequence
                else:
                    # Not an ANSI sequence, might be Alt+key or just ESC followed by another key
                    # Put it back conceptually by returning ESC
                    return 'ESC'
            return 'ESC'

        elif key == b'\r' or key == b'\n':
            return 'ENTER'
        else:
            try:
                return key.decode('utf-8').lower()
            except:
                return ''
    else:
        import tty
        import termios
        import select

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            key = sys.stdin.read(1)

            if key == '\x1b':  # Escape sequence
                # Use select to check if more data is available with timeout
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key2 = sys.stdin.read(1)
                    if key2 == '[':
                        if select.select([sys.stdin], [], [], 0.1)[0]:
                            key3 = sys.stdin.read(1)
                            if key3 == 'A':
                                return 'UP'
                            elif key3 == 'B':
                                return 'DOWN'
                            elif key3 == 'C':
                                return 'RIGHT'
                            elif key3 == 'D':
                                return 'LEFT'
                return 'ESC'
            elif key == '\r' or key == '\n':
                return 'ENTER'
            else:
                return key.lower()
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class ArrowKeyMenu:
    """A menu navigable with arrow keys."""

    def __init__(
        self,
        items: List[Tuple[str, str, str, str]],  # (key, action, name, description)
        title: str = "Menu",
        console: Optional[Console] = None,
        status_info: Optional[dict] = None,  # Status info to display on the right
    ):
        self.items = items
        self.title = title
        self.console = console or Console()
        self.selected_index = 0
        self.status_info = status_info or {}

    def _render_status_panel(self) -> Panel:
        """Render the status info panel."""
        from rich.text import Text as RichText

        lines = []

        # Auth status
        auth_status = self.status_info.get("auth_status", "Unknown")
        auth_user = self.status_info.get("auth_user", "")
        if auth_user:
            lines.append(RichText(f"🔐 {auth_user}", style="green"))
        else:
            lines.append(RichText(f"🔐 {auth_status}", style="yellow" if auth_status == "Not logged in" else "green"))

        # Dev mode
        dev_mode = self.status_info.get("dev_mode", False)
        if dev_mode:
            lines.append(RichText("🧪 Dev Mode: ON", style="yellow"))
        else:
            lines.append(RichText("🧪 Dev Mode: OFF", style="dim"))

        # Content flags
        content_flags = self.status_info.get("content_flags", 15)
        flags_desc = []
        if content_flags & 1:
            flags_desc.append("SFW")
        if content_flags & 2:
            flags_desc.append("NSFW")
        if content_flags & 4:
            flags_desc.append("NSFL")
        if content_flags & 8:
            flags_desc.append("POL")
        lines.append(RichText(f"🎭 {'+'.join(flags_desc)}", style="cyan"))

        # Database stats
        db_items = self.status_info.get("db_items", 0)
        if db_items > 0:
            lines.append(RichText(f"📊 {db_items:,} items", style="green"))
        else:
            lines.append(RichText("📊 No database", style="dim"))

        # GPU status
        gpu_available = self.status_info.get("gpu_available", None)
        if gpu_available is True:
            gpu_name = self.status_info.get("gpu_name", "GPU")
            lines.append(RichText(f"🎮 {gpu_name}", style="green"))
        elif gpu_available is False:
            lines.append(RichText("🎮 CPU only", style="dim"))

        # Model status
        model_ready = self.status_info.get("model_ready", False)
        if model_ready:
            lines.append(RichText("🤖 Model ready", style="green"))
        else:
            lines.append(RichText("🤖 No model", style="dim"))

        # Data dir
        data_dir = self.status_info.get("data_dir", "")
        if data_dir:
            # Truncate if too long
            if len(data_dir) > 25:
                data_dir = "..." + data_dir[-22:]
            lines.append(RichText(f"📁 {data_dir}", style="dim"))

        content = "\n".join(str(line) for line in lines)
        return Panel(
            "\n".join(str(line) for line in lines),
            title="[bold]Status[/bold]",
            border_style="dim",
            width=30,
        )

    def _render_menu(self) -> Panel:
        """Render the menu with current selection highlighted."""
        from rich.columns import Columns

        # Menu table
        table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
        table.add_column("", width=2)  # Selection indicator
        table.add_column("Key", style="bold cyan", width=4)
        table.add_column("Action", width=20)
        table.add_column("Description")

        for i, (key, _, name, desc) in enumerate(self.items):
            if i == self.selected_index:
                # Highlighted row
                indicator = "▶"
                key_style = "bold white on blue"
                name_style = "bold white on blue"
                desc_style = "white on blue"
            else:
                indicator = " "
                key_style = "bold cyan"
                name_style = "bold"
                desc_style = "dim"

            table.add_row(
                Text(indicator, style="bold cyan"),
                Text(f"[{key}]", style=key_style),
                Text(name, style=name_style),
                Text(desc, style=desc_style),
            )

        # Create menu panel
        menu_panel = Panel(
            table,
            title=f"[bold]{self.title}[/bold]",
            border_style="blue",
            subtitle="[dim]↑↓ Navigate • Enter Select • q Quit[/dim]"
        )

        # If we have status info, create a two-column layout
        if self.status_info:
            from rich.columns import Columns
            from rich.console import Group

            status_panel = self._render_status_panel()

            # Use a table to create side-by-side layout
            layout_table = Table.grid(padding=(0, 1))
            layout_table.add_column("menu", ratio=3)
            layout_table.add_column("status", ratio=1)
            layout_table.add_row(menu_panel, status_panel)

            return layout_table

        return menu_panel

    def run(self) -> Optional[str]:
        """Run the interactive menu and return the selected action."""
        from rich.live import Live

        # Use Live display for proper terminal handling
        with Live(self._render_menu(), console=self.console, refresh_per_second=10, transient=True) as live:
            while True:
                key = get_key()
                should_exit = False

                if key == 'UP':
                    self.selected_index = (self.selected_index - 1) % len(self.items)
                    live.update(self._render_menu())
                elif key == 'DOWN':
                    self.selected_index = (self.selected_index + 1) % len(self.items)
                    live.update(self._render_menu())
                elif key == 'ENTER':
                    # Exit live context and return
                    should_exit = True
                elif key == 'ESC' or key == 'q':
                    # Find quit action
                    for i, (_, action, _, _) in enumerate(self.items):
                        if action == "quit":
                            self.selected_index = i
                            break
                    should_exit = True
                elif key:  # Non-empty key
                    # Check if key matches a shortcut
                    for i, (shortcut, action, _, _) in enumerate(self.items):
                        if key == shortcut.lower():
                            self.selected_index = i
                            should_exit = True
                            break
                    if not should_exit:
                        live.update(self._render_menu())

                if should_exit:
                    break

        # Print final state after Live context exits
        self.console.print(self._render_menu())
        return self.items[self.selected_index][1]  # Return action


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

        self.console.print("[dim]Interactive Mode - Use ↑↓ arrows or type shortcut key[/dim]\n")

        # Show terminal info in screen/tmux
        if self.caps["is_screen"] or self.caps["is_tmux"]:
            session_type = "screen" if self.caps["is_screen"] else "tmux"
            self.console.print(f"[dim]Running in {session_type} session[/dim]\n")

    def _gather_status_info(self) -> dict:
        """Gather system status information for the menu sidebar."""
        status = {}

        try:
            from pr0loader.config import load_settings
            settings = load_settings()

            # Basic config
            status["dev_mode"] = settings.dev_mode
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
                    # Truncate long names
                    if len(status["gpu_name"]) > 20:
                        status["gpu_name"] = status["gpu_name"][:17] + "..."
                else:
                    status["gpu_available"] = False
            except ImportError:
                # PyTorch not installed
                status["gpu_available"] = None
            except Exception:
                status["gpu_available"] = False

        except Exception:
            # If settings can't be loaded, return minimal info
            pass

        return status

    def show_main_menu(self) -> Optional[str]:
        """Show the main menu and return selected action."""

        menu_items = [
            ("1", "sync", "🔄 Sync", "Fetch metadata and download assets"),
            ("2", "fetch", "📥 Fetch", "Fetch metadata only"),
            ("3", "download", "📁 Download", "Download media files"),
            ("4", "prepare", "📊 Prepare", "Prepare dataset for training"),
            ("5", "train", "🧠 Train", "Train tag prediction model"),
            ("6", "predict", "🔮 Predict", "Predict tags for images"),
            ("7", "api", "🌐 API", "Start inference API server"),
            ("8", "ui", "🎨 UI", "Start Gradio web interface"),
            ("9", "serve", "🚀 Serve", "Start both API and UI"),
            ("l", "login", "🔐 Login", "Login to pr0gramm"),
            ("o", "logout", "🚪 Logout", "Clear stored credentials"),
            ("s", "setup", "⚙️ Setup", "Reconfigure pr0loader"),
            ("i", "info", "📊 Info", "Show system information"),
            ("q", "quit", "❌ Quit", "Exit pr0loader"),
        ]

        # Gather status info
        status_info = self._gather_status_info()

        # Use arrow key menu if terminal supports it
        if not self.caps["is_dumb"]:
            menu = ArrowKeyMenu(
                items=menu_items,
                title="Main Menu",
                console=self.console,
                status_info=status_info,
            )
            return menu.run()
        else:
            # Fallback to simple prompt-based menu
            return self._show_simple_menu(menu_items)

    def _show_simple_menu(self, menu_items: List[Tuple[str, str, str, str]]) -> Optional[str]:
        """Fallback simple menu for dumb terminals."""
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

    def configure_api(self) -> dict:
        """Configure API server options interactively."""
        self.console.print("\n[bold]Configure API Server Options[/bold]\n")

        options = {}

        options["host"] = Prompt.ask("Host to bind to", default="0.0.0.0")
        options["port"] = IntPrompt.ask("Port", default=8000)

        if Confirm.ask("Use custom model path?", default=False):
            options["model"] = Prompt.ask("Model path")
        else:
            options["model"] = None

        self._show_options_summary("API Server", options)

        if not Confirm.ask("\nProceed with these options?", default=True):
            return self.configure_api()

        return options

    def configure_ui(self) -> dict:
        """Configure Gradio UI options interactively."""
        self.console.print("\n[bold]Configure Gradio UI Options[/bold]\n")

        options = {}

        # Mode selection
        use_api = Confirm.ask("Connect to remote API server? (No = use local model)", default=False)

        if use_api:
            options["api_url"] = Prompt.ask("API server URL", default="http://localhost:8000")
            options["model"] = None
        else:
            options["api_url"] = None
            if Confirm.ask("Use custom model path?", default=False):
                options["model"] = Prompt.ask("Model path")
            else:
                options["model"] = None

        options["host"] = Prompt.ask("Host to bind to", default="0.0.0.0")
        options["port"] = IntPrompt.ask("Port", default=7860)
        options["share"] = Confirm.ask("Create public Gradio link?", default=False)

        self._show_options_summary("Gradio UI", options)

        if not Confirm.ask("\nProceed with these options?", default=True):
            return self.configure_ui()

        return options

    def configure_serve(self) -> dict:
        """Configure combined server options interactively."""
        self.console.print("\n[bold]Configure Full Server Options[/bold]\n")

        options = {}

        options["host"] = Prompt.ask("Host to bind to", default="0.0.0.0")
        options["api_port"] = IntPrompt.ask("API server port", default=8000)
        options["ui_port"] = IntPrompt.ask("Gradio UI port", default=7860)

        if Confirm.ask("Use custom model path?", default=False):
            options["model"] = Prompt.ask("Model path")
        else:
            options["model"] = None

        self._show_options_summary("Full Server", options)

        if not Confirm.ask("\nProceed with these options?", default=True):
            return self.configure_serve()

        return options

    def configure_login(self) -> dict:
        """Configure login options interactively."""
        self.console.print("\n[bold]Configure Login Options[/bold]\n")

        options = {}

        # Show current status
        try:
            from pr0loader.auth import get_auth_manager
            auth = get_auth_manager()
            status = auth.get_status()

            if status.get("stored_credentials"):
                self.console.print(f"[green]Currently logged in as: {status.get('username')}[/green]\n")

            browsers = status.get("available_browsers", [])
            if browsers:
                self.console.print(f"[dim]Available browsers: {', '.join(browsers)}[/dim]\n")
        except:
            pass

        self.console.print("[dim]Select login method:[/dim]")
        method = Prompt.ask(
            "Method",
            choices=["auto", "browser", "interactive"],
            default="auto"
        )

        options["method"] = method

        if method == "browser":
            browser = Prompt.ask(
                "Which browser",
                choices=["firefox", "chrome", "edge", "brave"],
                default="firefox"
            )
            options["browser"] = browser

        self._show_options_summary("Login", options)

        if not Confirm.ask("\nProceed?", default=True):
            return self.configure_login()

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
    elif action == "api":
        options = menu.configure_api()
    elif action == "ui":
        options = menu.configure_ui()
    elif action == "serve":
        options = menu.configure_serve()
    elif action == "login":
        options = menu.configure_login()
    # info, logout don't need configuration

    return action, options

