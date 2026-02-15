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
        self.status_info = status_info or {}

        # Find first selectable item (non-separator)
        self.selected_index = 0
        for i, (key, _, _, _) in enumerate(items):
            if key:  # Non-empty key means selectable
                self.selected_index = i
                break

    def _is_separator(self, index: int) -> bool:
        """Check if an item at index is a separator (empty key)."""
        return not self.items[index][0]

    def _find_next_selectable(self, current: int, direction: int) -> int:
        """Find the next selectable item in the given direction."""
        n = len(self.items)
        next_idx = (current + direction) % n

        # Loop until we find a selectable item or return to start
        attempts = 0
        while self._is_separator(next_idx) and attempts < n:
            next_idx = (next_idx + direction) % n
            attempts += 1

        return next_idx

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
        table.add_column("Key", style="bold cyan", width=6)
        table.add_column("Action", width=22)
        table.add_column("Description")

        for i, (key, action, name, desc) in enumerate(self.items):
            # Check if this is a separator
            if not key:
                # Render separator row
                table.add_row("", "", Text(name, style="dim"), "")
                continue

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

            # Show arrow for submenu items
            display_key = f"[{key}]" if key != "!" else "[!]"

            table.add_row(
                Text(indicator, style="bold cyan"),
                Text(display_key, style=key_style),
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
                    self.selected_index = self._find_next_selectable(self.selected_index, -1)
                    live.update(self._render_menu())
                elif key == 'DOWN':
                    self.selected_index = self._find_next_selectable(self.selected_index, 1)
                    live.update(self._render_menu())
                elif key == 'ENTER':
                    # Exit live context and return
                    should_exit = True
                elif key == 'ESC':
                    # Find back or quit action
                    for i, (_, action, _, _) in enumerate(self.items):
                        if action in ("back", "quit"):
                            self.selected_index = i
                            break
                    should_exit = True
                elif key == 'q':
                    # Find quit action specifically
                    for i, (_, action, _, _) in enumerate(self.items):
                        if action == "quit":
                            self.selected_index = i
                            should_exit = True
                            break
                    # If no quit, try back
                    if not should_exit:
                        for i, (_, action, _, _) in enumerate(self.items):
                            if action == "back":
                                self.selected_index = i
                                should_exit = True
                                break
                elif key == 'b':
                    # Find back action
                    for i, (_, action, _, _) in enumerate(self.items):
                        if action == "back":
                            self.selected_index = i
                            should_exit = True
                            break
                elif key:  # Non-empty key
                    # Check if key matches a shortcut
                    for i, (shortcut, action, _, _) in enumerate(self.items):
                        if shortcut and key == shortcut.lower():
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
        from pr0loader.utils.ui import clear_screen
        clear_screen()

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

        # Hierarchical menu structure
        menu_items = [
            # Big red button - automated workflow
            ("!", "auto_pipeline", "🚀 AUTO: Full Pipeline", "Dev mode: fetch→prepare→train→validate"),
            ("", "", "─" * 40, ""),  # Separator

            # Data Ingestion
            ("1", "menu_ingest", "📥 Data Ingestion", "Fetch & download from pr0gramm →"),

            # Data Preparation
            ("2", "menu_prepare", "📊 Data Preparation", "Prepare datasets for training →"),

            # Model Training & Validation
            ("3", "menu_training", "🧠 Training & Validation", "Train and evaluate models →"),

            # Inference / Production
            ("4", "menu_inference", "🔮 Inference Mode", "Run predictions & serve API →"),

            ("", "", "─" * 40, ""),  # Separator

            # Settings & Config
            ("s", "menu_settings", "⚙️ Settings", "Configure pr0loader →"),

            # Quick actions
            ("i", "info", "📊 System Info", "Show status and statistics"),
            ("q", "quit", "❌ Quit", "Exit pr0loader"),
        ]

        # Filter out separators for key handling
        valid_items = [(k, a, n, d) for k, a, n, d in menu_items if k]

        # Gather status info
        status_info = self._gather_status_info()

        # Use arrow key menu if terminal supports it
        if not self.caps["is_dumb"]:
            menu = ArrowKeyMenu(
                items=menu_items,
                title="pr0loader - Main Menu",
                console=self.console,
                status_info=status_info,
            )
            return menu.run()
        else:
            # Fallback to simple prompt-based menu
            return self._show_simple_menu(valid_items)

    def show_submenu(self, submenu_type: str) -> Optional[str]:
        """Show a submenu based on type and return selected action."""
        from pr0loader.utils.ui import clear_screen
        clear_screen()

        if submenu_type == "menu_ingest":
            menu_items = [
                ("1", "sync", "🔄 Full Sync", "Fetch metadata + download assets"),
                ("2", "fetch", "📥 Fetch Metadata", "Download metadata only"),
                ("3", "download", "📁 Download Assets", "Download media files"),
                ("", "", "─" * 35, ""),
                ("l", "login", "🔐 Login", "Authenticate with pr0gramm"),
                ("o", "logout", "🚪 Logout", "Clear credentials"),
                ("", "", "─" * 35, ""),
                ("b", "back", "← Back", "Return to main menu"),
            ]
            title = "Data Ingestion"

        elif submenu_type == "menu_prepare":
            menu_items = [
                ("1", "prepare", "📊 Prepare Dataset", "Generate training CSV"),
                ("2", "prepare_split", "✂️ Split Dataset", "Create train/val/test split"),
                ("3", "prepare_stats", "📈 Dataset Stats", "Show dataset statistics"),
                ("", "", "─" * 35, ""),
                ("b", "back", "← Back", "Return to main menu"),
            ]
            title = "Data Preparation"

        elif submenu_type == "menu_training":
            menu_items = [
                ("1", "train", "🧠 Train Model", "Train tag prediction model"),
                ("2", "validate", "✅ Validate Model", "Evaluate on test set"),
                ("3", "train_resume", "▶️ Resume Training", "Continue from checkpoint"),
                ("", "", "─" * 35, ""),
                ("b", "back", "← Back", "Return to main menu"),
            ]
            title = "Training & Validation"

        elif submenu_type == "menu_inference":
            menu_items = [
                ("1", "predict", "🔮 Predict Tags", "Predict tags for images"),
                ("2", "api", "🌐 Start API", "Launch inference API server"),
                ("3", "ui", "🎨 Start UI", "Launch Gradio web interface"),
                ("4", "serve", "🚀 Serve Both", "Start API + UI together"),
                ("", "", "─" * 35, ""),
                ("b", "back", "← Back", "Return to main menu"),
            ]
            title = "Inference Mode"

        elif submenu_type == "menu_settings":
            menu_items = [
                ("1", "settings_dev", "🧪 Dev Mode", "Toggle development mode (1k images)"),
                ("2", "settings_flags", "🎭 Content Flags", "Configure SFW/NSFW/NSFL/POL"),
                ("3", "settings_performance", "⚡ Performance", "Database batch size & caching"),
                ("4", "setup", "⚙️ Full Setup", "Run setup wizard"),
                ("5", "init", "🏗️ Init Dirs", "Create data directories"),
                ("", "", "─" * 35, ""),
                ("b", "back", "← Back", "Return to main menu"),
            ]
            title = "Settings"
        else:
            return "back"

        # Filter out separators for key handling
        valid_items = [(k, a, n, d) for k, a, n, d in menu_items if k]

        # Gather status info
        status_info = self._gather_status_info()

        if not self.caps["is_dumb"]:
            menu = ArrowKeyMenu(
                items=menu_items,
                title=title,
                console=self.console,
                status_info=status_info,
            )
            return menu.run()
        else:
            return self._show_simple_menu(valid_items)

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
        from pr0loader.utils.ui import DialogBuilder

        dialog = DialogBuilder("🔄 Sync Configuration", "Configure metadata fetch and asset download")
        dialog.add_info("Sync will fetch metadata and download media files.")
        dialog.add_separator()
        dialog.add_confirm("full", "Full sync (re-fetch all metadata)?", default=False,
                          description="Warning: This will re-download all metadata from the beginning")
        dialog.add_confirm("include_videos", "Include videos?", default=False,
                          description="By default, only images are downloaded")
        dialog.add_confirm("verify", "Verify existing files?", default=True,
                          description="Use HEAD requests to check if local files match remote")
        dialog.add_confirm("metadata_only", "Metadata only (skip downloads)?", default=False)

        result = dialog.run()
        if result is None:
            return {"cancelled": True}

        # Handle start_from separately if not full sync
        if not result.get("full"):
            from pr0loader.utils.ui import clear_screen, console
            if Confirm.ask("\nStart from a specific ID?", default=False):
                result["start_from"] = IntPrompt.ask("Enter starting ID")
            else:
                result["start_from"] = None
        else:
            result["start_from"] = None

        return result


    def configure_download(self) -> dict:
        """Configure download options interactively."""
        from pr0loader.utils.ui import DialogBuilder

        dialog = DialogBuilder("📁 Download Configuration", "Configure media file downloads")
        dialog.add_info("Download media files from pr0gramm based on stored metadata.")
        dialog.add_separator()
        dialog.add_confirm("include_videos", "Include videos?", default=False,
                          description="By default, only images (jpg, png, gif) are downloaded")

        result = dialog.run()
        return result if result else {"cancelled": True}

    def configure_fetch(self) -> dict:
        """Configure fetch options interactively."""
        from pr0loader.utils.ui import DialogBuilder

        dialog = DialogBuilder("📥 Fetch Configuration", "Configure metadata fetching from pr0gramm API")
        dialog.add_info("Fetch will download metadata (tags, scores, etc.) without media files.")
        dialog.add_separator()
        dialog.add_confirm("full", "Full update (re-fetch all)?", default=False,
                          description="Warning: This will start from the beginning")

        result = dialog.run()
        if result is None:
            return {"cancelled": True}

        # Handle start_from separately if not full
        if not result.get("full"):
            if Confirm.ask("\nStart from a specific ID?", default=False):
                result["start_from"] = IntPrompt.ask("Enter starting ID")
            else:
                result["start_from"] = None
        else:
            result["start_from"] = None

        return result

    def configure_prepare(self) -> dict:
        """Configure prepare options interactively."""
        from pr0loader.utils.ui import DialogBuilder

        dialog = DialogBuilder("📊 Prepare Configuration", "Configure dataset preparation")
        dialog.add_info("Prepare will generate a training-ready CSV from the database.")
        dialog.add_separator()
        dialog.add_number("min_tags", "Minimum valid tags per item", default=5,
                         description="Items with fewer tags will be excluded", min_value=1)
        dialog.add_text("output", "Custom output file (leave empty for default)", default="")

        result = dialog.run()
        if result is None:
            return {"cancelled": True}

        # Convert empty string to None
        if not result.get("output"):
            result["output"] = None

        return result

    def configure_train(self) -> dict:
        """Configure train options interactively."""
        from pr0loader.utils.ui import DialogBuilder

        from pr0loader.utils.ui import DialogBuilder, clear_screen, console as ui_console

        clear_screen()
        ui_console.print(Panel(
            "[bold cyan]🧠 Train Configuration[/bold cyan]\n[dim]Configure model training parameters[/dim]",
            box=box.DOUBLE, border_style="blue"
        ))
        ui_console.print()

        # Find available datasets
        from pr0loader.config import load_settings
        settings = load_settings()
        datasets = sorted(settings.output_dir.glob("*_dataset.csv"), reverse=True)

        options = {}

        if datasets:
            ui_console.print("[cyan]Available datasets:[/cyan]")
            for i, ds in enumerate(datasets[:5], 1):
                ui_console.print(f"  {i}. {ds.name}")
            ui_console.print()

            if Confirm.ask("Use most recent dataset?", default=True):
                options["dataset"] = str(datasets[0])
            else:
                options["dataset"] = Prompt.ask("Enter dataset path")
        else:
            options["dataset"] = Prompt.ask("Enter dataset path")

        ui_console.print()
        options["epochs"] = IntPrompt.ask("[cyan]Number of epochs[/cyan]", default=5)
        options["batch_size"] = IntPrompt.ask("[cyan]Batch size[/cyan]", default=128)
        options["dev_mode"] = Confirm.ask("[cyan]Development mode (use subset)?[/cyan]", default=False)

        if Confirm.ask("[cyan]Specify custom output model path?[/cyan]", default=False):
            options["output"] = Prompt.ask("Output model path")
        else:
            options["output"] = None

        # Show summary
        ui_console.print()
        table = Table(title="Training Configuration", box=box.ROUNDED)
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")
        for k, v in options.items():
            table.add_row(k.replace("_", " ").title(), str(v) if v else "[dim]default[/dim]")
        ui_console.print(table)

        if not Confirm.ask("\n[bold]Proceed with training?[/bold]", default=True):
            return {"cancelled": True}

        return options

    def configure_predict(self) -> dict:
        """Configure predict options interactively."""
        from pr0loader.utils.ui import DialogBuilder, clear_screen, console as ui_console

        clear_screen()
        ui_console.print(Panel(
            "[bold cyan]🔮 Predict Configuration[/bold cyan]\n[dim]Configure tag prediction[/dim]",
            box=box.DOUBLE, border_style="blue"
        ))
        ui_console.print()

        options = {}

        # Get image paths
        image_paths = []
        ui_console.print("[cyan]Enter image paths (type 'done' when finished):[/cyan]")
        while True:
            path = Prompt.ask("  Image path", default="done")
            if path.lower() == 'done':
                break
            if Path(path).exists():
                image_paths.append(path)
                ui_console.print(f"    [green]✓[/green] Added: {path}")
            else:
                ui_console.print(f"    [red]✗[/red] File not found: {path}")

        if not image_paths:
            ui_console.print("[red]No images specified![/red]")
            return {"cancelled": True}

        options["images"] = image_paths
        ui_console.print()
        options["top_k"] = IntPrompt.ask("[cyan]Number of top tags to show[/cyan]", default=5)
        options["json_output"] = Confirm.ask("[cyan]Output as JSON?[/cyan]", default=False)

        if Confirm.ask("[cyan]Use custom model?[/cyan]", default=False):
            options["model"] = Prompt.ask("Model path")
        else:
            options["model"] = None

        if not Confirm.ask("\n[bold]Run prediction?[/bold]", default=True):
            return {"cancelled": True}

        return options

    def configure_api(self) -> dict:
        """Configure API server options interactively."""
        from pr0loader.utils.ui import DialogBuilder

        dialog = DialogBuilder("🌐 API Server Configuration", "Configure the inference API server")
        dialog.add_info("The API server provides REST endpoints for tag prediction.")
        dialog.add_separator()
        dialog.add_text("host", "Host to bind to", default="0.0.0.0",
                       description="Use 0.0.0.0 for all interfaces, 127.0.0.1 for localhost only")
        dialog.add_number("port", "Port number", default=8000)
        dialog.add_text("model", "Custom model path (leave empty for default)", default="")

        result = dialog.run()
        if result is None:
            return {"cancelled": True}

        if not result.get("model"):
            result["model"] = None

        return result


        return options

    def configure_ui(self) -> dict:
        """Configure Gradio UI options interactively."""
        from pr0loader.utils.ui import DialogBuilder, clear_screen, console as ui_console

        clear_screen()
        ui_console.print(Panel(
            "[bold cyan]🎨 Gradio UI Configuration[/bold cyan]\n[dim]Configure the web interface[/dim]",
            box=box.DOUBLE, border_style="blue"
        ))
        ui_console.print()

        options = {}

        # Mode selection
        use_api = Confirm.ask("[cyan]Connect to remote API server?[/cyan]", default=False)

        if use_api:
            options["api_url"] = Prompt.ask("[cyan]API server URL[/cyan]", default="http://localhost:8000")
            options["model"] = None
        else:
            options["api_url"] = None
            if Confirm.ask("[cyan]Use custom model path?[/cyan]", default=False):
                options["model"] = Prompt.ask("Model path")
            else:
                options["model"] = None

        ui_console.print()
        options["host"] = Prompt.ask("[cyan]Host to bind to[/cyan]", default="0.0.0.0")
        options["port"] = IntPrompt.ask("[cyan]Port[/cyan]", default=7860)
        options["share"] = Confirm.ask("[cyan]Create public Gradio link?[/cyan]", default=False)

        if not Confirm.ask("\n[bold]Start Gradio UI?[/bold]", default=True):
            return {"cancelled": True}

        return options

    def configure_serve(self) -> dict:
        """Configure combined server options interactively."""
        from pr0loader.utils.ui import DialogBuilder

        dialog = DialogBuilder("🚀 Full Server Configuration", "Configure API + Gradio UI together")
        dialog.add_info("This will start both the inference API and Gradio web interface.")
        dialog.add_separator()
        dialog.add_text("host", "Host to bind to", default="0.0.0.0")
        dialog.add_number("api_port", "API server port", default=8000)
        dialog.add_number("ui_port", "Gradio UI port", default=7860)
        dialog.add_text("model", "Custom model path (leave empty for default)", default="")

        result = dialog.run()
        if result is None:
            return {"cancelled": True}

        if not result.get("model"):
            result["model"] = None

        return result

    def configure_login(self) -> dict:
        """Configure login options interactively."""
        from pr0loader.utils.ui import clear_screen, console as ui_console

        clear_screen()
        ui_console.print(Panel(
            "[bold cyan]🔐 Login Configuration[/bold cyan]\n[dim]Authenticate with pr0gramm[/dim]",
            box=box.DOUBLE, border_style="blue"
        ))
        ui_console.print()

        options = {}

        # Show current status
        try:
            from pr0loader.auth import get_auth_manager
            auth = get_auth_manager()
            status = auth.get_status()

            if status.get("stored_credentials"):
                ui_console.print(f"[green]✓ Currently logged in as: {status.get('username')}[/green]\n")

            browsers = status.get("available_browsers", [])
            if browsers:
                ui_console.print(f"[dim]Available browsers: {', '.join(browsers)}[/dim]\n")
        except:
            pass

        ui_console.print("[cyan]Login methods:[/cyan]")
        ui_console.print("  • [bold]auto[/bold] - Try to extract cookies from all browsers")
        ui_console.print("  • [bold]browser[/bold] - Extract from specific browser")
        ui_console.print("  • [bold]interactive[/bold] - Manual login with username/password")
        ui_console.print()

        method = Prompt.ask(
            "[cyan]Select method[/cyan]",
            choices=["auto", "browser", "interactive"],
            default="auto"
        )

        options["method"] = method

        if method == "browser":
            browser = Prompt.ask(
                "[cyan]Which browser[/cyan]",
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

    def configure_auto_pipeline(self) -> dict:
        """Configure the automatic pipeline (big red button)."""
        from pr0loader.utils.ui import clear_screen, console as ui_console, ActionConfirm

        clear_screen()
        ui_console.print(Panel(
            "[bold red]🚀 AUTOMATIC PIPELINE[/bold red]\n"
            "[yellow]Complete workflow in development mode[/yellow]",
            box=box.DOUBLE, border_style="red"
        ))
        ui_console.print()

        ui_console.print("[bold]This will run the complete workflow:[/bold]")
        ui_console.print()
        steps = [
            ("1", "Enable dev mode", "Limit dataset to 1000 images"),
            ("2", "Fetch metadata", "Download item info from pr0gramm API"),
            ("3", "Download assets", "Download image files"),
            ("4", "Prepare dataset", "Generate training CSV with tags"),
            ("5", "Train model", "Train tag prediction neural network"),
            ("6", "Validate", "Evaluate model on test set"),
        ]

        table = Table(box=box.SIMPLE, show_header=False)
        table.add_column("Step", style="bold cyan", width=3)
        table.add_column("Action", style="bold", width=20)
        table.add_column("Description", style="dim")
        for step, action, desc in steps:
            table.add_row(step, action, desc)
        ui_console.print(table)
        ui_console.print()

        options = {
            "dev_mode": True,
            "dev_limit": 1000,
            "skip_fetch": False,
            "skip_download": False,
            "skip_prepare": False,
            "skip_train": False,
            "validate": True,
            "include_videos": False,
        }

        # Allow customization
        if Confirm.ask("[cyan]Customize settings?[/cyan]", default=False):
            options["dev_limit"] = IntPrompt.ask("[cyan]Number of images to use[/cyan]", default=1000)
            options["include_videos"] = Confirm.ask("[cyan]Include videos?[/cyan]", default=False)

        ui_console.print()
        if not Confirm.ask("[bold red]Start automatic pipeline?[/bold red]", default=True):
            return {"cancelled": True}

        return options

    def configure_dev_mode(self) -> dict:
        """Configure development mode settings."""
        from pr0loader.utils.ui import clear_screen, console as ui_console

        clear_screen()
        ui_console.print(Panel(
            "[bold cyan]🧪 Development Mode Settings[/bold cyan]\n"
            "[dim]Configure development mode for faster testing[/dim]",
            box=box.DOUBLE, border_style="blue"
        ))
        ui_console.print()

        # Show current status
        try:
            from pr0loader.config import load_settings
            settings = load_settings()
            current_dev_mode = settings.dev_mode
            current_limit = settings.dev_limit

            status_table = Table(box=box.ROUNDED, title="Current Settings")
            status_table.add_column("Setting", style="cyan")
            status_table.add_column("Value", style="green")
            status_table.add_row("Dev Mode", "✓ Enabled" if current_dev_mode else "✗ Disabled")
            status_table.add_row("Item Limit", str(current_limit))
            ui_console.print(status_table)
            ui_console.print()
        except:
            current_dev_mode = False
            current_limit = 1000

        options = {}
        options["enabled"] = Confirm.ask("[cyan]Enable development mode?[/cyan]", default=not current_dev_mode)

        if options["enabled"]:
            options["limit"] = IntPrompt.ask("[cyan]Max items to process[/cyan]", default=1000)

        ui_console.print()
        if not Confirm.ask("[bold]Save settings?[/bold]", default=True):
            return {"cancelled": True}

        return options

    def configure_content_flags(self) -> dict:
        """Configure content flags interactively."""
        from pr0loader.utils.ui import clear_screen, console as ui_console

        clear_screen()
        ui_console.print(Panel(
            "[bold cyan]🎭 Content Flags Settings[/bold cyan]\n"
            "[dim]Configure which content types to include[/dim]",
            box=box.DOUBLE, border_style="blue"
        ))
        ui_console.print()

        # Show current status
        try:
            from pr0loader.config import load_settings
            settings = load_settings()
            current_flags = settings.content_flags
        except:
            current_flags = 15

        ui_console.print(f"[dim]Current flags value: {current_flags}[/dim]")
        ui_console.print()
        ui_console.print("[cyan]Select content types to include:[/cyan]")
        ui_console.print()

        flags = 0
        if Confirm.ask("  [green]SFW[/green] (Safe for Work)?", default=bool(current_flags & 1)):
            flags |= 1
        if Confirm.ask("  [yellow]NSFW[/yellow] (Not Safe for Work)?", default=bool(current_flags & 2)):
            flags |= 2
        if Confirm.ask("  [red]NSFL[/red] (Not Safe for Life)?", default=bool(current_flags & 4)):
            flags |= 4
        if Confirm.ask("  [magenta]POL[/magenta] (Political)?", default=bool(current_flags & 8)):
            flags |= 8

        if flags == 0:
            ui_console.print("[yellow]Warning: No content types selected. Setting to SFW only.[/yellow]")
            flags = 1

        options = {"flags": flags}

        self._show_options_summary("Content Flags", options)

        if not Confirm.ask("\nSave settings?", default=True):
            return {"cancelled": True}

        return options

    def configure_performance(self) -> dict:
        """Configure performance settings."""
        from pr0loader.utils.ui import clear_screen, console as ui_console

        clear_screen()
        ui_console.print(Panel(
            "[bold cyan]⚡ Performance Settings[/bold cyan]\n"
            "[dim]Tune performance for your system[/dim]",
            box=box.DOUBLE, border_style="blue"
        ))
        ui_console.print()

        # Show current setting
        try:
            from pr0loader.config import load_settings
            settings = load_settings()
            current_batch = settings.db_batch_size
        except:
            current_batch = 200

        ui_console.print(f"[dim]Current batch size: {current_batch}[/dim]")
        ui_console.print()
        ui_console.print("[yellow]Database Batch Size:[/yellow]")
        ui_console.print("  • Higher values = faster (especially on HDDs)")
        ui_console.print("  • Lower values = less RAM usage")
        ui_console.print()
        ui_console.print("[dim]Recommended values:[/dim]")
        ui_console.print("  • HDD: 100-200")
        ui_console.print("  • SSD: 200-500")
        ui_console.print("  • RAM disk: 500-1000")
        ui_console.print()

        batch_size = IntPrompt.ask(
            "[cyan]Database batch size[/cyan]",
            default=current_batch
        )

        options = {"db_batch_size": batch_size}

        self._show_options_summary("Performance", options)

        if not Confirm.ask("\nSave settings?", default=True):
            return {"cancelled": True}

        return options

    def configure_validate(self) -> dict:
        """Configure model validation options."""
        from pr0loader.utils.ui import clear_screen, console as ui_console

        clear_screen()
        ui_console.print(Panel(
            "[bold cyan]✅ Validate Model[/bold cyan]\n[dim]Evaluate model on test set[/dim]",
            box=box.DOUBLE, border_style="blue"
        ))
        ui_console.print()

        options = {}

        # Find available test datasets and models
        from pr0loader.config import load_settings
        settings = load_settings()

        test_datasets = sorted(
            settings.output_dir.glob("*_test.csv"),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )

        if test_datasets:
            ui_console.print("[cyan]Available test datasets:[/cyan]")
            for i, ds in enumerate(test_datasets[:5], 1):
                ui_console.print(f"  {i}. {ds.name}")
            ui_console.print()

            if Confirm.ask("Use most recent test dataset?", default=True):
                options["test_csv_path"] = test_datasets[0]
            else:
                path = Prompt.ask("Test dataset path")
                options["test_csv_path"] = Path(path) if path else None
        else:
            ui_console.print("[yellow]No test datasets found[/yellow]")
            if Confirm.ask("Specify test dataset path?", default=False):
                path = Prompt.ask("Test dataset path")
                options["test_csv_path"] = Path(path) if path else None
            else:
                options["test_csv_path"] = None

        ui_console.print()
        if Confirm.ask("Use custom model?", default=False):
            options["model_path"] = Path(Prompt.ask("Model path"))
        else:
            options["model_path"] = None

        options["top_k"] = IntPrompt.ask("[cyan]Top-K accuracy to evaluate[/cyan]", default=5)

        if not Confirm.ask("\n[bold]Run validation?[/bold]", default=True):
            return {"cancelled": True}

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

    while True:
        action = menu.show_main_menu()

        if action == "quit" or action is None:
            return None, {}

        # Handle submenus
        if action.startswith("menu_"):
            while True:
                sub_action = menu.show_submenu(action)
                if sub_action == "back" or sub_action is None:
                    break
                # Return the sub_action to be executed
                return _configure_action(menu, sub_action)
            continue  # Back to main menu

        # Handle direct actions
        return _configure_action(menu, action)


def _configure_action(menu: InteractiveMenu, action: str) -> tuple[Optional[str], dict]:
    """Configure options for an action and return (action, options)."""
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
    elif action == "auto_pipeline":
        options = menu.configure_auto_pipeline()
    elif action == "settings_dev":
        options = menu.configure_dev_mode()
    elif action == "settings_flags":
        options = menu.configure_content_flags()
    elif action == "settings_performance":
        options = menu.configure_performance()
    elif action == "validate":
        options = menu.configure_validate()
    # info, logout, setup, init don't need configuration

    return action, options

