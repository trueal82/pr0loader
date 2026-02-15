"""
pr0loader CLI - A beautiful command-line interface for the pr0loader toolchain.

Usage:
    pr0loader info              Show database and configuration info
    pr0loader fetch             Fetch metadata from pr0gramm API
    pr0loader download          Download media files
    pr0loader prepare           Prepare dataset for training
    pr0loader train             Train tag prediction model
    pr0loader predict           Predict tags for images
    pr0loader run-all           Run complete pipeline
"""

import logging
from pathlib import Path
from typing import Optional

import typer
from rich.table import Table
from rich import box

from pr0loader import __version__
from pr0loader.config import Settings, load_settings
from pr0loader.utils.console import (
    set_headless,
    print_header,
    print_info,
    print_success,
    print_error,
    print_warning,
    print_stats_table,
    console,
)

# Create Typer app with rich markup
app = typer.Typer(
    name="pr0loader",
    help="🚀 A toolchain for fetching, processing, and training ML models on pr0gramm data",
    add_completion=True,
    rich_markup_mode="rich",
)


def setup_logging(verbose: bool, headless: bool):
    """Configure logging based on options."""
    level = logging.DEBUG if verbose else logging.INFO

    if headless:
        # Simple format for headless mode
        logging.basicConfig(
            level=level,
            format='[%(asctime)s] %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        # Quieter logging when using Rich
        logging.basicConfig(
            level=logging.WARNING,
            format='%(message)s'
        )

    set_headless(headless)


def check_setup(headless: bool = False) -> bool:
    """
    Check if setup is needed and run wizard if interactive.

    Returns:
        True if setup is complete, False otherwise.
    """
    from pr0loader.utils.setup import env_file_exists, check_and_run_setup

    if env_file_exists():
        return True

    if headless:
        console.print("[yellow]No .env file found. Run 'pr0loader setup' first.[/yellow]")
        return False


def check_auth_for_content_flags(settings: "Settings", headless: bool = False) -> bool:
    """
    Check if authentication is available when content flags require it.

    Content flags > 1 means non-SFW content, which requires authentication.

    Returns:
        True if auth is available or not needed, False if auth is required but missing.
    """
    # SFW only (flags=1) doesn't require auth
    if settings.content_flags == 1:
        return True

    # Check if we have cookies in settings
    if settings.pp and settings.me:
        return True

    # Check if we have stored credentials
    try:
        from pr0loader.auth import get_auth_manager
        auth = get_auth_manager()
        creds = auth.store.load()
        if creds and creds.is_valid():
            return True
    except Exception:
        pass

    # No auth available
    if headless:
        console.print("[red]Error: Authentication required for content flags > 1[/red]")
        console.print("[yellow]Content flags are set to {}, which includes non-SFW content.[/yellow]".format(settings.content_flags))
        console.print("[yellow]Run 'pr0loader login' first, or set CONTENT_FLAGS=1 for SFW only.[/yellow]")
    else:
        print_error("Authentication required for content flags > 1")
        print_warning(f"Content flags are set to {settings.content_flags}, which includes non-SFW content.")
        print_info("Options:")
        print_info("  1. Run 'pr0loader login --auto' to authenticate")
        print_info("  2. Set CONTENT_FLAGS=1 in .env for SFW-only content")

    return False

    return check_and_run_setup()


def show_banner():
    """Show application banner."""
    banner = """
[bold magenta]              ___  _                 _           
 _ __  _ __  / _ \\| | ___   __ _  __| | ___ _ __ 
| '_ \\| '__|/ | | | |/ _ \\ / _` |/ _` |/ _ \\ '__|
| |_) | |  / |_| | | (_) | (_| | (_| |  __/ |   
| .__/|_|  \\___/|_|\\___/ \\__,_|\\__,_|\\___|_|   
|_|[/bold magenta]                                              
    """
    console.print(banner)
    console.print(f"[dim]Version {__version__} - Tag Prediction Toolchain[/dim]\n")


def _run_interactive(ctx: typer.Context):
    """Run interactive menu mode."""
    from pr0loader.utils.menu import run_interactive_menu, is_interactive

    if not is_interactive():
        console.print("[yellow]Not running in interactive terminal. Use --help for commands.[/yellow]")
        raise typer.Exit(0)

    # Check if setup is needed
    if not check_setup(headless=False):
        raise typer.Exit(0)

    # Main loop - keep showing menu until quit
    while True:
        action, options = run_interactive_menu()

        if action is None or action == "quit":
            console.print("\n[dim]Goodbye![/dim]")
            raise typer.Exit(0)

        # Execute the selected action with options
        try:
            settings = load_settings()

            if action == "info":
                _execute_info()
            elif action == "sync":
                _execute_sync(settings, options)
            elif action == "fetch":
                _execute_fetch(settings, options)
            elif action == "download":
                _execute_download(settings, options)
            elif action == "prepare":
                _execute_prepare(settings, options)
            elif action == "train":
                _execute_train(settings, options)
            elif action == "predict":
                _execute_predict(settings, options)
            elif action == "api":
                _execute_api(settings, options)
            elif action == "ui":
                _execute_ui(settings, options)
            elif action == "serve":
                _execute_serve(settings, options)
            elif action == "login":
                _execute_login(options)
            elif action == "logout":
                _execute_logout()
            elif action == "setup":
                _execute_setup()

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted[/yellow]")
        except Exception as e:
            print_error(f"Error: {e}")

        # Pause before returning to menu
        console.print()
        from rich.prompt import Prompt
        Prompt.ask("[dim]Press Enter to return to menu[/dim]", default="")


def _execute_info():
    """Execute info command from interactive menu."""
    # Show banner already shown by menu, so just run info logic
    print_header("System Information", "Database and configuration overview")

    try:
        settings = load_settings()

        # Data directory table
        dir_table = Table(title="Data Directory Structure", box=box.ROUNDED)
        dir_table.add_column("Path Type", style="cyan")
        dir_table.add_column("Location", style="green")
        dir_table.add_column("Status", style="yellow")

        def path_status(p: Path) -> str:
            if p.exists():
                if p.is_dir():
                    return "✓ exists"
                elif p.is_file():
                    size = p.stat().st_size
                    if size > 1024 * 1024:
                        return f"✓ {size / 1024 / 1024:.1f} MB"
                    elif size > 1024:
                        return f"✓ {size / 1024:.1f} KB"
                    return f"✓ {size} B"
            return "✗ not found"

        dir_table.add_row("Base data dir", str(settings.data_dir), path_status(settings.data_dir))
        dir_table.add_row("Database", str(settings.db_path), path_status(settings.db_path))
        dir_table.add_row("Media files", str(settings.filesystem_prefix), path_status(settings.filesystem_prefix))
        dir_table.add_row("Output/datasets", str(settings.output_dir), path_status(settings.output_dir))
        dir_table.add_row("Models", str(settings.model_path.parent), path_status(settings.model_path.parent))
        dir_table.add_row("Trained model", str(settings.model_path), path_status(settings.model_path))
        dir_table.add_row("Auth/credentials", str(settings.auth_dir), path_status(settings.auth_dir))

        console.print(dir_table)
        console.print()

        # Configuration table
        config_table = Table(title="Configuration", box=box.ROUNDED)
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Content flags", str(settings.content_flags))
        config_table.add_row("Dev mode", "✓ Yes" if settings.dev_mode else "✗ No")

        console.print(config_table)
        console.print()

        # Database stats
        if settings.db_path.exists():
            from pr0loader.storage import SQLiteStorage

            with SQLiteStorage(settings.db_path) as storage:
                db_table = Table(title="Database Statistics", box=box.ROUNDED)
                db_table.add_column("Metric", style="cyan")
                db_table.add_column("Value", style="green", justify="right")

                item_count = storage.get_item_count()
                min_id = storage.get_min_id()
                max_id = storage.get_max_id()

                db_table.add_row("Total items", f"{item_count:,}")
                db_table.add_row("ID range", f"{min_id:,} - {max_id:,}")

                # Top tags
                top_tags = storage.get_tag_counts(limit=10)
                if top_tags:
                    tags_str = ", ".join(f"{t}({c})" for t, c in top_tags[:5])
                    db_table.add_row("Top tags", tags_str)

                console.print(db_table)
        else:
            print_warning(f"Database not found: {settings.db_path}")

        # Model info
        console.print()
        if settings.model_path.exists():
            print_success(f"Model found: {settings.model_path}")
            mapping_path = settings.model_path.with_suffix('.tags.json')
            if mapping_path.exists():
                import json
                with open(mapping_path) as f:
                    mapping = json.load(f)
                print_info(f"Model has {mapping.get('num_classes', '?')} tag classes")
        else:
            print_warning(f"No trained model found at: {settings.model_path}")

    except Exception as e:
        print_error(f"Failed to load settings: {e}")


def _execute_sync(settings: Settings, options: dict):
    """Execute sync with options from interactive menu."""
    # Check auth if content flags require it
    if not check_auth_for_content_flags(settings, headless=False):
        return

    if options.get("full"):
        settings.full_update = True
    if options.get("start_from"):
        settings.start_from = options["start_from"]

    from pr0loader.pipeline import SyncPipeline
    pipeline = SyncPipeline(settings)
    pipeline.run(
        include_videos=options.get("include_videos", False),
        verify_existing=options.get("verify", True),
        download_media=not options.get("metadata_only", False),
    )


def _execute_fetch(settings: Settings, options: dict):
    """Execute fetch with options from interactive menu."""
    # Check auth if content flags require it
    if not check_auth_for_content_flags(settings, headless=False):
        return

    if options.get("full"):
        settings.full_update = True
    if options.get("start_from"):
        settings.start_from = options["start_from"]

    from pr0loader.pipeline import FetchPipeline
    pipeline = FetchPipeline(settings)
    pipeline.run()


def _execute_download(settings: Settings, options: dict):
    """Execute download with options from interactive menu."""
    # Check auth if content flags require it
    if not check_auth_for_content_flags(settings, headless=False):
        return

    from pr0loader.pipeline import DownloadPipeline
    pipeline = DownloadPipeline(settings)
    pipeline.run(include_videos=options.get("include_videos", False))


def _execute_prepare(settings: Settings, options: dict):
    """Execute prepare with options from interactive menu."""
    settings.min_valid_tags = options.get("min_tags", 5)

    output_path = None
    if options.get("output"):
        output_path = Path(options["output"])

    from pr0loader.pipeline import PreparePipeline
    pipeline = PreparePipeline(settings)
    pipeline.run(output_file=output_path)


def _execute_train(settings: Settings, options: dict):
    """Execute train with options from interactive menu."""
    settings.num_epochs = options.get("epochs", 5)
    settings.batch_size = options.get("batch_size", 128)
    settings.dev_mode = options.get("dev_mode", False)

    dataset_path = Path(options["dataset"])
    output_path = Path(options["output"]) if options.get("output") else None

    from pr0loader.pipeline import TrainPipeline
    pipeline = TrainPipeline(settings)
    pipeline.run(csv_path=dataset_path, output_path=output_path)


def _execute_predict(settings: Settings, options: dict):
    """Execute predict with options from interactive menu."""
    image_paths = [Path(p) for p in options["images"]]
    model_path = Path(options["model"]) if options.get("model") else None

    from pr0loader.pipeline import PredictPipeline
    pipeline = PredictPipeline(settings)
    results = pipeline.run(
        image_paths=image_paths,
        model_path=model_path,
        top_k=options.get("top_k", 5),
    )

    if options.get("json_output"):
        import json
        output = [
            {
                "image": r.image_path,
                "tags": [{"tag": t, "confidence": c} for t, c in r.top_5_tags]
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))


def _execute_run_all(settings: Settings, options: dict):
    """Execute run-all with options from interactive menu."""
    settings.dev_mode = options.get("dev_mode", False)

    console.print("\n[bold cyan]═══ Running Pipeline ═══[/bold cyan]\n")

    # Step 1: Fetch
    if not options.get("skip_fetch"):
        console.print("[bold cyan]Step 1: Fetch[/bold cyan]")
        from pr0loader.pipeline import FetchPipeline
        FetchPipeline(settings).run()

    # Step 2: Download
    if not options.get("skip_download"):
        console.print("\n[bold cyan]Step 2: Download[/bold cyan]")
        from pr0loader.pipeline import DownloadPipeline
        DownloadPipeline(settings).run(include_videos=options.get("include_videos", False))

    # Step 3: Prepare
    dataset_path = None
    if not options.get("skip_prepare"):
        console.print("\n[bold cyan]Step 3: Prepare[/bold cyan]")
        from pr0loader.pipeline import PreparePipeline
        _, dataset_path = PreparePipeline(settings).run()

    # Step 4: Train
    if not options.get("skip_train"):
        console.print("\n[bold cyan]Step 4: Train[/bold cyan]")

        if dataset_path is None:
            datasets = sorted(settings.output_dir.glob("*_dataset.csv"), reverse=True)
            if datasets:
                dataset_path = datasets[0]
            else:
                print_error("No dataset found. Run prepare step first.")
                return

        from pr0loader.pipeline import TrainPipeline
        TrainPipeline(settings).run(csv_path=dataset_path)

    print_header("🎉 Pipeline Complete!", "All stages finished successfully")


def _execute_login(options: dict):
    """Execute login with options from interactive menu."""
    from pr0loader.auth import get_auth_manager

    print_header("🔐 Login", "Authenticating with pr0gramm")

    auth = get_auth_manager()
    credentials = None
    method = options.get("method", "auto")

    if method == "auto":
        print_info("Trying to extract cookies from browsers...")
        credentials = auth.get_credentials(auto_login=True)
    elif method == "browser":
        browser = options.get("browser", "firefox")
        print_info(f"Extracting from {browser}...")
        credentials = auth.extract_from_browser(browser)
    elif method == "interactive":
        print_info("Starting interactive login...")
        credentials = auth.login_interactive()

    if credentials:
        print_success(f"Logged in as: {credentials.username}")
        print_info(f"NSFW Access: {'Yes' if credentials.is_verified else 'No'}")
    else:
        print_error("Login failed")


def _execute_logout():
    """Execute logout."""
    from pr0loader.auth import get_auth_manager

    auth = get_auth_manager()
    if auth.logout():
        print_success("Logged out successfully")
    else:
        print_warning("No credentials to clear")


def _execute_setup():
    """Execute setup wizard."""
    from pr0loader.utils.setup import run_setup_wizard
    run_setup_wizard()


def _execute_api(settings: Settings, options: dict):
    """Execute API server with options from interactive menu."""
    print_header("🌐 Starting API Server", f"http://{options.get('host', '0.0.0.0')}:{options.get('port', 8000)}")

    import uvicorn
    from pr0loader.api.server import create_app

    model_path = Path(options["model"]) if options.get("model") else None
    app_instance = create_app(model_path=model_path)

    print_info(f"API documentation: http://{options.get('host', '0.0.0.0')}:{options.get('port', 8000)}/docs")
    print_info("Press Ctrl+C to stop")
    console.print()

    uvicorn.run(
        app_instance,
        host=options.get("host", "0.0.0.0"),
        port=options.get("port", 8000),
        log_level="info",
    )


def _execute_ui(settings: Settings, options: dict):
    """Execute Gradio UI with options from interactive menu."""
    from pr0loader.api.gradio_ui import launch_gradio

    mode = "Remote API" if options.get("api_url") else "Local Model"
    print_header("🎨 Starting Gradio UI", f"Mode: {mode}")

    model_path = Path(options["model"]) if options.get("model") else None

    print_info(f"Web UI: http://{options.get('host', '0.0.0.0')}:{options.get('port', 7860)}")
    if options.get("api_url"):
        print_info(f"Using API: {options['api_url']}")
    print_info("Press Ctrl+C to stop")
    console.print()

    launch_gradio(
        model_path=model_path,
        api_url=options.get("api_url"),
        host=options.get("host", "0.0.0.0"),
        port=options.get("port", 7860),
        share=options.get("share", False),
    )


def _execute_serve(settings: Settings, options: dict):
    """Execute combined API + UI server with options from interactive menu."""
    import threading
    import uvicorn
    from pr0loader.api.server import create_app
    from pr0loader.api.gradio_ui import launch_gradio

    host = options.get("host", "0.0.0.0")
    api_port = options.get("api_port", 8000)
    ui_port = options.get("ui_port", 7860)

    print_header("🚀 Starting Full Server", "API + Gradio UI")
    print_info(f"API Server: http://{host}:{api_port}")
    print_info(f"Web UI: http://{host}:{ui_port}")
    print_info("Press Ctrl+C to stop")
    console.print()

    model_path = Path(options["model"]) if options.get("model") else None
    api_app = create_app(model_path=model_path)

    def run_api():
        uvicorn.run(api_app, host=host, port=api_port, log_level="warning")

    api_thread = threading.Thread(target=run_api, daemon=True)
    api_thread.start()

    print_success("API server started")

    api_url = f"http://{host}:{api_port}"
    launch_gradio(
        api_url=api_url,
        host=host,
        port=ui_port,
        share=False,
    )


# Common options
def common_options(
    headless: bool = typer.Option(False, "--headless", "-H", help="Run without fancy output (for scripts/CI)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """Common options for all commands."""
    setup_logging(verbose, headless)
    if not headless:
        show_banner()


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    headless: bool = typer.Option(False, "--headless", "-H", help="Run without fancy output"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging"),
):
    """🚀 pr0loader - Tag Prediction Toolchain for pr0gramm

    Run without arguments for interactive menu.
    """
    setup_logging(verbose, headless)
    ctx.ensure_object(dict)
    ctx.obj["headless"] = headless
    ctx.obj["verbose"] = verbose

    # If no command given, show interactive menu
    if ctx.invoked_subcommand is None and not headless:
        _run_interactive(ctx)


@app.command()
def info(
    ctx: typer.Context,
):
    """📊 Show database and configuration information."""
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    print_header("System Information", "Database and configuration overview")

    try:
        settings = load_settings()

        # Data directory table
        dir_table = Table(title="Data Directory Structure", box=box.ROUNDED)
        dir_table.add_column("Path Type", style="cyan")
        dir_table.add_column("Location", style="green")
        dir_table.add_column("Status", style="yellow")

        def path_status(p: Path) -> str:
            if p.exists():
                if p.is_dir():
                    return "✓ exists"
                elif p.is_file():
                    size = p.stat().st_size
                    if size > 1024 * 1024:
                        return f"✓ {size / 1024 / 1024:.1f} MB"
                    elif size > 1024:
                        return f"✓ {size / 1024:.1f} KB"
                    return f"✓ {size} B"
            return "✗ not found"

        dir_table.add_row("Base data dir", str(settings.data_dir), path_status(settings.data_dir))
        dir_table.add_row("Database", str(settings.db_path), path_status(settings.db_path))
        dir_table.add_row("Media files", str(settings.filesystem_prefix), path_status(settings.filesystem_prefix))
        dir_table.add_row("Output/datasets", str(settings.output_dir), path_status(settings.output_dir))
        dir_table.add_row("Models", str(settings.model_path.parent), path_status(settings.model_path.parent))
        dir_table.add_row("Trained model", str(settings.model_path), path_status(settings.model_path))
        dir_table.add_row("Auth/credentials", str(settings.auth_dir), path_status(settings.auth_dir))

        console.print(dir_table)
        console.print()

        # Configuration table
        config_table = Table(title="Configuration", box=box.ROUNDED)
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Content flags", str(settings.content_flags))
        config_table.add_row("Dev mode", "✓ Yes" if settings.dev_mode else "✗ No")

        console.print(config_table)
        console.print()

        # Database stats
        if settings.db_path.exists():
            from pr0loader.storage import SQLiteStorage

            with SQLiteStorage(settings.db_path) as storage:
                db_table = Table(title="Database Statistics", box=box.ROUNDED)
                db_table.add_column("Metric", style="cyan")
                db_table.add_column("Value", style="green", justify="right")

                item_count = storage.get_item_count()
                min_id = storage.get_min_id()
                max_id = storage.get_max_id()

                db_table.add_row("Total items", f"{item_count:,}")
                db_table.add_row("ID range", f"{min_id:,} - {max_id:,}")

                # Top tags
                top_tags = storage.get_tag_counts(limit=10)
                if top_tags:
                    tags_str = ", ".join(f"{t}({c})" for t, c in top_tags[:5])
                    db_table.add_row("Top tags", tags_str)

                console.print(db_table)
        else:
            print_warning(f"Database not found: {settings.db_path}")

        # Model info
        console.print()
        if settings.model_path.exists():
            print_success(f"Model found: {settings.model_path}")
            mapping_path = settings.model_path.with_suffix('.tags.json')
            if mapping_path.exists():
                import json
                with open(mapping_path) as f:
                    mapping = json.load(f)
                print_info(f"Model has {mapping.get('num_classes', '?')} tag classes")
        else:
            print_warning(f"No trained model found at: {settings.model_path}")

    except Exception as e:
        print_error(f"Failed to load settings: {e}")
        raise typer.Exit(1)


@app.command()
def init(
    ctx: typer.Context,
):
    """🏗️ Initialize the pr0loader data directory structure.

    Creates all required directories under the configured DATA_DIR.
    """
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    print_header("🏗️ Initialize", "Creating data directory structure")

    try:
        settings = load_settings()

        print_info(f"Data directory: {settings.data_dir}")
        console.print()

        # Create directories
        settings.ensure_directories()

        dirs_created = [
            ("Base data dir", settings.data_dir),
            ("Media files", settings.filesystem_prefix),
            ("Output/datasets", settings.output_dir),
            ("Models", settings.model_path.parent),
            ("Checkpoints", settings.checkpoint_dir),
            ("Auth/credentials", settings.auth_dir),
        ]

        for name, path in dirs_created:
            if path.exists():
                print_success(f"{name}: {path}")
            else:
                print_warning(f"{name}: {path} (failed to create)")

        console.print()
        print_success("Data directory structure initialized!")
        print_info("Run 'pr0loader login --auto' to authenticate")

    except Exception as e:
        print_error(f"Initialization failed: {e}")
        raise typer.Exit(1)


@app.command()
def setup(
    ctx: typer.Context,
    force: bool = typer.Option(False, "--force", "-f", help="Reconfigure even if .env exists"),
):
    """⚙️ Run the setup wizard to create or update configuration.

    Creates a .env file with your pr0loader configuration.
    If a config already exists, shows current values as defaults.
    """
    headless = ctx.obj.get("headless", False)
    if headless:
        print_error("Setup wizard requires interactive terminal")
        print_info("Create .env manually from template.env")
        raise typer.Exit(1)

    show_banner()

    from pr0loader.utils.setup import run_setup_wizard

    result = run_setup_wizard()

    if not result:
        raise typer.Exit(1)


@app.command()
def fetch(
    ctx: typer.Context,
    full_update: bool = typer.Option(False, "--full", "-f", help="Perform full update (re-fetch all)"),
    start_from: Optional[int] = typer.Option(None, "--start-from", "-s", help="Start from specific ID"),
):
    """📥 Fetch metadata from pr0gramm API."""
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    try:
        settings = load_settings()

        # Check auth if content flags require it
        if not check_auth_for_content_flags(settings, headless):
            raise typer.Exit(1)

        if full_update:
            settings.full_update = True
        if start_from:
            settings.start_from = start_from

        from pr0loader.pipeline import FetchPipeline
        pipeline = FetchPipeline(settings)
        pipeline.run()

    except Exception as e:
        print_error(f"Fetch failed: {e}")
        raise typer.Exit(1)


@app.command()
def download(
    ctx: typer.Context,
    include_videos: bool = typer.Option(False, "--include-videos", "-V", help="Also download videos (default: images only)"),
):
    """📁 Download media files from pr0gramm.

    By default, only images (jpg, jpeg, png, gif) are downloaded.
    Use --include-videos to also download video files.
    """
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    try:
        settings = load_settings()

        # Check auth if content flags require it
        if not check_auth_for_content_flags(settings, headless):
            raise typer.Exit(1)

        from pr0loader.pipeline import DownloadPipeline
        pipeline = DownloadPipeline(settings)
        pipeline.run(include_videos=include_videos)

    except Exception as e:
        print_error(f"Download failed: {e}")
        raise typer.Exit(1)


@app.command()
def sync(
    ctx: typer.Context,
    full: bool = typer.Option(False, "--full", "-f", help="Full sync (re-fetch all metadata)"),
    start_from: Optional[int] = typer.Option(None, "--start-from", "-s", help="Start from specific ID"),
    include_videos: bool = typer.Option(False, "--include-videos", "-V", help="Also download videos (default: images only)"),
    verify: bool = typer.Option(True, "--verify/--no-verify", help="Verify existing files with HEAD request"),
    metadata_only: bool = typer.Option(False, "--metadata-only", "-m", help="Only fetch metadata, skip downloads"),
):
    """🔄 Full sync: fetch metadata and download assets in one run.

    Metadata is always fetched for all items (images and videos).
    By default, only images are downloaded. Use --include-videos to also download videos.
    Existing files are verified via HEAD request - only re-downloaded if size differs.
    """
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    try:
        settings = load_settings()

        # Check auth if content flags require it
        if not check_auth_for_content_flags(settings, headless):
            raise typer.Exit(1)

        if full:
            settings.full_update = True
        if start_from:
            settings.start_from = start_from

        from pr0loader.pipeline import SyncPipeline
        pipeline = SyncPipeline(settings)
        pipeline.run(
            include_videos=include_videos,
            verify_existing=verify,
            download_media=not metadata_only,
        )

    except Exception as e:
        print_error(f"Sync failed: {e}")
        raise typer.Exit(1)


@app.command()
def prepare(
    ctx: typer.Context,
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output CSV file path"),
    min_tags: int = typer.Option(5, "--min-tags", "-t", help="Minimum valid tags per item"),
):
    """📊 Prepare dataset for ML training."""
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    try:
        settings = load_settings()
        settings.min_valid_tags = min_tags

        from pr0loader.pipeline import PreparePipeline
        pipeline = PreparePipeline(settings)
        stats, output_path = pipeline.run(output_file=output)

        print_info(f"Dataset saved to: {output_path}")

    except Exception as e:
        print_error(f"Prepare failed: {e}")
        raise typer.Exit(1)


@app.command()
def train(
    ctx: typer.Context,
    dataset: Path = typer.Argument(..., help="Path to training CSV file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output model path"),
    epochs: int = typer.Option(5, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(128, "--batch-size", "-b", help="Training batch size"),
    dev_mode: bool = typer.Option(False, "--dev", "-d", help="Development mode (use subset of data)"),
):
    """🧠 Train tag prediction model."""
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    if not dataset.exists():
        print_error(f"Dataset not found: {dataset}")
        raise typer.Exit(1)

    try:
        settings = load_settings()
        settings.num_epochs = epochs
        settings.batch_size = batch_size
        settings.dev_mode = dev_mode

        from pr0loader.pipeline import TrainPipeline
        pipeline = TrainPipeline(settings)
        model_path = pipeline.run(csv_path=dataset, output_path=output)

        if model_path:
            print_success(f"Model saved to: {model_path}")

    except Exception as e:
        print_error(f"Training failed: {e}")
        raise typer.Exit(1)


@app.command()
def predict(
    ctx: typer.Context,
    images: list[Path] = typer.Argument(..., help="Image file(s) to predict tags for"),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to trained model"),
    top_k: int = typer.Option(5, "--top", "-k", help="Number of top tags to show"),
    json_output: bool = typer.Option(False, "--json", "-j", help="Output results as JSON"),
):
    """🔮 Predict tags for images."""
    headless = ctx.obj.get("headless", False)
    if not headless and not json_output:
        show_banner()

    # Validate image paths
    for img in images:
        if not img.exists():
            print_error(f"Image not found: {img}")
            raise typer.Exit(1)

    try:
        settings = load_settings()

        from pr0loader.pipeline import PredictPipeline
        pipeline = PredictPipeline(settings)

        if json_output:
            # Quiet mode for JSON output
            set_headless(True)

        results = pipeline.run(image_paths=images, model_path=model, top_k=top_k)

        if json_output:
            import json
            output = [
                {
                    "image": r.image_path,
                    "tags": [{"tag": t, "confidence": c} for t, c in r.top_5_tags]
                }
                for r in results
            ]
            print(json.dumps(output, indent=2))

    except Exception as e:
        print_error(f"Prediction failed: {e}")
        raise typer.Exit(1)


@app.command("run-all")
def run_all(
    ctx: typer.Context,
    skip_fetch: bool = typer.Option(False, "--skip-fetch", help="Skip fetch step"),
    skip_download: bool = typer.Option(False, "--skip-download", help="Skip download step"),
    skip_prepare: bool = typer.Option(False, "--skip-prepare", help="Skip prepare step"),
    skip_train: bool = typer.Option(False, "--skip-train", help="Skip train step"),
    include_videos: bool = typer.Option(False, "--include-videos", "-V", help="Also download videos (default: images only)"),
    dev_mode: bool = typer.Option(False, "--dev", "-d", help="Development mode (use subset of data)"),
):
    """🚀 Run complete pipeline (fetch → download → prepare → train).

    Metadata is fetched for all items. By default, only images are downloaded.
    """
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    print_header(
        "Complete Pipeline",
        "Running all stages: fetch → download → prepare → train"
    )

    try:
        settings = load_settings()
        settings.dev_mode = dev_mode

        # Check auth if content flags require it (for fetch/download steps)
        if not skip_fetch or not skip_download:
            if not check_auth_for_content_flags(settings, headless):
                raise typer.Exit(1)

        total_steps = 4 - sum([skip_fetch, skip_download, skip_prepare, skip_train])
        current_step = 0

        # Step 1: Fetch
        if not skip_fetch:
            current_step += 1
            console.print(f"\n[bold cyan]═══ Step {current_step}/{total_steps}: Fetch ═══[/bold cyan]")
            from pr0loader.pipeline import FetchPipeline
            FetchPipeline(settings).run()

        # Step 2: Download
        if not skip_download:
            current_step += 1
            console.print(f"\n[bold cyan]═══ Step {current_step}/{total_steps}: Download ═══[/bold cyan]")
            from pr0loader.pipeline import DownloadPipeline
            DownloadPipeline(settings).run(include_videos=include_videos)

        # Step 3: Prepare
        dataset_path = None
        if not skip_prepare:
            current_step += 1
            console.print(f"\n[bold cyan]═══ Step {current_step}/{total_steps}: Prepare ═══[/bold cyan]")
            from pr0loader.pipeline import PreparePipeline
            _, dataset_path = PreparePipeline(settings).run()

        # Step 4: Train
        if not skip_train:
            current_step += 1
            console.print(f"\n[bold cyan]═══ Step {current_step}/{total_steps}: Train ═══[/bold cyan]")

            if dataset_path is None:
                # Find most recent dataset
                output_dir = settings.output_dir
                datasets = sorted(output_dir.glob("*_dataset.csv"), reverse=True)
                if datasets:
                    dataset_path = datasets[0]
                    print_info(f"Using dataset: {dataset_path}")
                else:
                    print_error("No dataset found. Run prepare step first.")
                    raise typer.Exit(1)

            from pr0loader.pipeline import TrainPipeline
            TrainPipeline(settings).run(csv_path=dataset_path)

        # Done!
        console.print()
        print_header("🎉 Pipeline Complete!", "All stages finished successfully")

    except KeyboardInterrupt:
        print_warning("\nPipeline interrupted by user")
        raise typer.Exit(130)
    except Exception as e:
        print_error(f"Pipeline failed: {e}")
        raise typer.Exit(1)


@app.command()
def api(
    ctx: typer.Context,
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to listen on"),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to trained model"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload (development)"),
):
    """🌐 Start the inference API server.

    Starts a FastAPI server for tag prediction inference.

    Endpoints:
    - GET  /         - Health check
    - GET  /health   - Health check
    - POST /predict  - Predict tags for single image
    - POST /predict/batch - Predict tags for multiple images
    """
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    print_header("🌐 Starting API Server", f"http://{host}:{port}")

    try:
        import uvicorn
        from pr0loader.api.server import create_app

        # Create app with model path
        app_instance = create_app(model_path=model)

        print_info(f"API documentation: http://{host}:{port}/docs")
        print_info(f"Health check: http://{host}:{port}/health")
        print_info("Press Ctrl+C to stop")
        console.print()

        uvicorn.run(
            app_instance,
            host=host,
            port=port,
            reload=reload,
            log_level="info",
        )

    except ImportError:
        print_error("FastAPI/uvicorn not installed. Install with: pip install fastapi uvicorn python-multipart")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"API server failed: {e}")
        raise typer.Exit(1)


@app.command()
def ui(
    ctx: typer.Context,
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(7860, "--port", "-p", help="Port to listen on"),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to trained model"),
    api_url: Optional[str] = typer.Option(None, "--api-url", "-a", help="Use remote API instead of local model"),
    share: bool = typer.Option(False, "--share", "-s", help="Create public Gradio link"),
):
    """🎨 Start the Gradio web UI for testing.

    Launches a web interface for uploading images and predicting tags.

    Can run in two modes:
    - Local: Uses the model directly (default)
    - Remote: Connects to an API server (use --api-url)
    """
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    mode = "Remote API" if api_url else "Local Model"
    print_header("🎨 Starting Gradio UI", f"Mode: {mode}")

    try:
        from pr0loader.api.gradio_ui import launch_gradio

        print_info(f"Web UI: http://{host}:{port}")
        if api_url:
            print_info(f"Using API: {api_url}")
        print_info("Press Ctrl+C to stop")
        console.print()

        launch_gradio(
            model_path=model,
            api_url=api_url,
            host=host,
            port=port,
            share=share,
        )

    except ImportError:
        print_error("Gradio not installed. Install with: pip install gradio")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Gradio UI failed: {e}")
        raise typer.Exit(1)


@app.command()
def serve(
    ctx: typer.Context,
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    api_port: int = typer.Option(8000, "--api-port", help="API server port"),
    ui_port: int = typer.Option(7860, "--ui-port", help="Gradio UI port"),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="Path to trained model"),
):
    """🚀 Start both API server and Gradio UI.

    Launches both the inference API and the web UI for testing.
    """
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    print_header("🚀 Starting Full Server", "API + Gradio UI")
    print_info(f"API Server: http://{host}:{api_port}")
    print_info(f"Web UI: http://{host}:{ui_port}")
    print_info("Press Ctrl+C to stop")
    console.print()

    import threading

    try:
        import uvicorn
        from pr0loader.api.server import create_app
        from pr0loader.api.gradio_ui import launch_gradio

        # Start API server in background thread
        api_app = create_app(model_path=model)

        def run_api():
            uvicorn.run(api_app, host=host, port=api_port, log_level="warning")

        api_thread = threading.Thread(target=run_api, daemon=True)
        api_thread.start()

        print_success("API server started")

        # Start Gradio in main thread (connects to API)
        api_url = f"http://{host}:{api_port}"
        launch_gradio(
            api_url=api_url,
            host=host,
            port=ui_port,
            share=False,
        )

    except ImportError as e:
        print_error(f"Missing dependencies: {e}")
        print_info("Install with: pip install fastapi uvicorn python-multipart gradio")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Server failed: {e}")
        raise typer.Exit(1)


@app.command()
def login(
    ctx: typer.Context,
    auto: bool = typer.Option(False, "--auto", "-a", help="Auto-detect from browsers"),
    browser: Optional[str] = typer.Option(None, "--browser", "-b", help="Extract from specific browser (firefox/chrome/edge/brave)"),
    interactive: bool = typer.Option(False, "--interactive", "-i", help="Interactive login with captcha"),
):
    """🔐 Login to pr0gramm.

    Authentication methods (tried in order):
    1. --auto: Try to extract cookies from installed browsers
    2. --browser: Extract from a specific browser
    3. --interactive: Manual login with captcha

    Without options, shows current auth status and available options.
    """
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    print_header("🔐 Authentication", "Login to pr0gramm")

    try:
        from pr0loader.auth import get_auth_manager, get_all_extractors

        auth = get_auth_manager()

        # Check current status first
        status = auth.get_status()

        if status.get("authenticated") or status.get("stored_credentials"):
            print_success(f"Already logged in as: {status.get('username', 'unknown')}")
            print_info(f"Verified (NSFW access): {'Yes' if status.get('verified') else 'No'}")

            if not (auto or browser or interactive):
                print_info("Use --interactive to re-login or 'pr0loader logout' to clear")
                return

        credentials = None

        # Method 1: Auto-detect from browsers
        if auto or (not browser and not interactive):
            print_info("Trying to extract cookies from browsers...")

            # Show available browsers
            available = status.get("available_browsers", [])
            if available:
                print_info(f"Available browsers: {', '.join(available)}")
            else:
                print_warning("No supported browsers found")

            credentials = auth.get_credentials(auto_login=True)

        # Method 2: Specific browser
        if browser and not credentials:
            print_info(f"Extracting from {browser}...")
            credentials = auth.extract_from_browser(browser)

        # Method 3: Interactive login
        if interactive and not credentials:
            print_info("Starting interactive login...")
            credentials = auth.login_interactive()

        # Result
        if credentials:
            print_success(f"Logged in as: {credentials.username}")
            print_info(f"User ID: {credentials.user_id}")
            print_info(f"Verified (NSFW access): {'Yes' if credentials.is_verified else 'No'}")
            print_success("Credentials saved securely")
        else:
            if not (auto or browser or interactive):
                # No method specified, show help
                print_info("\nLogin methods:")
                print_info("  pr0loader login --auto         Extract from any browser")
                print_info("  pr0loader login --browser firefox")
                print_info("  pr0loader login --interactive  Manual login with captcha")
            else:
                print_error("Login failed")
                raise typer.Exit(1)

    except ImportError as e:
        print_error(f"Missing dependencies: {e}")
        print_info("Some features may require: pip install pycryptodome keyring pillow")
        raise typer.Exit(1)
    except Exception as e:
        print_error(f"Login failed: {e}")
        raise typer.Exit(1)


@app.command()
def logout(ctx: typer.Context):
    """🚪 Logout and clear stored credentials."""
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    try:
        from pr0loader.auth import get_auth_manager

        auth = get_auth_manager()

        if auth.logout():
            print_success("Logged out successfully")
            print_info("Stored credentials have been cleared")
        else:
            print_warning("No credentials to clear")

    except Exception as e:
        print_error(f"Logout failed: {e}")
        raise typer.Exit(1)


@app.command("auth-status")
def auth_status(ctx: typer.Context):
    """📋 Show current authentication status."""
    headless = ctx.obj.get("headless", False)
    if not headless:
        show_banner()

    try:
        from pr0loader.auth import get_auth_manager

        auth = get_auth_manager()
        status = auth.get_status()

        # Create status table
        table = Table(title="Authentication Status", box=box.ROUNDED)
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        # Credential status
        if status.get("stored_credentials"):
            table.add_row("Status", "✓ Logged in")
            table.add_row("Username", status.get("username", "unknown"))
            table.add_row("User ID", str(status.get("user_id", "unknown")))
            table.add_row("NSFW Access", "✓ Yes" if status.get("verified") else "✗ No (not verified)")
        else:
            table.add_row("Status", "✗ Not logged in")

        # Available browsers
        browsers = status.get("available_browsers", [])
        if browsers:
            table.add_row("Available Browsers", ", ".join(browsers))
        else:
            table.add_row("Available Browsers", "None detected")

        console.print(table)

        if not status.get("stored_credentials"):
            console.print()
            print_info("Login with: pr0loader login --auto")

    except Exception as e:
        print_error(f"Error: {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    console.print(f"pr0loader version {__version__}")


if __name__ == "__main__":
    app()

