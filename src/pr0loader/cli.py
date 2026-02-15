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

    action, options = run_interactive_menu()

    if action is None:
        console.print("\n[dim]Goodbye![/dim]")
        raise typer.Exit(0)

    # Execute the selected action with options
    try:
        settings = load_settings()

        if action == "info":
            _execute_info(ctx)
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
        elif action == "run-all":
            _execute_run_all(settings, options)

    except Exception as e:
        print_error(f"Error: {e}")
        raise typer.Exit(1)


def _execute_info(ctx: typer.Context):
    """Execute info command."""
    ctx.invoke(info)


def _execute_sync(settings: Settings, options: dict):
    """Execute sync with options from interactive menu."""
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
    if options.get("full"):
        settings.full_update = True
    if options.get("start_from"):
        settings.start_from = options["start_from"]

    from pr0loader.pipeline import FetchPipeline
    pipeline = FetchPipeline(settings)
    pipeline.run()


def _execute_download(settings: Settings, options: dict):
    """Execute download with options from interactive menu."""
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

        # Configuration table
        config_table = Table(title="Configuration", box=box.ROUNDED)
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="green")

        config_table.add_row("Database path", str(settings.db_path))
        config_table.add_row("Media directory", str(settings.filesystem_prefix))
        config_table.add_row("Model path", str(settings.model_path))
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
        print_info("Make sure you have a .env file with required settings (PP, ME)")
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
def version():
    """Show version information."""
    console.print(f"pr0loader version {__version__}")


if __name__ == "__main__":
    app()

