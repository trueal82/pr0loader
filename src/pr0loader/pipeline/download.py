"""Download pipeline - download media files."""

import logging
from pathlib import Path

from pr0loader.api import APIClient
from pr0loader.config import Settings
from pr0loader.models import PipelineStats
from pr0loader.storage import SQLiteStorage
from pr0loader.utils.console import (
    create_progress,
    print_header,
    print_stats_table,
    print_success,
    print_info,
    print_warning,
    is_headless,
)

logger = logging.getLogger(__name__)


def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


class DownloadPipeline:
    """Pipeline stage for downloading media files."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api = APIClient(settings)
        self.stats = PipelineStats()

    def run(self, include_videos: bool = False, verify_existing: bool = False) -> PipelineStats:
        """
        Run the download pipeline.

        Args:
            include_videos: If True, also download videos. Default is images only.
            verify_existing: If True, verify existing files via HEAD request and
                           re-download if size differs. Default is False (skip existing).
        """
        print_header(
            "📁 Download Media",
            "Downloading media files from pr0gramm"
        )

        # Ensure media directory exists
        media_dir = self.settings.filesystem_prefix
        media_dir.mkdir(parents=True, exist_ok=True)
        print_info(f"Media directory: {media_dir}")
        print_info(f"Download mode: {'All media (images + videos)' if include_videos else 'Images only'}")
        print_info(f"Verify existing: {'Yes (HEAD check)' if verify_existing else 'No (skip existing)'}")

        with SQLiteStorage(self.settings.db_path) as storage:
            total_items = storage.get_item_count()
            print_info(f"Items in database: {total_items:,}")

            progress = create_progress("Downloading")
            with progress:
                task = progress.add_task(
                    "[cyan]Downloading media...",
                    total=total_items
                )

                items_processed = 0
                for item in storage.iter_items():
                    progress.update(task, advance=1)
                    items_processed += 1

                    # Verbose: periodic progress updates
                    if logger.isEnabledFor(logging.DEBUG) and items_processed % 100 == 0:
                        logger.debug(
                            f"Progress: {items_processed}/{total_items} | "
                            f"Downloaded: {self.stats.files_downloaded}, "
                            f"Skipped: {self.stats.files_skipped}, "
                            f"Failed: {self.stats.items_failed}"
                        )

                    # Skip videos unless explicitly included
                    if not include_videos:
                        ext = Path(item.image).suffix.lower()
                        if ext not in {'.jpg', '.jpeg', '.png', '.gif'}:
                            self.stats.items_skipped += 1
                            continue

                    # Build paths
                    destination = media_dir / item.image

                    # Check if we need to download
                    if destination.exists():
                        if verify_existing:
                            # Verify via HEAD request
                            if not self.api.needs_download(item.image, destination):
                                self.stats.files_skipped += 1
                                continue
                            # Size mismatch - will re-download
                        else:
                            # Skip existing files without verification
                            self.stats.files_skipped += 1
                            continue

                    try:
                        # Download file
                        size = self.api.download_media(item.image, destination)
                        self.stats.files_downloaded += 1
                        self.stats.bytes_downloaded += size

                    except Exception as e:
                        logger.error(f"Failed to download {item.image}: {e}")
                        self.stats.items_failed += 1

            # Print final stats
            print_stats_table("Download Results", {
                "Files downloaded": self.stats.files_downloaded,
                "Files skipped (exist)": self.stats.files_skipped,
                "Items skipped (videos)": self.stats.items_skipped,
                "Failed": self.stats.items_failed,
                "Total bytes": format_bytes(self.stats.bytes_downloaded),
            })

            print_success("Download complete!")

        return self.stats

