"""Sync pipeline - full update: fetch metadata and download assets in one run."""

import logging
import time
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
)

logger = logging.getLogger(__name__)


def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


class SyncPipeline:
    """Pipeline stage for full sync: fetch metadata + download assets."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api = APIClient(settings)
        self.stats = PipelineStats()
        self.files_verified = 0
        self.files_redownloaded = 0

    def determine_id_range(self, storage: SQLiteStorage) -> tuple[int, int]:
        """Determine the range of IDs to fetch."""
        highest_remote_id = self.api.get_highest_id()

        if self.settings.full_update:
            logger.info("Full update enabled, fetching all items")
            return highest_remote_id, 1

        if self.settings.start_from:
            logger.info(f"Starting from ID {self.settings.start_from}")
            return self.settings.start_from, 1

        min_db_id = storage.get_min_id()
        max_db_id = storage.get_max_id()

        logger.info(f"Local DB: min_id={min_db_id}, max_id={max_db_id}")
        logger.info(f"Remote: highest_id={highest_remote_id}")

        if max_db_id == -1 or min_db_id == -1:
            return highest_remote_id, 1
        elif min_db_id != 1:
            return min_db_id, 1
        else:
            return highest_remote_id, max_db_id

    def should_download(self, filename: str, destination: Path, verify: bool = True) -> bool:
        """
        Check if file needs to be downloaded.

        Args:
            filename: Remote filename
            destination: Local file path
            verify: If True, verify existing files with HEAD request

        Returns:
            True if file should be downloaded.
        """
        if not destination.exists():
            return True

        if verify:
            return self.api.needs_download(filename, destination)

        return False

    def run(
        self,
        include_videos: bool = False,
        verify_existing: bool = True,
        download_media: bool = True,
    ) -> PipelineStats:
        """
        Run the full sync pipeline.

        Args:
            include_videos: If True, also download videos. Default is images only.
            verify_existing: Verify existing files with HEAD request
            download_media: Whether to download media files
        """
        print_header(
            "🔄 Full Sync",
            "Fetching metadata and downloading assets"
        )

        media_dir = self.settings.filesystem_prefix
        media_dir.mkdir(parents=True, exist_ok=True)
        print_info(f"Media directory: {media_dir}")
        print_info(f"Verify existing files: {'Yes' if verify_existing else 'No'}")
        if download_media:
            print_info(f"Download mode: {'All media (images + videos)' if include_videos else 'Images only'}")

        with SQLiteStorage(self.settings.db_path) as storage:
            start_id, end_id = self.determine_id_range(storage)
            estimated_items = start_id - end_id

            print_info(f"Syncing items from ID {start_id} down to {end_id}")
            print_info(f"Estimated items: ~{estimated_items:,}")

            current_id = start_id

            progress = create_progress("Syncing")
            with progress:
                task = progress.add_task(
                    "[cyan]Syncing...",
                    total=estimated_items
                )

                while True:
                    try:
                        # Fetch batch of items
                        response = self.api.get_items(older_than=current_id)

                        if not response.items:
                            logger.info("No more items to fetch")
                            break

                        # Process each item
                        for item in response.items:
                            progress.update(task, advance=1)

                            try:
                                # Fetch detailed info (tags, comments)
                                info = self.api.get_item_info(item.id)
                                item.tags = info.tags
                                item.comments = info.comments

                                # Store in database
                                storage.upsert_item(item)
                                self.stats.items_processed += 1

                                # Download media if enabled
                                if download_media:
                                    # Skip videos unless explicitly included
                                    if not include_videos:
                                        ext = Path(item.image).suffix.lower()
                                        if ext not in {'.jpg', '.jpeg', '.png', '.gif'}:
                                            self.stats.items_skipped += 1
                                            continue

                                    destination = media_dir / item.image

                                    if self.should_download(item.image, destination, verify_existing):
                                        was_redownload = destination.exists()

                                        try:
                                            size = self.api.download_media(item.image, destination)
                                            self.stats.files_downloaded += 1
                                            self.stats.bytes_downloaded += size

                                            if was_redownload:
                                                self.files_redownloaded += 1

                                        except Exception as e:
                                            logger.error(f"Failed to download {item.image}: {e}")
                                            self.stats.items_failed += 1
                                    else:
                                        self.stats.files_skipped += 1
                                        if verify_existing and destination.exists():
                                            self.files_verified += 1

                            except Exception as e:
                                logger.error(f"Failed to process item {item.id}: {e}")
                                self.stats.items_failed += 1

                        # Get next page
                        next_id = response.items[-1].id if response.items else None
                        if next_id is None or next_id >= current_id or next_id <= end_id:
                            break

                        current_id = next_id
                        time.sleep(self.settings.request_delay)

                    except KeyboardInterrupt:
                        print_warning("Interrupted by user")
                        break
                    except Exception as e:
                        logger.error(f"Error during sync: {e}")
                        self.stats.items_failed += 1

            # Print final stats
            stats_dict = {
                "Items processed": self.stats.items_processed,
                "Items failed": self.stats.items_failed,
                "Total in database": storage.get_item_count(),
            }

            if download_media:
                stats_dict.update({
                    "Files downloaded": self.stats.files_downloaded,
                    "Files re-downloaded": self.files_redownloaded,
                    "Files verified OK": self.files_verified,
                    "Files skipped": self.stats.files_skipped,
                    "Total bytes": format_bytes(self.stats.bytes_downloaded),
                })

            print_stats_table("Sync Results", stats_dict)
            print_success("Sync complete!")

        return self.stats

