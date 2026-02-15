"""Sync pipeline - full update: fetch metadata and download assets in one run.

This pipeline composes FetchPipeline and DownloadPipeline following DRY principles.
"""

import logging
import time

from pr0loader.config import Settings
from pr0loader.models import PipelineStats
from pr0loader.pipeline.download import DownloadPipeline
from pr0loader.pipeline.fetch import FetchPipeline
from pr0loader.utils.console import (
    print_header,
    print_stats_table,
    print_success,
    print_info,
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
    """
    Pipeline stage for full sync: fetch metadata + download assets.

    This is a composite pipeline that orchestrates:
    1. FetchPipeline - to fetch/update metadata
    2. DownloadPipeline - to download media files

    Using composition instead of duplication follows DRY principles.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.stats = PipelineStats()

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
            verify_existing: Verify existing files with HEAD request (for download phase)
            download_media: Whether to download media files after fetching metadata
        """
        print_header(
            "🔄 Full Sync",
            "Fetching metadata and downloading assets"
        )

        sync_start_time = time.time()

        # Phase 1: Fetch metadata using FetchPipeline
        print_info("Phase 1: Fetching metadata...")
        fetch_pipeline = FetchPipeline(self.settings)
        fetch_stats = fetch_pipeline.run()

        # Accumulate stats
        self.stats.items_processed = fetch_stats.items_processed
        self.stats.items_failed = fetch_stats.items_failed

        # Phase 2: Download media using DownloadPipeline (if enabled)
        if download_media:
            print_info("Phase 2: Downloading media...")
            download_pipeline = DownloadPipeline(self.settings)
            download_stats = download_pipeline.run(
                include_videos=include_videos,
                verify_existing=verify_existing,
            )

            # Accumulate download stats
            self.stats.files_downloaded = download_stats.files_downloaded
            self.stats.files_skipped = download_stats.files_skipped
            self.stats.bytes_downloaded = download_stats.bytes_downloaded
            self.stats.items_failed += download_stats.items_failed

        sync_duration = time.time() - sync_start_time

        # Print combined summary
        print_header("🔄 Sync Summary", "Combined results")

        stats_dict = {
            "Total duration": f"{sync_duration:.1f}s",
            "Items fetched": self.stats.items_processed,
            "Fetch failures": fetch_stats.items_failed,
        }

        if download_media:
            stats_dict.update({
                "Files downloaded": self.stats.files_downloaded,
                "Files skipped": self.stats.files_skipped,
                "Download failures": download_stats.items_failed,
                "Total bytes": format_bytes(self.stats.bytes_downloaded),
            })

        print_stats_table("Sync Results", stats_dict)
        print_success("Sync complete!")

        return self.stats

