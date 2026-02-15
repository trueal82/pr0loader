"""Fetch pipeline - download metadata from pr0gramm API."""

import logging
import time

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


class FetchPipeline:
    """Pipeline stage for fetching metadata from the API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api = APIClient(settings)
        self.stats = PipelineStats()

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
            # No local data
            return highest_remote_id, 1
        elif min_db_id != 1:
            # Continue from where we left off
            return min_db_id, 1
        else:
            # Fetch new items
            return highest_remote_id, max_db_id

    def run(self) -> PipelineStats:
        """Run the fetch pipeline."""
        print_header(
            "📥 Fetch Metadata",
            "Downloading item metadata from pr0gramm API"
        )

        with SQLiteStorage(self.settings.db_path) as storage:
            start_id, end_id = self.determine_id_range(storage)
            estimated_items = start_id - end_id

            print_info(f"Fetching items from ID {start_id} down to {end_id}")
            print_info(f"Estimated items: ~{estimated_items:,}")

            current_id = start_id
            batch = []  # Accumulate items for batch insert
            batch_size = self.settings.db_batch_size

            logger.info(f"Using batch size: {batch_size} items per commit")
            print_info(f"DB batch size: {batch_size} items")

            progress = create_progress("Fetching")
            with progress:
                task = progress.add_task(
                    "[cyan]Fetching metadata...",
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
                            try:
                                # Fetch detailed info (tags, comments)
                                info = self.api.get_item_info(item.id)
                                item.tags = info.tags
                                item.comments = info.comments

                                # Verbose logging: show item details
                                if logger.isEnabledFor(logging.DEBUG):
                                    tag_names = [t.tag for t in item.tags[:5]]  # First 5 tags
                                    tag_str = ", ".join(tag_names) if tag_names else "no tags"
                                    logger.debug(
                                        f"Item {item.id}: {item.image} | "
                                        f"👍{item.up} 👎{item.down} | "
                                        f"Tags: {tag_str}"
                                    )

                                # Add to batch
                                batch.append(item)
                                self.stats.items_processed += 1
                                progress.update(task, advance=1)

                                # Flush batch to database when it reaches batch_size
                                if len(batch) >= batch_size:
                                    storage.upsert_items_batch(batch)
                                    logger.debug(f"Flushed batch of {len(batch)} items to DB")
                                    batch = []

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
                        logger.error(f"Error during fetch: {e}")
                        self.stats.items_failed += 1

                # Flush any remaining items in the batch
                if batch:
                    storage.upsert_items_batch(batch)
                    logger.debug(f"Flushed final batch of {len(batch)} items to DB")

            # Print final stats
            print_stats_table("Fetch Results", {
                "Items processed": self.stats.items_processed,
                "Items failed": self.stats.items_failed,
                "Total in database": storage.get_item_count(),
            })

            print_success("Fetch complete!")

        return self.stats

