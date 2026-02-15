"""Fetch pipeline - download metadata from pr0gramm API."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from pr0loader.api import APIClient
from pr0loader.config import Settings
from pr0loader.models import PipelineStats, Item
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


class FetchPipeline:
    """Pipeline stage for fetching metadata from the API."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api = APIClient(settings)
        self.stats = PipelineStats()

    def _fetch_item_info(self, item: Item) -> Optional[Item]:
        """
        Fetch detailed info (tags, comments) for a single item.

        Returns:
            Item with populated tags and comments, or None if failed.
        """
        try:
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

            return item
        except Exception as e:
            logger.error(f"Failed to fetch info for item {item.id}: {e}")
            return None

    def _fetch_batch_info_parallel(self, items: list[Item]) -> list[Item]:
        """
        Fetch detailed info for multiple items in parallel.

        Args:
            items: List of items to fetch info for

        Returns:
            List of successfully processed items with tags and comments
        """
        successful_items = []
        max_workers = self.settings.max_parallel_requests

        logger.debug(f"Fetching info for {len(items)} items with {max_workers} parallel workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fetch tasks
            future_to_item = {
                executor.submit(self._fetch_item_info, item): item
                for item in items
            }

            # Process completed tasks as they finish
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result is not None:
                        successful_items.append(result)
                        self.stats.items_processed += 1
                    else:
                        self.stats.items_failed += 1
                except Exception as e:
                    logger.error(f"Exception while processing item {item.id}: {e}")
                    self.stats.items_failed += 1

        return successful_items

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
            print_info(f"Parallel workers: {self.settings.max_parallel_requests}")

            current_id = start_id
            batch = []  # Accumulate items for batch insert
            batch_size = self.settings.db_batch_size

            # Track database write performance
            total_db_write_time = 0.0
            db_write_count = 0

            logger.info(f"Using batch size: {batch_size} items per commit")
            logger.info(f"Using {self.settings.max_parallel_requests} parallel workers for metadata fetching")
            print_info(f"DB batch size: {batch_size} items")

            progress = create_progress("Fetching")
            with progress:
                task = progress.add_task(
                    "[cyan]Fetching metadata...",
                    total=estimated_items
                )

                while True:
                    try:
                        # Fetch batch of items (basic info only)
                        response = self.api.get_items(older_than=current_id)

                        if not response.items:
                            logger.info("No more items to fetch")
                            break

                        # Fetch detailed info (tags, comments) for all items in parallel
                        logger.debug(f"Fetching detailed info for {len(response.items)} items in parallel")
                        processed_items = self._fetch_batch_info_parallel(response.items)

                        # Add successfully processed items to batch
                        batch.extend(processed_items)

                        # Update progress
                        progress.update(task, advance=len(response.items))

                        # Flush batch to database when it reaches batch_size
                        if len(batch) >= batch_size:
                            write_start = time.time()
                            storage.upsert_items_batch(batch)
                            write_time = time.time() - write_start
                            total_db_write_time += write_time
                            db_write_count += 1

                            logger.debug(f"Flushed batch of {len(batch)} items to DB in {write_time:.2f}s")
                            batch = []


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
                    write_start = time.time()
                    storage.upsert_items_batch(batch)
                    write_time = time.time() - write_start
                    total_db_write_time += write_time
                    db_write_count += 1
                    logger.debug(f"Flushed final batch of {len(batch)} items to DB in {write_time:.2f}s")

            # Optimize database after bulk inserts
            print_info("Optimizing database...")
            storage.optimize_database()

            # Calculate and log performance stats
            avg_write_time = total_db_write_time / db_write_count if db_write_count > 0 else 0
            logger.info(f"Database write performance: {db_write_count} batches, "
                       f"total {total_db_write_time:.2f}s, avg {avg_write_time:.2f}s per batch")

            # Print final stats
            print_stats_table("Fetch Results", {
                "Items processed": self.stats.items_processed,
                "Items failed": self.stats.items_failed,
                "Total in database": storage.get_item_count(),
                "DB write time": f"{total_db_write_time:.1f}s ({db_write_count} batches)",
            })

            print_success("Fetch complete!")

        return self.stats

