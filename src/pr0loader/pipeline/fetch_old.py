"""Fetch pipeline - download metadata from pr0gramm API."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
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
            return item
        except Exception as e:
            logger.warning(f"Failed to fetch info for item {item.id}: {e}")
            return None

    def _fetch_batch_info_parallel(
        self,
        items: list[Item],
        executor: Optional[ThreadPoolExecutor] = None
    ) -> list[Item]:
        """
        Fetch detailed info for multiple items in parallel.

        Args:
            items: List of items to fetch info for
            executor: Optional executor to reuse (creates one if not provided)

        Returns:
            List of successfully processed items with tags and comments
        """
        successful_items = []
        max_workers = self.settings.max_parallel_requests
        own_executor = executor is None


        if own_executor:
            executor = ThreadPoolExecutor(max_workers=max_workers)

        try:
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
        finally:
            if own_executor:
                executor.shutdown(wait=True)

        return successful_items

    def determine_id_range(self, storage: SQLiteStorage, fill_gaps: bool = False) -> tuple[int, int]:
        """
        Determine the range of IDs to fetch.

        Priority:
        1. If --full-update: fetch everything
        2. If --start-from: start from that ID
        3. If database is empty: fetch from highest remote ID down to 1
        4. If there are gaps at the bottom (min_id > 1): fill from min_id down to 1
        5. If there are new items at the top: fetch from highest_remote down to max_db_id
        6. Otherwise: nothing to fetch (unless fill_gaps is True)

        Returns:
            Tuple of (start_id, end_id) where we fetch from start_id DOWN to end_id
        """
        highest_remote_id = self.api.get_highest_id()

        if self.settings.full_update:
            logger.info("Full update enabled, fetching all items")
            return highest_remote_id, 1

        if self.settings.start_from:
            logger.info(f"Starting from ID {self.settings.start_from}")
            return self.settings.start_from, 1

        min_db_id = storage.get_min_id()
        max_db_id = storage.get_max_id()
        item_count = storage.get_item_count()

        logger.info(f"Local DB: min_id={min_db_id}, max_id={max_db_id}, count={item_count}")
        logger.info(f"Remote: highest_id={highest_remote_id}")

        if max_db_id == -1 or min_db_id == -1:
            # No local data - fetch everything
            logger.info("No local data, fetching all items")
            return highest_remote_id, 1

        # Calculate expected vs actual count to detect gaps
        expected_count = max_db_id - min_db_id + 1
        actual_count = item_count
        internal_gap_count = expected_count - actual_count

        # Check for gaps at the bottom (min_id > 1)
        bottom_gap = min_db_id - 1

        # Check for new items at the top
        top_gap = highest_remote_id - max_db_id

        logger.info(f"Gap analysis: bottom_gap={bottom_gap}, top_gap={top_gap}, internal_gaps={internal_gap_count}")

        # Priority 1: Fill bottom gap first (oldest items)
        if bottom_gap > 0:
            logger.info(f"Filling bottom gap: fetching from {min_db_id - 1} down to 1")
            print_info(f"Found {bottom_gap:,} missing items at bottom (IDs 1 to {min_db_id - 1})")
            return min_db_id - 1, 1

        # Priority 2: Fetch new items at top
        if top_gap > 0:
            logger.info(f"Fetching new items: from {highest_remote_id} down to {max_db_id + 1}")
            print_info(f"Found {top_gap:,} new items at top (IDs {max_db_id + 1} to {highest_remote_id})")
            return highest_remote_id, max_db_id + 1

        # Priority 3: Fill internal gaps if requested
        if internal_gap_count > 0:
            print_warning(f"Database has ~{internal_gap_count:,} internal gaps (deleted/missing items)")
            if fill_gaps:
                # Re-fetch the entire range to fill gaps
                # (The API will skip items that don't exist)
                logger.info(f"Filling internal gaps: fetching from {max_db_id} down to {min_db_id}")
                print_info(f"Re-scanning range {min_db_id} to {max_db_id} to fill gaps...")
                return max_db_id, min_db_id
            else:
                print_info("Use --full to re-fetch and fill internal gaps")

        # No gaps detected (or gaps exist but fill_gaps=False)
        logger.info("Database appears complete, nothing to fetch")
        print_info("Database is up to date, no new items to fetch")
        return -1, -1  # Signal nothing to fetch

    def run(self) -> PipelineStats:
        """Run the fetch pipeline with optimized parallel fetching."""
        print_header(
            "📥 Fetch Metadata",
            "Downloading item metadata from pr0gramm API"
        )

        with SQLiteStorage(self.settings.db_path) as storage:
            start_id, end_id = self.determine_id_range(storage)

            # Handle case where nothing to fetch
            if start_id == -1 and end_id == -1:
                print_stats_table("Fetch Results", {
                    "Items processed": 0,
                    "Items failed": 0,
                    "Total in database": storage.get_item_count(),
                    "Status": "Up to date",
                })
                print_success("Database is already up to date!")
                return self.stats

            estimated_items = start_id - end_id + 1

            print_info(f"Fetching items from ID {start_id} down to {end_id}")
            print_info(f"Estimated items: ~{estimated_items:,}")
            print_info(f"Parallel workers: {self.settings.max_parallel_requests}")

            current_id = start_id
            batch = []  # Accumulate items for batch insert
            batch_size = self.settings.db_batch_size

            # Track performance
            total_db_write_time = 0.0
            db_write_count = 0

            logger.info(f"Using batch size: {batch_size} items per commit")
            logger.info(f"Using {self.settings.max_parallel_requests} parallel workers for metadata fetching")
            print_info(f"DB batch size: {batch_size} items")

            # Use a persistent thread pool for the entire fetch operation
            executor = ThreadPoolExecutor(max_workers=self.settings.max_parallel_requests)

            progress = create_progress("Fetching")
            fetch_start_time = time.time()

            try:
                with progress:
                    task = progress.add_task(
                        "[cyan]Fetching metadata...",
                        total=estimated_items
                    )

                    # Pre-fetch first page
                    next_page_future: Optional[Future] = executor.submit(
                        self.api.get_items, current_id
                    )

                    while True:
                        try:
                            # Get current page (from pre-fetched future or fetch now)
                            if next_page_future:
                                response = next_page_future.result()
                                next_page_future = None
                            else:
                                response = self.api.get_items(older_than=current_id)

                            if not response.items:
                                logger.info("No more items to fetch")
                                break

                            # Determine next page ID for pre-fetching
                            next_id = response.items[-1].id if response.items else None
                            should_continue = (
                                next_id is not None and
                                next_id < current_id and
                                next_id > end_id
                            )

                            # Pre-fetch next page while we process current page
                            if should_continue:
                                next_page_future = executor.submit(
                                    self.api.get_items, next_id
                                )

                            # Fetch detailed info (tags, comments) for all items in parallel
                            processed_items = self._fetch_batch_info_parallel(response.items, executor)

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
                                batch = []

                            if not should_continue:
                                break

                            current_id = next_id

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

            finally:
                # Shutdown executor
                executor.shutdown(wait=True)

            fetch_duration = time.time() - fetch_start_time

            # Optimize database after bulk inserts
            print_info("Optimizing database...")
            storage.optimize_database()

            # Calculate and log performance stats
            items_per_second = self.stats.items_processed / fetch_duration if fetch_duration > 0 else 0
            logger.info(f"Fetch performance: {self.stats.items_processed} items in {fetch_duration:.1f}s "
                       f"({items_per_second:.1f} items/sec)")

            # Print final stats
            print_stats_table("Fetch Results", {
                "Items processed": self.stats.items_processed,
                "Items failed": self.stats.items_failed,
                "Total in database": storage.get_item_count(),
                "Duration": f"{fetch_duration:.1f}s",
                "Speed": f"{items_per_second:.1f} items/sec",
                "DB writes": f"{db_write_count} batches ({total_db_write_time:.1f}s)",
            })

            print_success("Fetch complete!")

        return self.stats

