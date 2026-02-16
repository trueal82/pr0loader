"""Fetch pipeline - download metadata from pr0gramm API.

🚀 OPTIMIZED FETCH EDITION 🚀

Architecture:
┌──────────────────────────────────────────────────────────────────┐
│  1. GAP DETECTION (Fast, DataFrame-based)                        │
│     - Load all local IDs into pandas Series                      │
│     - Get remote max ID from API                                 │
│     - Compare: expected range vs actual IDs → missing IDs        │
│     Duration: ~5 seconds for 6M items                            │
├──────────────────────────────────────────────────────────────────┤
│  2. FETCH PHASE (Page-based API iteration)                       │
│     - Iterate pages from highest missing ID downward             │
│     - Skip ranges we already have (using gap info)               │
│     - Parallel metadata fetching for each page                   │
│     Duration: depends on gaps, ~100 items/sec typical            │
└──────────────────────────────────────────────────────────────────┘

Key optimizations:
- DataFrame-based gap detection (instant for millions of IDs)
- Skip already-fetched ranges during iteration
- Connection pooling with appropriate size
- Fibonacci backoff for rate limits
"""

from __future__ import annotations

import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

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


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GapAnalysis:
    """Result of analyzing gaps between local and remote data."""
    remote_max_id: int
    local_min_id: int
    local_max_id: int
    local_count: int

    # Gap counts
    top_gap: int      # New items at top (remote_max - local_max)
    bottom_gap: int   # Missing at bottom (local_min - 1)
    internal_gaps: int  # Holes in the middle

    # What to fetch
    missing_ids: Optional[pd.Series] = None  # Actual missing IDs if computed

    @property
    def total_missing(self) -> int:
        return self.top_gap + self.bottom_gap + self.internal_gaps

    @property
    def is_complete(self) -> bool:
        return self.total_missing == 0


# =============================================================================
# FAST GAP DETECTION
# =============================================================================

def load_local_ids(db_path: Path) -> pd.Series:
    """Load all item IDs from database into a pandas Series.

    Fast SQL query, returns sorted Series of IDs.
    """
    start = time.perf_counter()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT id FROM items ORDER BY id", conn)
    conn.close()

    elapsed = time.perf_counter() - start
    logger.debug(f"Loaded {len(df):,} IDs from DB in {elapsed:.2f}s")

    return df['id']


def analyze_gaps(local_ids: pd.Series, remote_max_id: int) -> GapAnalysis:
    """Analyze gaps between local data and remote availability.

    Uses set operations for fast comparison.
    """
    start = time.perf_counter()

    if len(local_ids) == 0:
        # No local data - everything is missing
        return GapAnalysis(
            remote_max_id=remote_max_id,
            local_min_id=0,
            local_max_id=0,
            local_count=0,
            top_gap=remote_max_id,
            bottom_gap=0,
            internal_gaps=0,
        )

    local_min = int(local_ids.min())
    local_max = int(local_ids.max())
    local_count = len(local_ids)

    # Calculate gaps
    top_gap = remote_max_id - local_max
    bottom_gap = local_min - 1

    # Internal gaps: expected count vs actual count in our range
    expected_in_range = local_max - local_min + 1
    internal_gaps = expected_in_range - local_count

    elapsed = time.perf_counter() - start
    logger.debug(f"Gap analysis completed in {elapsed:.2f}s")

    return GapAnalysis(
        remote_max_id=remote_max_id,
        local_min_id=local_min,
        local_max_id=local_max,
        local_count=local_count,
        top_gap=max(0, top_gap),
        bottom_gap=max(0, bottom_gap),
        internal_gaps=max(0, internal_gaps),
    )


def compute_missing_ids(local_ids: pd.Series, start_id: int, end_id: int) -> pd.Series:
    """Compute exact missing IDs in a range.

    Uses set difference for fast computation.
    Returns Series of missing IDs sorted descending (for top-down fetching).
    """
    start = time.perf_counter()

    # Expected IDs in range
    expected = pd.Series(range(end_id, start_id + 1))

    # Convert local_ids to set for O(1) lookup
    local_set = set(local_ids)

    # Find missing
    missing = expected[~expected.isin(local_set)]

    # Sort descending (fetch newest first)
    missing = missing.sort_values(ascending=False)

    elapsed = time.perf_counter() - start
    logger.debug(f"Computed {len(missing):,} missing IDs in {elapsed:.2f}s")

    return missing


# =============================================================================
# FETCH PIPELINE
# =============================================================================

class FetchPipeline:
    """Pipeline for fetching metadata from the pr0gramm API.

    Uses DataFrame-based gap detection for fast startup.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.api = APIClient(settings)
        self.stats = PipelineStats()

    def _fetch_item_info(self, item: Item) -> Optional[Item]:
        """Fetch detailed info (tags, comments) for a single item."""
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
        executor: ThreadPoolExecutor,
    ) -> list[Item]:
        """Fetch detailed info for multiple items in parallel."""
        successful = []

        futures = {
            executor.submit(self._fetch_item_info, item): item
            for item in items
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    successful.append(result)
                    self.stats.items_processed += 1
                else:
                    self.stats.items_failed += 1
            except Exception as e:
                logger.error(f"Exception during item info fetch: {e}")
                self.stats.items_failed += 1

        return successful

    def run(self, fill_gaps: bool = False) -> PipelineStats:
        """Run the fetch pipeline with DataFrame-based gap detection."""
        print_header(
            "📥 Fetch Metadata",
            "Fast gap detection + parallel fetching"
        )

        with SQLiteStorage(self.settings.db_path) as storage:
            # =================================================================
            # PHASE 1: GAP DETECTION
            # =================================================================
            print_info("Phase 1: Analyzing gaps...")
            phase1_start = time.perf_counter()

            # Get remote max ID
            remote_max_id = self.api.get_highest_id()
            print_info(f"Remote max ID: {remote_max_id:,}")

            # Load local IDs
            local_ids = load_local_ids(self.settings.db_path)
            print_info(f"Local items: {len(local_ids):,}")

            # Analyze gaps
            gaps = analyze_gaps(local_ids, remote_max_id)

            phase1_time = time.perf_counter() - phase1_start
            print_info(f"Gap analysis: {phase1_time:.1f}s")

            # Report gaps
            print_info(f"  Top gap (new items): {gaps.top_gap:,}")
            print_info(f"  Bottom gap (old items): {gaps.bottom_gap:,}")
            print_info(f"  Internal gaps: {gaps.internal_gaps:,}")
            print_info(f"  Total missing: {gaps.total_missing:,}")

            # Check if we need to fetch
            if gaps.is_complete:
                print_success("Database is complete - nothing to fetch!")
                print_stats_table("Fetch Results", {
                    "Items processed": 0,
                    "Total in database": gaps.local_count,
                    "Status": "Up to date",
                })
                return self.stats

            # Handle --full flag
            if self.settings.full_update:
                print_info("Full update requested - fetching all items")
                start_id = remote_max_id
                end_id = 1
            elif self.settings.start_from:
                print_info(f"Starting from ID {self.settings.start_from}")
                start_id = self.settings.start_from
                end_id = 1
            elif gaps.top_gap > 0:
                # Fetch new items at top first
                print_info(f"Fetching {gaps.top_gap:,} new items at top")
                start_id = remote_max_id
                end_id = gaps.local_max_id + 1
            elif gaps.bottom_gap > 0:
                # Then fill bottom
                print_info(f"Fetching {gaps.bottom_gap:,} items at bottom")
                start_id = gaps.local_min_id - 1
                end_id = 1
            elif gaps.internal_gaps > 0 and fill_gaps:
                # Fill internal gaps if requested
                print_info(f"Filling {gaps.internal_gaps:,} internal gaps")
                start_id = gaps.local_max_id
                end_id = gaps.local_min_id
            else:
                print_warning(f"Database has {gaps.internal_gaps:,} internal gaps")
                print_info("Use --fill-gaps to re-scan and fill them")
                print_stats_table("Fetch Results", {
                    "Items processed": 0,
                    "Total in database": gaps.local_count,
                    "Internal gaps": gaps.internal_gaps,
                    "Status": "Gaps exist but not filling",
                })
                return self.stats

            estimated_items = start_id - end_id + 1
            print_info(f"Will fetch IDs {start_id:,} down to {end_id:,} (~{estimated_items:,} items)")

            # =================================================================
            # PHASE 2: FETCH
            # =================================================================
            print_info("Phase 2: Fetching metadata...")

            # Create thread pool for parallel fetching
            executor = ThreadPoolExecutor(max_workers=self.settings.max_parallel_requests)

            current_id = start_id
            batch = []
            batch_size = self.settings.db_batch_size

            total_db_write_time = 0.0
            db_write_count = 0

            progress = create_progress("Fetching")
            fetch_start = time.perf_counter()

            try:
                with progress:
                    task = progress.add_task(
                        "[cyan]Fetching metadata...",
                        total=estimated_items,
                    )

                    # Pre-fetch first page
                    next_page_future: Optional[Future] = executor.submit(
                        self.api.get_items, current_id
                    )

                    while True:
                        try:
                            # Get current page
                            if next_page_future:
                                response = next_page_future.result()
                                next_page_future = None
                            else:
                                response = self.api.get_items(older_than=current_id)

                            if not response.items:
                                break

                            # Check if we should continue
                            next_id = response.items[-1].id if response.items else None
                            should_continue = (
                                next_id is not None and
                                next_id < current_id and
                                next_id > end_id
                            )

                            # Pre-fetch next page
                            if should_continue:
                                next_page_future = executor.submit(
                                    self.api.get_items, next_id
                                )

                            # Filter items we don't already have
                            local_set = set(local_ids)
                            new_items = [
                                item for item in response.items
                                if item.id not in local_set and item.id >= end_id
                            ]

                            if new_items:
                                # Fetch detailed info in parallel
                                processed = self._fetch_batch_info_parallel(new_items, executor)
                                batch.extend(processed)

                            # Update progress
                            progress.update(task, advance=len(response.items))

                            # Flush batch
                            if len(batch) >= batch_size:
                                write_start = time.time()
                                storage.upsert_items_batch(batch)
                                write_time = time.time() - write_start
                                total_db_write_time += write_time
                                db_write_count += 1
                                # Update local_ids with new items
                                new_ids = [item.id for item in batch]
                                local_ids = pd.concat([local_ids, pd.Series(new_ids)], ignore_index=True)
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

                    # Flush remaining
                    if batch:
                        write_start = time.time()
                        storage.upsert_items_batch(batch)
                        write_time = time.time() - write_start
                        total_db_write_time += write_time
                        db_write_count += 1

            finally:
                executor.shutdown(wait=True)

            fetch_time = time.perf_counter() - fetch_start

            # Optimize database
            print_info("Optimizing database...")
            storage.optimize_database()

            # Stats
            items_per_sec = self.stats.items_processed / fetch_time if fetch_time > 0 else 0

            print_stats_table("Fetch Results", {
                "Items processed": f"{self.stats.items_processed:,}",
                "Items failed": f"{self.stats.items_failed:,}",
                "Total in database": f"{storage.get_item_count():,}",
                "Gap analysis": f"{phase1_time:.1f}s",
                "Fetch time": f"{fetch_time:.1f}s",
                "Speed": f"{items_per_sec:.1f} items/sec",
                "DB writes": f"{db_write_count} batches ({total_db_write_time:.1f}s)",
            })

            print_success("Fetch complete!")

        return self.stats

