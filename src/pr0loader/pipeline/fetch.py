"""Fetch pipeline - download metadata from pr0gramm API.
🚀 FAST AS FUCK EDITION 🚀
Architecture (Simple is Fast):
┌──────────────────────────────────────────────────────────────────┐
│  1. GAP ANALYSIS (DataFrame-based, ~5 seconds)                   │
│     - Load local IDs: SQL → pandas Series                        │
│     - Get remote max: API call                                   │
│     - Find gaps: set operations → missing ID ranges              │
├──────────────────────────────────────────────────────────────────┤
│  2. FETCH PHASE (Async HTTP + parallel metadata)                 │
│     - Iterate API pages (already sorted newest→oldest)           │
│     - Skip IDs we already have (set membership)                  │
│     - Parallel fetch metadata (tags/comments)                    │
│     - Batch DB writes (commit every N items)                     │
└──────────────────────────────────────────────────────────────────┘
Key optimizations:
- No complex gap logic - just load IDs and compare
- Skip already-fetched items during iteration (set lookup O(1))
- Parallel metadata fetching with thread pool
- Batch DB writes to reduce SQLite overhead
- Connection pooling for HTTP efficiency
"""
from __future__ import annotations
import logging
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import pandas as pd
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
class FetchStats:
    """Statistics for fetch run."""
    remote_max_id: int = 0
    local_count: int = 0
    items_fetched: int = 0
    items_failed: int = 0
    items_skipped: int = 0  # Already had
    # Timing
    gap_analysis_time: float = 0.0
    fetch_time: float = 0.0
    db_write_time: float = 0.0
# =============================================================================
# FAST GAP ANALYSIS
# =============================================================================
def load_local_ids(db_path: Path) -> set[int]:
    """Load all item IDs from database into a set.
    Fast SQL query, returns set for O(1) membership checks.
    """
    start = time.perf_counter()
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT id FROM items", conn)
    conn.close()
    ids = set(df['id'].tolist())
    elapsed = time.perf_counter() - start
    logger.debug(f"Loaded {len(ids):,} IDs in {elapsed:.2f}s")
    return ids
def analyze_gaps(local_ids: set[int], remote_max_id: int) -> dict:
    """Analyze what needs to be fetched.
    Returns dict with gap info.
    """
    if not local_ids:
        return {
            'local_min': 0,
            'local_max': 0,
            'top_gap': remote_max_id,
            'bottom_gap': 0,
            'total_missing': remote_max_id,
            'is_complete': False,
        }
    local_min = min(local_ids)
    local_max = max(local_ids)
    top_gap = remote_max_id - local_max
    bottom_gap = local_min - 1
    # Rough estimate of internal gaps
    expected_count = local_max - local_min + 1
    internal_gaps = expected_count - len(local_ids)
    return {
        'local_min': local_min,
        'local_max': local_max,
        'top_gap': max(0, top_gap),
        'bottom_gap': max(0, bottom_gap),
        'internal_gaps': max(0, internal_gaps),
        'total_missing': max(0, top_gap) + max(0, bottom_gap) + max(0, internal_gaps),
        'is_complete': top_gap <= 0 and bottom_gap <= 0 and internal_gaps <= 0,
    }
# =============================================================================
# FETCH PIPELINE
# =============================================================================
class FetchPipeline:
    """Fast metadata fetch pipeline.
    Simple architecture:
    1. Load IDs → find what's missing
    2. Iterate API pages → skip what we have → fetch metadata
    3. Batch write to DB
    """
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api = APIClient(settings)
        self.stats = FetchStats()
    def _fetch_item_info(self, item: Item) -> Optional[Item]:
        """Fetch detailed info (tags, comments) for one item."""
        try:
            info = self.api.get_item_info(item.id)
            item.tags = info.tags
            item.comments = info.comments
            return item
        except Exception as e:
            logger.debug(f"Failed to fetch info for {item.id}: {e}")
            return None
    def run(self) -> PipelineStats:
        """Run the fetch pipeline."""
        print_header(
            "📥 Fetch Metadata",
            "Fast as fuck gap detection + parallel fetch"
        )

        # Get settings
        full_update = getattr(self.settings, 'full_update', False)
        fill_gaps = getattr(self.settings, 'fill_gaps', False)

        total_start = time.perf_counter()
        with SQLiteStorage(self.settings.db_path) as storage:
            # =================================================================
            # PHASE 1: GAP ANALYSIS
            # =================================================================
            print_info("Phase 1: Gap analysis...")
            phase1_start = time.perf_counter()
            # Get remote max
            self.stats.remote_max_id = self.api.get_highest_id()
            print_info(f"  Remote max ID: {self.stats.remote_max_id:,}")
            # Load local IDs
            local_ids = load_local_ids(self.settings.db_path)
            self.stats.local_count = len(local_ids)
            print_info(f"  Local items: {self.stats.local_count:,}")
            # Analyze gaps
            gaps = analyze_gaps(local_ids, self.stats.remote_max_id)
            self.stats.gap_analysis_time = time.perf_counter() - phase1_start
            print_info(f"  Analysis time: {self.stats.gap_analysis_time:.1f}s")
            # Report gaps
            if gaps['top_gap'] > 0:
                print_info(f"  → New items at top: {gaps['top_gap']:,}")
            if gaps['bottom_gap'] > 0:
                print_info(f"  → Missing at bottom: {gaps['bottom_gap']:,}")
            if gaps['internal_gaps'] > 0:
                print_info(f"  → Internal gaps: {gaps['internal_gaps']:,}")
            # Check if complete
            if gaps['is_complete'] and not full_update:
                print_success("Database is complete!")
                print_stats_table("Fetch Results", {
                    "Status": "Up to date",
                    "Items in DB": f"{self.stats.local_count:,}",
                    "Gap analysis": f"{self.stats.gap_analysis_time:.1f}s",
                })
                return self._to_pipeline_stats()
            # Determine fetch range
            if full_update or self.settings.full_update:
                print_info("Full update: fetching all items")
                start_id = self.stats.remote_max_id
                end_id = 1
            elif self.settings.start_from:
                print_info(f"Starting from ID {self.settings.start_from}")
                start_id = self.settings.start_from
                end_id = 1
            elif gaps['top_gap'] > 0:
                # Fetch new items first
                print_info(f"Fetching {gaps['top_gap']:,} new items at top")
                start_id = self.stats.remote_max_id
                end_id = gaps['local_max'] + 1
            elif gaps['bottom_gap'] > 0:
                # Then old items
                print_info(f"Fetching {gaps['bottom_gap']:,} items at bottom")
                start_id = gaps['local_min'] - 1
                end_id = 1
            elif gaps['internal_gaps'] > 0 and fill_gaps:
                # Fill internal gaps
                print_info(f"Filling {gaps['internal_gaps']:,} internal gaps")
                start_id = gaps['local_max']
                end_id = gaps['local_min']
            else:
                if gaps['internal_gaps'] > 0:
                    print_warning(f"Database has {gaps['internal_gaps']:,} internal gaps")
                    print_info("Use --fill-gaps to re-scan")
                print_success("Nothing to fetch (gaps exist but not filling)")
                return self._to_pipeline_stats()
            estimated = start_id - end_id + 1
            print_info(f"Will fetch: IDs {start_id:,} → {end_id:,} (~{estimated:,} items)")
            # =================================================================
            # PHASE 2: FETCH
            # =================================================================
            print_info("Phase 2: Fetching...")
            phase2_start = time.perf_counter()
            # Create thread pool for parallel metadata fetching
            executor = ThreadPoolExecutor(max_workers=self.settings.max_parallel_requests)
            batch = []
            batch_size = self.settings.db_batch_size
            current_id = start_id
            progress = create_progress("Fetching")
            try:
                with progress:
                    task = progress.add_task(
                        "[cyan]Fetching...",
                        total=estimated
                    )
                    while current_id >= end_id:
                        try:
                            # Get page from API
                            response = self.api.get_items(older_than=current_id)
                            if not response.items:
                                break
                            # Filter: skip items we already have
                            new_items = [
                                item for item in response.items
                                if item.id not in local_ids and item.id >= end_id
                            ]
                            self.stats.items_skipped += len(response.items) - len(new_items)
                            if new_items:
                                # Fetch metadata in parallel
                                futures = [
                                    executor.submit(self._fetch_item_info, item)
                                    for item in new_items
                                ]
                                for future in futures:
                                    result = future.result()
                                    if result:
                                        batch.append(result)
                                        local_ids.add(result.id)  # Update set
                                        self.stats.items_fetched += 1
                                    else:
                                        self.stats.items_failed += 1
                            # Update progress
                            progress.update(task, advance=len(response.items))
                            # Flush batch to DB
                            if len(batch) >= batch_size:
                                write_start = time.time()
                                storage.upsert_items_batch(batch)
                                self.stats.db_write_time += time.time() - write_start
                                batch = []
                            # Next page
                            if response.items:
                                next_id = response.items[-1].id
                                if next_id >= current_id or next_id < end_id:
                                    break
                                current_id = next_id
                            else:
                                break
                        except KeyboardInterrupt:
                            print_warning("\nInterrupted by user")
                            break
                        except Exception as e:
                            logger.error(f"Error during fetch: {e}")
                            break
                    # Flush remaining
                    if batch:
                        write_start = time.time()
                        storage.upsert_items_batch(batch)
                        self.stats.db_write_time += time.time() - write_start
            finally:
                executor.shutdown(wait=True)
            self.stats.fetch_time = time.perf_counter() - phase2_start
            # Optimize DB
            if self.stats.items_fetched > 0:
                print_info("Optimizing database...")
                storage.optimize_database()
            # =================================================================
            # DONE
            # =================================================================
            total_time = time.perf_counter() - total_start
            items_per_sec = self.stats.items_fetched / self.stats.fetch_time if self.stats.fetch_time > 0 else 0
            print_stats_table("Fetch Results", {
                "Fetched": f"{self.stats.items_fetched:,}",
                "Skipped": f"{self.stats.items_skipped:,}",
                "Failed": f"{self.stats.items_failed:,}",
                "Total in DB": f"{storage.get_item_count():,}",
                "Gap analysis": f"{self.stats.gap_analysis_time:.1f}s",
                "Fetch time": f"{self.stats.fetch_time:.1f}s",
                "Speed": f"{items_per_sec:.1f} items/s",
                "DB writes": f"{self.stats.db_write_time:.1f}s",
                "Total": f"{total_time:.1f}s",
            })
            print_success("Fetch complete! 🎉")
        return self._to_pipeline_stats()
    def _to_pipeline_stats(self) -> PipelineStats:
        """Convert to PipelineStats for compatibility."""
        stats = PipelineStats()
        stats.items_processed = self.stats.items_fetched
        stats.items_failed = self.stats.items_failed
        return stats
