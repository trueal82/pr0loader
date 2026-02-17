"""Download pipeline - download media files.

🚀 FAST AS FUCK EDITION 🚀

Architecture (Simple is Fast):
┌──────────────────────────────────────────────────────────────────┐
│  1. LOAD PHASE (Thread Pool)                                     │
│     - Load DB into DataFrame (pandas SQL query)                  │
│     - Scan filesystem into set (os.scandir)                      │
│     - Compare in memory → get download list                      │
│     Duration: ~30 seconds for 6M items                           │
├──────────────────────────────────────────────────────────────────┤
│  2. DOWNLOAD PHASE (Async I/O)                                   │
│     - Token bucket rate limiter (5 req/s for auth content)       │
│     - N concurrent workers with aiohttp                          │
│     - Progress bar updates in real-time                          │
│     Duration: ~5 files/sec for authenticated (rate limited)      │
└──────────────────────────────────────────────────────────────────┘

Key optimizations:
- No item-by-item iteration (vectorized pandas ops)
- No complex queue patterns (simple list → workers)
- No blocking in event loop (all heavy work in executor)
- Minimal object creation (work with raw data)
- Connection pooling with keep-alive
- Fibonacci backoff for rate limits (be nice to server)

Rate Limiting (tested 2026-02-17):
- pr0gramm WAF has DIFFERENT limits for auth vs unauth
- Unauthenticated (SFW): ~15 req/s OK
- Authenticated (NSFW/NSFL): ~5 req/s is the limit!
- On 429: ALL requests blocked for 60-120s cooldown
- Strategy: Proactive token bucket rate limiter at 5 req/s
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import signal
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import aiohttp
import pandas as pd
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

from pr0loader.config import Settings
from pr0loader.models import PipelineStats
from pr0loader.utils.backoff import fibonacci_backoff
from pr0loader.utils.console import (
    print_header,
    print_stats_table,
    print_success,
    print_info,
    console,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BROWSER-LIKE HEADERS (helps avoid WAF rate limiting)
# =============================================================================

# pr0gramm's WAF tolerates "typical browser behavior" - use full browser headers
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,de;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Referer": "https://pr0gramm.com/",
    "Sec-Fetch-Dest": "image",
    "Sec-Fetch-Mode": "no-cors",
    "Sec-Fetch-Site": "same-site",
    "Sec-CH-UA": '"Not_A Brand";v="8", "Chromium";v="120", "Google Chrome";v="120"',
    "Sec-CH-UA-Mobile": "?0",
    "Sec-CH-UA-Platform": '"Windows"',
}


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class DownloadItem:
    """Minimal download task - just what we need."""
    __slots__ = ('path', 'local_path')
    path: str           # Remote path like "2024/01/01/abc.jpg"
    local_path: Path    # Full local path


@dataclass
class DownloadStats:
    """Statistics for download run."""
    total_in_db: int = 0
    total_on_disk: int = 0
    to_download: int = 0
    downloaded: int = 0
    skipped: int = 0
    failed: int = 0
    bytes_downloaded: int = 0

    # Timing
    load_time: float = 0.0
    download_time: float = 0.0


# =============================================================================
# HELPERS
# =============================================================================

def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ('B', 'KB', 'MB', 'GB', 'TB'):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


# =============================================================================
# FAST DATA LOADING
# =============================================================================

def load_db_images(db_path: Path, include_videos: bool = False) -> pd.DataFrame:
    """Load all image paths from database into DataFrame.

    Returns DataFrame with columns: [id, image]
    Uses raw SQL for maximum speed.
    """
    start = time.perf_counter()

    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(
        "SELECT id, image FROM items WHERE image IS NOT NULL",
        conn
    )
    conn.close()

    # Filter by extension (vectorized)
    allowed = {'.jpg', '.jpeg', '.png', '.gif'}
    if include_videos:
        allowed |= {'.mp4', '.webm'}

    # Extract extension and filter
    df['ext'] = df['image'].str.extract(r'(\.[^.]+)$', expand=False).str.lower()
    df = df[df['ext'].isin(allowed)].drop(columns=['ext'])

    elapsed = time.perf_counter() - start
    logger.debug(f"Loaded {len(df):,} items from DB in {elapsed:.2f}s")

    return df


def scan_filesystem(media_dir: Path, include_videos: bool = False) -> set[str]:
    """Scan filesystem and return set of existing file paths.

    Uses os.scandir for speed (faster than os.walk for flat checks).
    Returns relative paths like "2024/01/01/abc.jpg"
    """
    start = time.perf_counter()

    allowed = {'.jpg', '.jpeg', '.png', '.gif'}
    if include_videos:
        allowed |= {'.mp4', '.webm'}

    existing: set[str] = set()

    def scan_recursive(path: Path, prefix: str = ""):
        """Recursively scan directory."""
        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_file(follow_symlinks=False):
                        ext = os.path.splitext(entry.name)[1].lower()
                        if ext in allowed:
                            rel_path = f"{prefix}{entry.name}" if prefix else entry.name
                            existing.add(rel_path)
                    elif entry.is_dir(follow_symlinks=False):
                        new_prefix = f"{prefix}{entry.name}/" if prefix else f"{entry.name}/"
                        scan_recursive(Path(entry.path), new_prefix)
        except PermissionError:
            pass

    if media_dir.exists():
        scan_recursive(media_dir)

    elapsed = time.perf_counter() - start
    logger.debug(f"Scanned {len(existing):,} files in {elapsed:.2f}s")

    return existing


def compute_download_list(
    df_db: pd.DataFrame,
    existing_files: set[str],
    media_dir: Path,
) -> list[DownloadItem]:
    """Compute list of files to download.

    Fast set-based comparison.
    Returns list of DownloadItem objects.
    """
    start = time.perf_counter()

    # Vectorized membership check
    mask = ~df_db['image'].isin(existing_files)
    to_download = df_db[mask]

    # Convert to download items
    items = [
        DownloadItem(
            path=row['image'],
            local_path=media_dir / row['image'],
        )
        for _, row in to_download.iterrows()
    ]

    elapsed = time.perf_counter() - start
    logger.debug(f"Computed {len(items):,} items to download in {elapsed:.2f}s")

    return items


# =============================================================================
# RATE LIMITER (Token Bucket)
# =============================================================================

class TokenBucket:
    """Token bucket rate limiter for controlling request rate.

    Proactively limits requests to avoid hitting WAF limits.
    Much better than reacting to 429s after the fact.
    """

    def __init__(self, rate: float, burst: int = 10):
        """
        Args:
            rate: Tokens per second (requests per second)
            burst: Maximum burst size (bucket capacity)
        """
        self.rate = rate
        self.burst = burst
        self._tokens = float(burst)
        self._last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """Acquire a token, waiting if necessary.

        Returns: Time waited in seconds
        """
        async with self._lock:
            now = time.time()

            # Refill tokens based on time passed
            elapsed = now - self._last_update
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_update = now

            if self._tokens >= 1:
                self._tokens -= 1
                return 0.0

            # Need to wait for a token
            wait_time = (1 - self._tokens) / self.rate
            self._tokens = 0
            self._last_update = now + wait_time

        await asyncio.sleep(wait_time)
        return wait_time


# =============================================================================
# ASYNC DOWNLOADER
# =============================================================================

class AsyncDownloader:
    """Fast async file downloader with connection pooling and rate limiting.

    Uses a token bucket rate limiter to proactively limit requests per second,
    avoiding WAF rate limits rather than reacting to 429s.
    """

    def __init__(
        self,
        base_url: str,
        cookies: dict[str, str],
        max_concurrent: int = 20,
        max_retries: int = 3,
        max_backoff: float = 60.0,
        rate_limit: float = 5.0,
        burst: int = 10,
        delay_min: float = 0.0,
        delay_max: float = 0.0,
    ):
        self.base_url = base_url.rstrip('/')
        self.cookies = cookies
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.max_backoff = max_backoff
        self.delay_min = delay_min
        self.delay_max = delay_max

        # Proactive rate limiter - prevents hitting WAF limits
        self._rate_limiter = TokenBucket(rate=rate_limit, burst=burst)

        # Stats
        self.downloaded = 0
        self.failed = 0
        self.bytes_total = 0

        # Rate limiting state (TESTED: pr0gramm uses IP-based cooldown)
        # On 429, ALL requests are blocked for ~60s - no point retrying individual files
        self._backoff_attempt = 0
        self._consecutive_429s = 0
        self._global_pause_until: float = 0  # timestamp when we can resume
        self._cooldown_lock = asyncio.Lock()  # coordinate pause across workers

        # Graceful shutdown support
        self._shutdown = asyncio.Event()
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown (Ctrl+C)."""
        def signal_handler(signum, frame):
            logger.warning("⚠️  Shutdown requested (Ctrl+C) - finishing current downloads...")
            self._shutdown.set()

        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def download_all(
        self,
        items: list[DownloadItem],
        progress: Progress,
        task_id,
    ) -> tuple[int, int, int]:
        """Download all items with progress updates.

        Returns: (downloaded, failed, bytes)
        """
        if not items:
            return 0, 0, 0

        # Create connector with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=self.max_concurrent,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=30,
        )

        # Semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async with aiohttp.ClientSession(
            connector=connector,
            cookies=self.cookies,
            headers=BROWSER_HEADERS,
            timeout=aiohttp.ClientTimeout(total=120, connect=10),
        ) as session:

            async def download_one(item: DownloadItem) -> tuple[bool, int]:
                """Download single file. Returns (success, bytes)."""
                # Check for shutdown request before starting
                if self._shutdown.is_set():
                    return False, 0

                async with semaphore:
                    # Check if we're in global cooldown (another worker hit 429)
                    now = time.time()
                    if self._global_pause_until > now:
                        wait = self._global_pause_until - now
                        logger.debug(f"Waiting {wait:.1f}s for global cooldown")

                        # Wait but check for shutdown periodically
                        try:
                            await asyncio.wait_for(self._shutdown.wait(), timeout=wait)
                            return False, 0  # Shutdown requested during wait
                        except asyncio.TimeoutError:
                            pass  # Wait completed normally

                    # Proactive rate limiting - wait for token before making request
                    await self._rate_limiter.acquire()

                    # Check shutdown again after waiting
                    if self._shutdown.is_set():
                        return False, 0

                    url = f"{self.base_url}/{item.path}"

                    # Add random delay if configured (usually 0)
                    if self.delay_max > 0:
                        delay = random.uniform(self.delay_min, self.delay_max)
                        await asyncio.sleep(delay)

                    for attempt in range(self.max_retries):
                        try:
                            async with session.get(url) as resp:
                                if resp.status == 200:
                                    # Reset backoff on success
                                    self._backoff_attempt = 0
                                    self._consecutive_429s = 0

                                    # Ensure directory exists
                                    item.local_path.parent.mkdir(parents=True, exist_ok=True)

                                    # Stream to file
                                    size = 0
                                    with open(item.local_path, 'wb') as f:
                                        async for chunk in resp.content.iter_chunked(65536):
                                            f.write(chunk)
                                            size += len(chunk)

                                    return True, size

                                elif resp.status == 429:
                                    # Rate limited - pr0gramm blocks ALL requests for ~60s
                                    # Trigger global pause - no point other workers retrying
                                    async with self._cooldown_lock:
                                        self._consecutive_429s += 1
                                        self._backoff_attempt += 1

                                        # Check for Retry-After header
                                        retry_after = resp.headers.get('Retry-After')
                                        if retry_after and retry_after.isdigit():
                                            wait = min(float(retry_after), self.max_backoff)
                                        else:
                                            # pr0gramm cooldown is ~60s, use at least that
                                            wait = max(60, fibonacci_backoff(self._backoff_attempt, self.max_backoff))

                                        # Set global pause - affects ALL workers
                                        self._global_pause_until = time.time() + wait

                                        logger.warning(
                                            f"🛑 Rate limited (x{self._consecutive_429s}) - "
                                            f"ALL workers pausing for {wait:.0f}s"
                                        )

                                    # Wait for cooldown
                                    await asyncio.sleep(wait)
                                    continue

                                elif resp.status == 404:
                                    # File doesn't exist remotely - skip
                                    return False, 0

                                else:
                                    logger.debug(f"HTTP {resp.status} for {item.path}")

                        except asyncio.TimeoutError:
                            logger.debug(f"Timeout for {item.path}, attempt {attempt + 1}")
                        except Exception as e:
                            logger.debug(f"Error for {item.path}: {e}, attempt {attempt + 1}")

                        # Exponential backoff between retries
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2 ** min(attempt, 3))

                    # All retries failed
                    logger.warning(f"Failed after {self.max_retries} attempts: {item.path}")
                    return False, 0

            # Process all items
            tasks = []
            for item in items:
                task = asyncio.create_task(download_one(item))
                tasks.append(task)

            # Gather with progress updates
            downloaded = 0
            failed = 0
            bytes_total = 0

            try:
                for coro in asyncio.as_completed(tasks):
                    try:
                        success, size = await coro
                        if success:
                            downloaded += 1
                            bytes_total += size
                            self.downloaded = downloaded
                            self.bytes_total = bytes_total
                        else:
                            failed += 1
                            self.failed = failed

                        # Update progress
                        progress.update(task_id, advance=1)

                        # Check for shutdown after each completed task
                        if self._shutdown.is_set():
                            logger.warning("Cancelling remaining downloads...")
                            # Cancel all pending tasks
                            for task in tasks:
                                if not task.done():
                                    task.cancel()
                            break

                    except asyncio.CancelledError:
                        failed += 1
                        self.failed = failed
                        progress.update(task_id, advance=1)

            except KeyboardInterrupt:
                logger.warning("KeyboardInterrupt in download loop - cancelling tasks...")
                self._shutdown.set()
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Let tasks finish cancelling
                await asyncio.gather(*tasks, return_exceptions=True)

        return downloaded, failed, bytes_total


# =============================================================================
# MAIN PIPELINE
# =============================================================================

class DownloadPipeline:
    """High-performance download pipeline.

    Fast as fuck boi! 🚀

    Optimizations:
    - Pandas for bulk data loading (no ORM overhead)
    - Set operations for file comparison (O(n) not O(n²))
    - Async I/O for concurrent downloads
    - Connection pooling with keep-alive
    - Minimal object creation
    - No blocking in event loop
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.stats = DownloadStats()

        # Tunable parameters
        self.max_concurrent = settings.max_parallel_requests
        self.max_retries = settings.max_retries
        self.max_backoff = settings.max_backoff_seconds

    def _create_progress(self) -> Progress:
        """Create progress bar."""
        return Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=console,
            transient=False,
        )

    def run(
        self,
        include_videos: bool = False,
        verify_existing: bool = False,
    ) -> PipelineStats:
        """Run the download pipeline.

        Args:
            include_videos: Download videos too (default: images only)
            verify_existing: Re-download if size differs (default: skip existing)
        """
        print_header(
            "📁 Download Media",
            "Fast as fuck downloader 🚀"
        )

        media_dir = self.settings.filesystem_prefix
        media_dir.mkdir(parents=True, exist_ok=True)

        print_info(f"Media directory: {media_dir}")
        print_info(f"Mode: {'Images + Videos' if include_videos else 'Images only'}")
        print_info(f"Workers: {self.max_concurrent}")

        total_start = time.perf_counter()

        # =====================================================================
        # PHASE 1: LOAD DATA
        # =====================================================================
        print_info("Phase 1: Loading data...")
        load_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Load DB and scan FS in parallel
            db_future = executor.submit(
                load_db_images,
                self.settings.db_path,
                include_videos,
            )
            fs_future = executor.submit(
                scan_filesystem,
                media_dir,
                include_videos,
            )

            df_db = db_future.result()
            existing_files = fs_future.result()

        self.stats.total_in_db = len(df_db)
        self.stats.total_on_disk = len(existing_files)

        print_info(f"  Database: {self.stats.total_in_db:,} items")
        print_info(f"  On disk:  {self.stats.total_on_disk:,} files")

        # Compute what to download
        items = compute_download_list(df_db, existing_files, media_dir)
        self.stats.to_download = len(items)
        self.stats.skipped = self.stats.total_in_db - self.stats.to_download

        self.stats.load_time = time.perf_counter() - load_start
        print_info(f"  To download: {self.stats.to_download:,} files")
        print_info(f"  Load time: {format_duration(self.stats.load_time)}")

        if not items:
            print_success("Nothing to download - all files exist!")
            return self._to_pipeline_stats()

        # =====================================================================
        # PHASE 2: DOWNLOAD
        # =====================================================================
        print_info(f"Phase 2: Downloading {len(items):,} files...")
        print_info(f"  Rate limit: {self.settings.download_rate_limit:.1f} req/s (burst: {self.settings.download_burst})")
        download_start = time.perf_counter()

        # Create downloader with token bucket rate limiter to avoid WAF limits
        downloader = AsyncDownloader(
            base_url=self.settings.media_base_url,
            cookies={
                'me': self.settings.me,
                'pp': self.settings.pp,
            },
            max_concurrent=self.max_concurrent,
            max_retries=self.max_retries,
            max_backoff=self.max_backoff,
            rate_limit=self.settings.download_rate_limit,
            burst=self.settings.download_burst,
            delay_min=self.settings.download_delay_min,
            delay_max=self.settings.download_delay_max,
        )

        # Run downloads with progress
        progress = self._create_progress()

        with progress:
            task_id = progress.add_task(
                "[cyan]Downloading...",
                total=len(items),
            )

            try:
                downloaded, failed, bytes_total = asyncio.run(
                    downloader.download_all(items, progress, task_id)
                )
            except KeyboardInterrupt:
                logger.warning("\n⚠️  Download interrupted by user (Ctrl+C)")
                print_info("\nDownload interrupted - partial results:")
                # Get partial stats from downloader
                downloaded = downloader.downloaded
                failed = downloader.failed
                bytes_total = downloader.bytes_total

        self.stats.downloaded = downloaded
        self.stats.failed = failed
        self.stats.bytes_downloaded = bytes_total
        self.stats.download_time = time.perf_counter() - download_start

        # =====================================================================
        # DONE
        # =====================================================================
        total_time = time.perf_counter() - total_start

        # Calculate rates
        if self.stats.download_time > 0:
            files_per_sec = self.stats.downloaded / self.stats.download_time
            bytes_per_sec = self.stats.bytes_downloaded / self.stats.download_time
        else:
            files_per_sec = 0
            bytes_per_sec = 0

        print_stats_table("Download Results", {
            "Downloaded": f"{self.stats.downloaded:,} files",
            "Skipped": f"{self.stats.skipped:,} files",
            "Failed": f"{self.stats.failed:,} files",
            "Data": format_bytes(self.stats.bytes_downloaded),
            "Speed": f"{files_per_sec:.1f} files/s ({format_bytes(int(bytes_per_sec))}/s)",
            "Load time": format_duration(self.stats.load_time),
            "Download time": format_duration(self.stats.download_time),
            "Total time": format_duration(total_time),
        })

        print_success("Download complete! 🎉")

        return self._to_pipeline_stats()

    def _to_pipeline_stats(self) -> PipelineStats:
        """Convert to PipelineStats for compatibility."""
        stats = PipelineStats()
        stats.files_downloaded = self.stats.downloaded
        stats.files_skipped = self.stats.skipped
        stats.items_failed = self.stats.failed
        stats.bytes_downloaded = self.stats.bytes_downloaded
        return stats

