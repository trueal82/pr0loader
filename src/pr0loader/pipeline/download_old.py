"""Download pipeline - download media files.

Optimized for large-scale downloads (6M+ assets):
- Async I/O with aiohttp for concurrent downloads
- Producer-consumer pattern with queues
- Parallel local file checks (filesystem I/O)
- Batched HEAD verification requests
- Connection pooling with proper limits
- Fibonacci backoff for rate limiting (friendly client)
"""

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp
import pandas as pd
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pr0loader.config import Settings
from pr0loader.models import PipelineStats
from pr0loader.storage import SQLiteStorage
from pr0loader.utils.backoff import fibonacci_backoff
from pr0loader.utils.console import (
    create_progress,
    print_header,
    print_stats_table,
    print_success,
    print_info,
    console,
)

logger = logging.getLogger(__name__)

# Queue sentinel to signal completion
_DONE = object()

# Global log buffer for the Live display
_LOG_BUFFER = deque(maxlen=10)  # Keep last 10 log messages


class _BufferingHandler(logging.Handler):
    """Custom logging handler that buffers ALL messages for Live display."""
    def emit(self, record):
        try:
            msg = self.format(record)
            # Buffer ALL debug messages (we suppress console output anyway)
            _LOG_BUFFER.append(msg)
        except Exception:
            self.handleError(record)


def format_bytes(size: int) -> str:
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def _format_seconds(seconds: float) -> str:
    """Format seconds to human-readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.2f} min"


@dataclass
class DownloadTask:
    """A task representing a file to potentially download."""
    item_id: int
    remote_path: str  # e.g., "2024/01/01/abc123.jpg"
    local_path: Path
    needs_download: bool = True
    remote_size: Optional[int] = None
    reason: str = "new"


class DownloadPipeline:
    """High-performance async download pipeline.

    Architecture:
    ```
    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │  DB Reader  │ ──► │ Check Queue │ ──► │  Checkers   │
    │  (producer) │     │             │     │  (workers)  │
    └─────────────┘     └─────────────┘     └──────┬──────┘
                                                   │
                                                   ▼
                                           ┌─────────────┐
                                           │Download Queue│
                                           └──────┬──────┘
                                                   │
                                                   ▼
                                           ┌─────────────┐
                                           │ Downloaders │
                                           │  (workers)  │
                                           └─────────────┘
    ```

    - DB Reader: iterates items from SQLite, creates tasks
    - Checkers: verify if download needed (local exists? HEAD check?)
    - Downloaders: actually download the files

    This decouples I/O-bound operations and maximizes throughput.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.stats = PipelineStats()
        self._backoff_attempt = 0
        self._max_backoff = settings.max_backoff_seconds

        # Tunable parameters for 6M scale
        self.check_workers = min(32, settings.max_parallel_requests * 2)  # Local checks are fast
        self.download_workers = settings.max_parallel_requests  # Network-bound
        self.check_queue_size = 10000  # Buffer for checking
        self.download_queue_size = 1000  # Buffer for downloads

        # For queue monitoring
        self._monitor_running = False
        self._buffering_handler = None
        self._root_logger_state: Optional[tuple[int, list[logging.Handler]]] = None
        self._fs_index: Optional[set[str]] = None

    def _format_queue_bar(self, current: int, maximum: int, width: int = 20) -> str:
        """Create ASCII progress bar for queue fill level."""
        if maximum == 0:
            return "│" + "░" * width + "│"

        fill_ratio = min(current / maximum, 1.0)
        filled = int(fill_ratio * width)
        empty = width - filled

        # Color coding: green < 50%, yellow 50-80%, red > 80%
        if fill_ratio < 0.5:
            bar_char = "█"
        elif fill_ratio < 0.8:
            bar_char = "▓"
        else:
            bar_char = "▒"

        return f"│{bar_char * filled}{'░' * empty}│"

    def _build_queue_panel(
        self,
        check_queue_size: int,
        download_queue_size: int,
        skipped: int,
        downloaded: int,
        failed: int,
        bytes_dl: int,
    ) -> Panel:
        """Build a Rich Panel with queue status, stats, and recent logs."""
        check_pct = (check_queue_size / self.check_queue_size * 100) if self.check_queue_size else 0
        dl_pct = (download_queue_size / self.download_queue_size * 100) if self.download_queue_size else 0

        check_bar = self._format_queue_bar(check_queue_size, self.check_queue_size)
        dl_bar = self._format_queue_bar(download_queue_size, self.download_queue_size)

        # Build queue section
        queue_lines = [
            f"Check    {check_bar} {check_queue_size:>6,}/{self.check_queue_size:<6,} ({check_pct:>3.0f}%)",
            f"Download {dl_bar} {download_queue_size:>6,}/{self.download_queue_size:<6,} ({dl_pct:>3.0f}%)",
        ]

        # Build stats section
        stats_line = f"✓ {skipped:,} skip | ⬇ {downloaded:,} dl ({format_bytes(bytes_dl)}) | ✗ {failed:,} fail"

        # Build table
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column(style="cyan")

        # Add queue status
        for line in queue_lines:
            table.add_row(line)

        table.add_row()  # Spacing
        table.add_row(Text(stats_line, style="dim"))

        # Add recent logs if available
        table.add_row()  # Spacing
        table.add_row(Text("─" * 35, style="dim"))

        buffer_size = len(_LOG_BUFFER)
        table.add_row(Text(f"Recent activity: ({buffer_size} buffered)", style="dim bold"))

        if _LOG_BUFFER:
            # Show last 8 logs, truncated to fit
            for log_msg in list(_LOG_BUFFER)[-8:]:
                # Truncate long messages
                if len(log_msg) > 35:
                    log_msg = log_msg[:32] + "..."
                table.add_row(Text(f"  {log_msg}", style="dim white"))
        else:
            table.add_row(Text("  (no messages yet)", style="dim italic"))

        return Panel(
            table,
            title="[bold cyan]QUEUE STATUS[/bold cyan]",
            border_style="blue",
            expand=False,
            padding=(1, 2),
        )

    async def _queue_monitor_live(
        self,
        check_queue: asyncio.Queue,
        download_queue: asyncio.Queue,
        stats_lock: asyncio.Lock,
        layout: Layout,
        interval: float = 1.0,
    ):
        """Background task that updates Live layout with queue status."""
        while self._monitor_running:
            await asyncio.sleep(interval)

            check_size = check_queue.qsize()
            download_size = download_queue.qsize()

            async with stats_lock:
                skipped = self.stats.files_skipped
                downloaded = self.stats.files_downloaded
                failed = self.stats.items_failed
                bytes_dl = self.stats.bytes_downloaded

            panel = self._build_queue_panel(
                check_size, download_size,
                skipped, downloaded, failed, bytes_dl
            )

            layout["panel"].update(panel)

    def _v(self, msg: str):
        """Verbose log helper - logs to buffer when verbose mode active."""
        # Always log when we have a buffering handler
        if self._buffering_handler:
            _LOG_BUFFER.append(msg)  # Direct append for testing
        # Also try logger (in case it works)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(msg)

    async def _get_remote_size(
        self,
        session: aiohttp.ClientSession,
        url: str
    ) -> Optional[int]:
        """Get remote file size with HEAD request + backoff."""
        for attempt in range(self.settings.max_retries):
            try:
                async with session.head(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        self._backoff_attempt = 0
                        content_length = resp.headers.get('Content-Length')
                        return int(content_length) if content_length else None
                    elif resp.status == 429:
                        wait = fibonacci_backoff(self._backoff_attempt, self._max_backoff)
                        self._backoff_attempt += 1
                        logger.warning(f"Rate limited on HEAD, waiting {wait}s")
                        await asyncio.sleep(wait)
                    elif resp.status == 404:
                        return None  # File doesn't exist remotely
                    else:
                        logger.debug(f"HEAD {url} returned {resp.status}")
                        return None
            except asyncio.TimeoutError:
                logger.debug(f"HEAD timeout for {url}")
            except Exception as e:
                logger.debug(f"HEAD error for {url}: {e}")

            if attempt < self.settings.max_retries - 1:
                await asyncio.sleep(1)

        return None

    async def _download_file(
        self,
        session: aiohttp.ClientSession,
        url: str,
        destination: Path
    ) -> int:
        """Download a file with retry and backoff."""
        destination.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(self.settings.max_retries):
            try:
                async with session.get(
                    url,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as resp:
                    if resp.status == 200:
                        self._backoff_attempt = 0
                        size = 0
                        with open(destination, 'wb') as f:
                            async for chunk in resp.content.iter_chunked(8192):
                                f.write(chunk)
                                size += len(chunk)
                        return size
                    elif resp.status == 429:
                        wait = fibonacci_backoff(self._backoff_attempt, self._max_backoff)
                        self._backoff_attempt += 1
                        logger.warning(f"Rate limited on download, waiting {wait}s")
                        await asyncio.sleep(wait)
                    elif resp.status == 404:
                        raise FileNotFoundError(f"Remote file not found: {url}")
                    else:
                        raise Exception(f"HTTP {resp.status}")
            except asyncio.TimeoutError:
                logger.warning(f"Download timeout for {url}, attempt {attempt + 1}")
            except FileNotFoundError:
                raise
            except Exception as e:
                logger.warning(f"Download error for {url}: {e}, attempt {attempt + 1}")

            if attempt < self.settings.max_retries - 1:
                await asyncio.sleep(2 ** min(attempt, 4))

        raise Exception(f"Failed to download {url} after {self.settings.max_retries} attempts")

    def _build_fs_index(self, media_dir: Path, include_videos: bool) -> set[str]:
        """Build an in-memory index of existing files for fast membership checks."""
        allowed_exts = {".jpg", ".jpeg", ".png", ".gif"}
        if include_videos:
            allowed_exts |= {".mp4", ".webm"}

        start = time.perf_counter()
        index: set[str] = set()
        file_count = 0

        for root, _, files in os.walk(media_dir):
            for name in files:
                ext = Path(name).suffix.lower()
                if ext not in allowed_exts:
                    continue
                full_path = Path(root) / name
                rel_path = full_path.relative_to(media_dir).as_posix()
                index.add(rel_path)
                file_count += 1

        print_info(f"FS index: {file_count:,} files in {_format_seconds(time.perf_counter() - start)}")
        return index

    async def _checker_worker(
        self,
        worker_id: int,
        check_queue: asyncio.Queue,
        download_queue: asyncio.Queue,
        session: aiohttp.ClientSession,
        media_base_url: str,
        verify_existing: bool,
        stats_lock: asyncio.Lock,
    ):
        """Worker that checks if files need downloading (verify_existing path only)."""
        loop = asyncio.get_event_loop()
        items_processed = 0

        while True:
            task = await check_queue.get()
            if task is _DONE:
                check_queue.task_done()
                break

            try:
                # Run blocking filesystem check in thread pool
                exists = await loop.run_in_executor(None, task.local_path.exists)

                if exists:
                    local_size = await loop.run_in_executor(
                        None, lambda: task.local_path.stat().st_size
                    )

                    if not verify_existing:
                        task.needs_download = False
                        task.reason = "exists_no_verify"
                        self._v(f"SKIP({task.reason}) {task.remote_path}")
                        async with stats_lock:
                            self.stats.files_skipped += 1
                    else:
                        url = f"{media_base_url}/{task.remote_path}"
                        remote_size = await self._get_remote_size(session, url)

                        if remote_size is None:
                            task.needs_download = False
                            task.reason = "verify_failed_keep_local"
                            async with stats_lock:
                                self.stats.files_skipped += 1
                        elif remote_size == local_size:
                            task.needs_download = False
                            task.reason = "verified_ok"
                            async with stats_lock:
                                self.stats.files_skipped += 1
                        else:
                            task.needs_download = True
                            task.remote_size = remote_size
                            task.reason = "size_mismatch"

                        self._v(f"{'DOWNLOAD' if task.needs_download else 'SKIP'}({task.reason}) {task.remote_path}")
                else:
                    task.needs_download = True
                    task.reason = "new"
                    self._v(f"DOWNLOAD({task.reason}) {task.remote_path}")

                if task.needs_download:
                    await download_queue.put(task)

            except Exception as e:
                logger.error(f"Checker error for {task.remote_path}: {e}")
                async with stats_lock:
                    self.stats.items_failed += 1
            finally:
                check_queue.task_done()
                items_processed += 1
                if items_processed % 100 == 0:
                    await asyncio.sleep(0)

    async def _downloader_worker(
        self,
        worker_id: int,
        download_queue: asyncio.Queue,
        session: aiohttp.ClientSession,
        media_base_url: str,
        stats_lock: asyncio.Lock,
    ):
        """Worker that downloads files."""
        while True:
            task = await download_queue.get()
            if task is _DONE:
                download_queue.task_done()
                break

            url = f"{media_base_url}/{task.remote_path}"

            try:
                size = await self._download_file(session, url, task.local_path)
                async with stats_lock:
                    self.stats.files_downloaded += 1
                    self.stats.bytes_downloaded += size
                self._v(f"DONE {task.remote_path} {format_bytes(size)}")
            except FileNotFoundError:
                # Remote file doesn't exist (404)
                async with stats_lock:
                    self.stats.items_skipped += 1
                self._v(f"SKIP(not_found_remote) {task.remote_path}")
            except Exception as e:
                logger.error(f"Download failed for {task.remote_path}: {e}")
                async with stats_lock:
                    self.stats.items_failed += 1
            finally:
                download_queue.task_done()

    async def _run_async(
        self,
        include_videos: bool,
        verify_existing: bool,
        verbose: bool = False,
    ) -> PipelineStats:
        """Async implementation of the download pipeline.

        Args:
            verbose: If True, show live queue monitoring panel with log capture
        """
        # Set up buffering handler and suppress external logs if verbose
        if verbose and not self._buffering_handler:
            root_logger = logging.getLogger()
            self._root_logger_state = (root_logger.level, root_logger.handlers[:])

            # Silence root logger (prevents sqlite/asyncio debug spam)
            root_logger.handlers = []
            root_logger.setLevel(logging.WARNING)

            # Remove all existing handlers from this logger
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Add only our buffering handler
            self._buffering_handler = _BufferingHandler()
            self._buffering_handler.setFormatter(logging.Formatter('%(message)s'))
            self._buffering_handler.setLevel(logging.DEBUG)
            logger.addHandler(self._buffering_handler)
            logger.setLevel(logging.DEBUG)
            logger.propagate = False
            logger.debug("Buffering handler initialized")

        media_dir = self.settings.filesystem_prefix
        media_dir.mkdir(parents=True, exist_ok=True)
        media_base_url = self.settings.media_base_url

        # Create queues
        check_queue: asyncio.Queue = asyncio.Queue(maxsize=self.check_queue_size)
        download_queue: asyncio.Queue = asyncio.Queue(maxsize=self.download_queue_size)

        # Fast path: build FS index once when verify_existing is False
        if not verify_existing:
            self._fs_index = self._build_fs_index(media_dir, include_videos)

        # Lock for thread-safe stats updates
        stats_lock = asyncio.Lock()

        # Get total count for progress
        with SQLiteStorage(self.settings.db_path) as storage:
            total_items = storage.get_item_count()

        print_info(f"Items in database: {total_items:,}")
        print_info(f"Check workers: {self.check_workers}, Download workers: {self.download_workers}")

        # Configure aiohttp with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.download_workers + self.check_workers,
            limit_per_host=self.download_workers + self.check_workers,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
        )

        cookies = {
            'me': self.settings.me,
            'pp': self.settings.pp,
        }

        progress = create_progress("Downloading")

        # Live layout only in verbose mode
        live_display = None
        layout = None
        if verbose:
            layout = Layout()
            layout.split_row(
                Layout(progress, name="progress"),
                Layout(Panel("Initializing...", title="[bold cyan]QUEUE STATUS[/bold cyan]"), name="panel", size=60),
            )
            live_display = Live(layout, console=console, refresh_per_second=4)
            live_display.start()
            progress.start()

        async with aiohttp.ClientSession(
            connector=connector,
            cookies=cookies,
        ) as session:
            checker_tasks = []
            if verify_existing:
                checker_tasks = [
                    asyncio.create_task(
                        self._checker_worker(
                            i, check_queue, download_queue, session,
                            media_base_url, verify_existing, stats_lock
                        )
                    )
                    for i in range(self.check_workers)
                ]

            # Start downloader workers
            downloader_tasks = [
                asyncio.create_task(
                    self._downloader_worker(
                        i, download_queue, session, media_base_url, stats_lock
                    )
                )
                for i in range(self.download_workers)
            ]

            # Start queue monitor (if verbose)
            self._monitor_running = True
            monitor_task = None
            if verbose and layout:
                monitor_task = asyncio.create_task(
                    self._queue_monitor_live(
                        check_queue, download_queue, stats_lock,
                        layout, interval=1.0
                    )
                )

            async def _produce(task):
                """Fast producer using pandas dataframe operations."""
                loop = asyncio.get_event_loop()

                def _load_and_compare():
                    """Load DB and FS data, find missing files - runs in executor."""
                    print_info("Loading database...")
                    start = time.perf_counter()

                    # Load all items from database into dataframe
                    with SQLiteStorage(self.settings.db_path) as storage:
                        # Get raw data via SQL query for speed
                        import sqlite3
                        conn = sqlite3.connect(self.settings.db_path)
                        df_db = pd.read_sql_query(
                            "SELECT id, image FROM items WHERE image IS NOT NULL",
                            conn
                        )
                        conn.close()

                    print_info(f"Loaded {len(df_db):,} items from database in {time.perf_counter() - start:.1f}s")

                    # Filter by extension
                    allowed_exts = {'.jpg', '.jpeg', '.png', '.gif'}
                    if include_videos:
                        allowed_exts |= {'.mp4', '.webm'}

                    df_db['ext'] = df_db['image'].str.lower().str.extract(r'(\.[^.]+)$')
                    df_db = df_db[df_db['ext'].isin(allowed_exts)].copy()

                    print_info(f"Filtered to {len(df_db):,} image items")

                    if not verify_existing:
                        # Use fast FS index set comparison
                        print_info(f"Using FS index with {len(self._fs_index):,} files")
                        df_db['in_fs'] = df_db['image'].isin(self._fs_index)
                        to_download = df_db[~df_db['in_fs']].copy()
                        skipped = len(df_db) - len(to_download)
                    else:
                        # All items need verification
                        to_download = df_db.copy()
                        skipped = 0

                    print_info(f"Found {len(to_download):,} files to download ({skipped:,} skipped)")
                    return to_download[['id', 'image']]

                # Run the comparison in executor (thread pool)
                to_download_df = await loop.run_in_executor(None, _load_and_compare)

                # Enqueue all items
                print_info(f"Enqueueing {len(to_download_df):,} items...")
                enqueued = 0
                last_update = 0

                for _, row in to_download_df.iterrows():
                    download_task = DownloadTask(
                        item_id=row['id'],
                        remote_path=row['image'],
                        local_path=media_dir / row['image'],
                    )

                    if verify_existing:
                        await check_queue.put(download_task)
                    else:
                        await download_queue.put(download_task)

                    enqueued += 1

                    # Update progress periodically
                    if enqueued - last_update >= 1000:
                        progress.update(task, completed=enqueued)
                        last_update = enqueued

                progress.update(task, completed=enqueued, description="[cyan]Checking/downloading...")
                print_info(f"Enqueued {enqueued:,} items")


            # Producer: read from database and fill check queue
            if verbose:
                task = progress.add_task("[cyan]Processing...", total=total_items)
                await _produce(task)
            else:
                with progress:
                    task = progress.add_task("[cyan]Processing...", total=total_items)
                    await _produce(task)

            if verify_existing:
                for _ in range(self.check_workers):
                    await check_queue.put(_DONE)
                await asyncio.gather(*checker_tasks)

            for _ in range(self.download_workers):
                await download_queue.put(_DONE)
            await asyncio.gather(*downloader_tasks)

            # Stop queue monitor
            self._monitor_running = False
            if monitor_task:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

            # Final progress update
            progress.update(task, completed=total_items)

        # Stop queue monitor and live display
        if verbose:
            self._monitor_running = False
            if monitor_task:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass

            if live_display:
                progress.stop()
                live_display.stop()

        # Restore root logger and clean up buffering handler
        if self._buffering_handler:
            logger.removeHandler(self._buffering_handler)
            logger.propagate = True
            self._buffering_handler = None

        if self._root_logger_state:
            root_level, root_handlers = self._root_logger_state
            root_logger = logging.getLogger()
            root_logger.setLevel(root_level)
            root_logger.handlers = root_handlers
            self._root_logger_state = None

        return self.stats

    def run(self, include_videos: bool = False, verify_existing: bool = False) -> PipelineStats:
        """Run the download pipeline.

        Uses async I/O with producer-consumer pattern for maximum throughput.

        Args:
            include_videos: If True, also download videos. Default is images only.
            verify_existing: If True, verify existing files via HEAD request and
                           re-download if size differs. Default is False (skip existing).
        """
        print_header(
            "📁 Download Media",
            "High-performance async downloader"
        )

        media_dir = self.settings.filesystem_prefix
        media_dir.mkdir(parents=True, exist_ok=True)
        print_info(f"Media directory: {media_dir}")
        print_info(f"Download mode: {'All media (images + videos)' if include_videos else 'Images only'}")
        print_info(f"Verify existing: {'Yes (HEAD check)' if verify_existing else 'No (skip existing)'}")

        # Detect verbose mode from logger
        verbose = logger.isEnabledFor(logging.DEBUG)
        if verbose:
            print_info("📊 Queue monitoring enabled")

        # Run async pipeline
        start_time = time.time()
        asyncio.run(self._run_async(include_videos, verify_existing, verbose=verbose))
        elapsed = time.time() - start_time

        # Calculate rates
        total_processed = (
            self.stats.files_downloaded +
            self.stats.files_skipped +
            self.stats.items_skipped +
            self.stats.items_failed
        )
        rate = total_processed / elapsed if elapsed > 0 else 0

        # Print final stats
        print_stats_table("Download Results", {
            "Files downloaded": self.stats.files_downloaded,
            "Files skipped (exist)": self.stats.files_skipped,
            "Items skipped (videos)": self.stats.items_skipped,
            "Failed": self.stats.items_failed,
            "Total bytes": format_bytes(self.stats.bytes_downloaded),
            "Time": f"{elapsed:.1f}s",
            "Rate": f"{rate:.0f} items/s",
        })

        print_success("Download complete!")

        return self.stats

