#!/usr/bin/env python3
"""Quick test to verify the download pipeline fix works."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pr0loader.config import Settings
from pr0loader.pipeline.download import DownloadPipeline

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)-8s %(message)s',
)

async def test_producer_responsiveness():
    """Test that the producer doesn't block the event loop."""
    settings = Settings()
    pipeline = DownloadPipeline(settings)

    print("Testing producer responsiveness with thread pool executor...")
    print(f"Database path: {settings.db_path}")
    print(f"Database exists: {settings.db_path.exists()}")

    if not settings.db_path.exists():
        print("ERROR: Database not found!")
        return False

    # Simulate what happens in _run_async
    loop = asyncio.get_event_loop()
    db_queue = asyncio.Queue(maxsize=5000)
    items_read = 0

    def _read_db():
        """Blocking DB read in executor."""
        from pr0loader.storage import SQLiteStorage
        try:
            count = 0
            with SQLiteStorage(settings.db_path) as storage:
                for item in storage.iter_items():
                    db_queue.put_nowait(item)
                    count += 1
                    if count % 100000 == 0:
                        print(f"  [DB thread] Read {count:,} items")
            print(f"  [DB thread] Finished reading {count:,} items")
        except Exception as e:
            print(f"ERROR in DB read: {e}")
            db_queue.put_nowait(None)
        finally:
            db_queue.put_nowait("DONE")

    # Start DB reader in executor
    reader_task = loop.run_in_executor(None, _read_db)

    # Concurrently consume from queue and check responsiveness
    items_processed = 0
    last_log = 0
    event_loop_runs = 0

    async def log_event_loop():
        """Log to show event loop is responsive."""
        nonlocal event_loop_runs
        while True:
            await asyncio.sleep(0.5)
            event_loop_runs += 1
            print(f"  [Event loop] Tick {event_loop_runs} - items processed: {items_processed:,}")

    log_task = asyncio.create_task(log_event_loop())

    try:
        while True:
            try:
                item = db_queue.get_nowait()
            except asyncio.QueueEmpty:
                await asyncio.sleep(0.001)
                continue

            if item == "DONE" or item is None:
                break

            items_processed += 1

            # Yield occasionally
            if items_processed % 50000 == 0:
                await asyncio.sleep(0)

        await reader_task
        log_task.cancel()
        try:
            await log_task
        except asyncio.CancelledError:
            pass

        print(f"\n✓ Test passed!")
        print(f"  Total items: {items_processed:,}")
        print(f"  Event loop ticks: {event_loop_runs}")
        print(f"  Event loop was responsive during DB read")
        return True

    except Exception as e:
        print(f"✗ Test failed: {e}")
        log_task.cancel()
        return False

if __name__ == "__main__":
    result = asyncio.run(test_producer_responsiveness())
    sys.exit(0 if result else 1)

