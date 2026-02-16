"""Benchmark filesystem existence-check strategies for pr0loader.

Reads settings from .env via load_settings(). Designed for large datasets.

Strategies:
1) per_item_stat: per-item Path.exists() check
2) set_diff: build sets of DB paths and FS paths, then set difference
3) dir_batched: group by directory and listdir once per directory

Usage:
    python3 scripts/benchmark_fs_checks.py --method all --max-items 200000
    python3 scripts/benchmark_fs_checks.py --method set_diff --max-files 2000000

Notes:
- Use --max-items and --max-files for safe testing.
- For a real 6M run, remove limits.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from pr0loader.config import load_settings


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif"}
VIDEO_EXTS = {".mp4", ".webm"}


def _iter_db_images(db_path: Path, include_videos: bool, limit: int | None) -> Iterator[str]:
    """Yield image paths from SQLite DB."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT image FROM items WHERE image IS NOT NULL")
        count = 0
        for (image_path,) in cursor:
            if not image_path:
                continue
            ext = Path(image_path).suffix.lower()
            if include_videos:
                if ext not in IMAGE_EXTS and ext not in VIDEO_EXTS:
                    continue
            else:
                if ext not in IMAGE_EXTS:
                    continue

            yield image_path
            count += 1
            if limit and count >= limit:
                break
    finally:
        conn.close()


def _iter_fs_files(media_dir: Path, limit: int | None) -> Iterator[str]:
    """Yield relative file paths under media_dir."""
    count = 0
    for root, _, files in os.walk(media_dir):
        for name in files:
            full_path = Path(root) / name
            rel_path = full_path.relative_to(media_dir).as_posix()
            yield rel_path
            count += 1
            if limit and count >= limit:
                return


def _benchmark_per_item_stat(db_paths: list[str], media_dir: Path) -> dict:
    """Strategy 1: Per-item stat check."""
    start = time.perf_counter()
    missing = 0
    for rel in db_paths:
        if not (media_dir / rel).exists():
            missing += 1
    elapsed = time.perf_counter() - start
    return {
        "missing": missing,
        "elapsed": elapsed,
        "checked": len(db_paths),
    }


def _benchmark_set_diff(db_paths: list[str], media_dir: Path, fs_limit: int | None) -> dict:
    """Strategy 2: Build sets and do set difference."""
    start = time.perf_counter()
    db_set = set(db_paths)
    t_db = time.perf_counter() - start

    start = time.perf_counter()
    fs_set = set(_iter_fs_files(media_dir, fs_limit))
    t_fs = time.perf_counter() - start

    start = time.perf_counter()
    missing = len(db_set - fs_set)
    t_diff = time.perf_counter() - start

    return {
        "missing": missing,
        "elapsed": t_db + t_fs + t_diff,
        "checked": len(db_set),
        "t_db_set": t_db,
        "t_fs_set": t_fs,
        "t_diff": t_diff,
        "fs_count": len(fs_set),
    }


def _benchmark_dir_batched(db_paths: list[str], media_dir: Path) -> dict:
    """Strategy 3: Group by directory and listdir once per directory."""
    start = time.perf_counter()
    by_dir: dict[str, set[str]] = defaultdict(set)
    for rel in db_paths:
        rel_path = Path(rel)
        by_dir[str(rel_path.parent)] .add(rel_path.name)
    t_group = time.perf_counter() - start

    start = time.perf_counter()
    missing = 0
    for dir_rel, names in by_dir.items():
        dir_full = media_dir / dir_rel
        if not dir_full.exists():
            missing += len(names)
            continue
        try:
            entries = set(os.listdir(dir_full))
        except OSError:
            missing += len(names)
            continue
        missing += len(names - entries)
    t_check = time.perf_counter() - start

    return {
        "missing": missing,
        "elapsed": t_group + t_check,
        "checked": len(db_paths),
        "t_group": t_group,
        "t_check": t_check,
        "dirs": len(by_dir),
    }


def _format_seconds(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    if seconds < 60:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.2f} min"


def _print_result(name: str, result: dict):
    print(f"\n=== {name} ===")
    print(f"Checked: {result.get('checked', 0):,}")
    print(f"Missing: {result.get('missing', 0):,}")
    print(f"Elapsed: {_format_seconds(result.get('elapsed', 0.0))}")

    # Optional breakdowns
    for key in ["t_db_set", "t_fs_set", "t_diff", "t_group", "t_check"]:
        if key in result:
            print(f"{key}: {_format_seconds(result[key])}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark file existence strategies")
    parser.add_argument("--method", choices=["per_item_stat", "set_diff", "dir_batched", "all"], default="all")
    parser.add_argument("--include-videos", action="store_true", help="Include videos in DB list")
    parser.add_argument("--max-items", type=int, default=None, help="Limit DB items (for testing)")
    parser.add_argument("--max-files", type=int, default=None, help="Limit filesystem scan size (set_diff only)")
    args = parser.parse_args()

    settings = load_settings()
    db_path = settings.db_path
    media_dir = settings.filesystem_prefix

    print("=== pr0loader FS Check Benchmark ===")
    print(f"DB: {db_path}")
    print(f"Media: {media_dir}")
    print(f"Include videos: {args.include_videos}")
    if args.max_items:
        print(f"DB limit: {args.max_items:,}")
    if args.max_files:
        print(f"FS limit: {args.max_files:,}")

    # Load DB paths once
    t0 = time.perf_counter()
    db_paths = list(_iter_db_images(db_path, args.include_videos, args.max_items))
    t_db = time.perf_counter() - t0
    print(f"Loaded {len(db_paths):,} DB paths in {_format_seconds(t_db)}")

    if args.method in {"per_item_stat", "all"}:
        res = _benchmark_per_item_stat(db_paths, media_dir)
        _print_result("Per-item stat", res)

    if args.method in {"set_diff", "all"}:
        res = _benchmark_set_diff(db_paths, media_dir, args.max_files)
        _print_result("Set diff", res)

    if args.method in {"dir_batched", "all"}:
        res = _benchmark_dir_batched(db_paths, media_dir)
        _print_result("Directory batched", res)


if __name__ == "__main__":
    main()

