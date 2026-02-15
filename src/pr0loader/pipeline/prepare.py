"""Prepare pipeline - prepare dataset for ML training."""

import csv
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

from pr0loader.config import Settings
from pr0loader.models import Item, PipelineStats
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

# Tag filtering
VALID_TAG_REGEX = re.compile(r'^[a-zA-Z0-9]+$')
NSFW_TAGS = {'nsfw', 'nsfl', 'nsfp'}

# Trash tags that should be blacklisted regardless of frequency
# These are measured from actual data - common but not informative
TRASH_TAGS_BLACKLIST = {
    'repost', 'nice', 'oc', 'gay', 'alt', 'fake', 'old', 'video',
    'webm', 'gif', 'jpg', 'png', 'sound', 'teil',  # Format indicators
    'arsch', 'titten',  # Generic body parts
    # Add more as discovered from data analysis
}


class PreparePipeline:
    """Pipeline stage for preparing training dataset."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.stats = PipelineStats()
        self.tag_counts: Optional[dict[str, int]] = None
        self.trash_tags: set[str] = set()
        self.high_confidence_threshold = 0.4  # Tags with >0.4 confidence are likely user-verified

    def analyze_tag_quality(self, storage: SQLiteStorage) -> tuple[dict[str, int], set[str]]:
        """
        Analyze tags to identify trash tags and build vocabulary.

        Strategy:
        1. Find VERY common tags (top 0.5%) - likely trash like "repost", "nice"
        2. Combine with manual blacklist
        3. Build full tag counts for reference

        Returns:
            Tuple of (all_tag_counts, trash_tags_to_filter)
        """
        print_info("Analyzing tag quality...")

        # Get all tag counts from database
        all_tag_counts = storage.get_tag_counts(limit=None)
        total_tags = len(all_tag_counts)

        print_info(f"Total unique tags: {total_tags}")

        # Find extremely common tags (top 0.5%) - likely trash
        top_n = max(int(total_tags * 0.005), 20)  # At least top 20
        top_tags = all_tag_counts[:top_n]

        # Start with manual blacklist
        trash_tags = TRASH_TAGS_BLACKLIST.copy()

        # Add extremely common tags to potential trash list
        # (Manual review would refine this, but these are usually meta-tags)
        for tag, count in top_tags:
            tag_lower = tag.lower()
            if tag_lower not in NSFW_TAGS:  # Don't add NSFW flags
                # Very common tags are often trash (repost, nice, etc.)
                trash_tags.add(tag)

        print_info(f"Identified {len(trash_tags)} potential trash tags")
        print_info("Top 20 most common tags (potential trash):")
        for tag, count in top_tags[:20]:
            marker = "🗑️" if tag in trash_tags else "✓"
            print_info(f"  {marker} {tag}: {count:,} occurrences")

        # Show tag count distribution
        if logger.isEnabledFor(logging.DEBUG):
            tag_counts_list = [count for _, count in all_tag_counts]
            logger.debug(f"Tag count distribution:")
            logger.debug(f"  Max: {max(tag_counts_list):,}")
            logger.debug(f"  Top 100 avg: {sum(tag_counts_list[:100]) / 100:.1f}")
            logger.debug(f"  Median: {tag_counts_list[len(tag_counts_list)//2]}")
            logger.debug(f"  Tags with 1 occurrence: {sum(1 for c in tag_counts_list if c == 1)}")

        return dict(all_tag_counts), trash_tags

    def process_tags(self, item: Item) -> dict:
        """
        Process tags for an item with smart filtering.

        Filtering strategy (your insight!):
        1. Remove NSFW flags (extracted separately)
        2. Validate format (alphanumeric)
        3. Blacklist trash tags (repost, nice, etc.) - ALWAYS filtered
        4. Keep rare tags IF they have high confidence (>0.4) - user-verified specific tags
        5. Sort by confidence

        This allows specific rare tags (e.g., "quantenphysik" with high confidence)
        while filtering common trash (e.g., "repost" even with high confidence).

        Returns dict with NSFW flags and valid tags.
        """
        is_nsfw = is_nsfl = is_nsfp = False
        valid_tags = []

        for tag in item.tags:
            tag_name = tag.tag.lower()

            # Check NSFW flags
            if tag_name == 'nsfw':
                is_nsfw = True
                continue
            elif tag_name == 'nsfl':
                is_nsfl = True
                continue
            elif tag_name == 'nsfp':
                is_nsfp = True
                continue

            # Validate tag name (alphanumeric only)
            if not VALID_TAG_REGEX.match(tag.tag):
                continue

            # ALWAYS filter trash tags (regardless of confidence or frequency)
            if tag.tag in self.trash_tags or tag.tag.lower() in self.trash_tags:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Filtered trash tag: {tag.tag} (confidence: {tag.confidence:.3f})")
                continue

            # Keep tags with HIGH confidence (>0.4) - likely user-verified
            # These are valuable even if rare (specific technical terms, etc.)
            if tag.confidence >= self.high_confidence_threshold:
                valid_tags.append(tag)
                continue

            # For low-confidence tags, require them to be in vocabulary (popular enough)
            # This filters out rare misspellings, noise, etc.
            if self.tag_counts and tag.tag not in self.tag_counts:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Filtered rare low-confidence tag: {tag.tag} (confidence: {tag.confidence:.3f})")
                continue

            valid_tags.append(tag)

        # Sort by confidence (highest first)
        valid_tags.sort(key=lambda t: t.confidence, reverse=True)

        return {
            'is_nsfw': is_nsfw,
            'is_nsfl': is_nsfl,
            'is_nsfp': is_nsfp,
            'valid_tags': valid_tags,
        }

    def prepare_row(self, item: Item) -> Optional[dict]:
        """Prepare a CSV row for an item."""
        result = self.process_tags(item)
        valid_tags = result['valid_tags']

        # Need minimum valid tags
        if len(valid_tags) < self.settings.min_valid_tags:
            return None

        row = {
            'id': item.id,
            'image': item.image,
            'is_nsfw': 'true' if result['is_nsfw'] else 'false',
            'is_nsfl': 'true' if result['is_nsfl'] else 'false',
            'is_nsfp': 'true' if result['is_nsfp'] else 'false',
        }

        # Add top N tags
        for i, tag in enumerate(valid_tags[:self.settings.min_valid_tags], 1):
            row[f'tag{i}'] = tag.tag
            row[f'confidence{i}'] = tag.confidence

        return row

    def run(self, output_file: Optional[Path] = None) -> tuple[PipelineStats, Path]:
        """
        Run the prepare pipeline.

        Returns:
            Tuple of (stats, output_path)
        """
        print_header(
            "📊 Prepare Dataset",
            "Processing items and generating training CSV"
        )

        # Determine output path
        if output_file is None:
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.settings.output_dir / f"{timestamp}_dataset.csv"

        print_info(f"Output file: {output_file}")
        print_info(f"Minimum valid tags: {self.settings.min_valid_tags}")

        # Build fieldnames
        fieldnames = ['id', 'image', 'is_nsfw', 'is_nsfl', 'is_nsfp']
        for i in range(1, self.settings.min_valid_tags + 1):
            fieldnames.extend([f'tag{i}', f'confidence{i}'])

        with SQLiteStorage(self.settings.db_path) as storage:
            # STEP 1: Analyze tag quality
            # - Identify trash tags (very common meta-tags like "repost")
            # - Build full tag count reference
            self.tag_counts, self.trash_tags = self.analyze_tag_quality(storage)

            print_info(f"Strategy: Blacklist {len(self.trash_tags)} trash tags, keep rare tags with high confidence (>{self.high_confidence_threshold})")

            total_items = storage.get_item_count()
            print_info(f"Processing {total_items:,} items")

            # Apply dev mode limit
            limit = self.settings.dev_limit if self.settings.dev_mode else None
            if self.settings.dev_mode:
                print_warning(f"Dev mode: limiting to {limit} items")

            output_file.parent.mkdir(parents=True, exist_ok=True)

            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

                progress = create_progress("Processing")
                with progress:
                    task = progress.add_task(
                        "[cyan]Processing items...",
                        total=limit or total_items
                    )

                    count = 0
                    for item in storage.iter_items_with_tags(
                        min_tags=self.settings.min_valid_tags,
                        image_only=True
                    ):
                        if limit and count >= limit:
                            break

                        progress.update(task, advance=1)
                        count += 1

                        row = self.prepare_row(item)
                        if row:
                            writer.writerow(row)
                            self.stats.items_processed += 1

                            # Verbose logging: show prepared item
                            if logger.isEnabledFor(logging.DEBUG):
                                tags = [row.get(f'tag{i}') for i in range(1, 6) if row.get(f'tag{i}')]
                                nsfw_flags = []
                                if row.get('is_nsfw') == 'true': nsfw_flags.append('NSFW')
                                if row.get('is_nsfl') == 'true': nsfw_flags.append('NSFL')
                                if row.get('is_nsfp') == 'true': nsfw_flags.append('NSFP')
                                flags_str = f" [{', '.join(nsfw_flags)}]" if nsfw_flags else ""
                                logger.debug(
                                    f"Prepared item {item.id}: {', '.join(tags[:3])}{flags_str}"
                                )
                        else:
                            self.stats.items_skipped += 1

            print_stats_table("Prepare Results", {
                "Items processed": self.stats.items_processed,
                "Items skipped": self.stats.items_skipped,
                "Trash tags filtered": len(self.trash_tags),
                "High confidence threshold": self.high_confidence_threshold,
                "Output file": str(output_file),
            })

            print_success("Dataset preparation complete!")

        return self.stats, output_file

