"""Prepare pipeline - prepare dataset for ML training."""

import csv
import logging
import re
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


class PreparePipeline:
    """Pipeline stage for preparing training dataset."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.stats = PipelineStats()

    def process_tags(self, item: Item) -> dict:
        """
        Process tags for an item.

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
            if VALID_TAG_REGEX.match(tag.tag):
                valid_tags.append(tag)

        # Sort by confidence
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
                        else:
                            self.stats.items_skipped += 1

            print_stats_table("Prepare Results", {
                "Items processed": self.stats.items_processed,
                "Items skipped": self.stats.items_skipped,
                "Output file": str(output_file),
            })

            print_success("Dataset preparation complete!")

        return self.stats, output_file

