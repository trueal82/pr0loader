"""Prepare pipeline - prepare dataset for ML training.

Fully data-driven approach:
- ALL thresholds calculated from actual data distribution
- Trash tag detection based on statistical analysis, not hardcoded patterns
- Uses pandas for vectorized analytics
- No static assumptions - everything adapts to the dataset
- Optional image embedding with ResNet50 preprocessing (ready for training)
"""

import json
import logging
import multiprocessing as mp
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from pr0loader.config import Settings
from pr0loader.models import PipelineStats
from pr0loader.storage import SQLiteStorage
from pr0loader.utils.console import (
    create_progress,
    print_header,
    print_stats_table,
    print_success,
    print_info,
    print_warning,
    print_error,
)

logger = logging.getLogger(__name__)

# Content flags - these ARE hardcoded because they're part of pr0gramm's data model
# They describe content rating, not content itself
CONTENT_FLAGS = frozenset({'nsfw', 'nsfl', 'nsfp', 'sfw'})

# ImageNet mean values for ResNet50 preprocessing (BGR order)
IMAGENET_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def _preprocess_image_for_resnet(args: tuple) -> Optional[np.ndarray]:
    """
    Load and preprocess a single image for ResNet50.

    This is a module-level function to work with multiprocessing (avoids GIL).
    All operations use numpy/PIL which release the GIL.

    Args:
        args: Tuple of (image_path, media_prefix, image_size)

    Returns:
        Preprocessed float32 array (H, W, 3) ready for ResNet50, or None if failed
    """
    image_path, media_prefix_str, image_size = args

    try:
        from PIL import Image

        full_path = Path(media_prefix_str) / image_path
        if not full_path.exists():
            return None

        with Image.open(full_path) as img:
            # Convert to RGB
            img = img.convert('RGB')
            # Resize with high-quality resampling
            img = img.resize(image_size, Image.LANCZOS)
            # Convert to numpy array (uint8, RGB)
            arr = np.array(img, dtype=np.float32)

        # ResNet50 preprocessing (matches keras.applications.resnet50.preprocess_input):
        # 1. Convert RGB to BGR
        arr = arr[..., ::-1]
        # 2. Zero-center by ImageNet mean
        arr -= IMAGENET_MEAN

        return arr

    except Exception:
        return None


@dataclass
class DataDrivenAnalysis:
    """Results of data-driven tag analysis."""
    total_items: int
    total_tags: int
    unique_tags: int

    # DataFrames for analysis
    tag_stats: pd.DataFrame  # Per-tag statistics
    confidence_stats: dict[str, float]  # Confidence distribution

    # Computed thresholds (all from data)
    high_confidence_threshold: float
    min_corpus_frequency: int
    trash_tags: frozenset[str]

    # Trash tag breakdown for reporting
    trash_tag_reasons: dict[str, str] = field(default_factory=dict)


class PreparePipeline:
    """Pipeline stage for preparing training dataset.

    Fully data-driven - all thresholds computed from actual data.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.stats = PipelineStats()
        self.analysis: Optional[DataDrivenAnalysis] = None

    def _check_concurrent_access(self) -> bool:
        """Check if another pr0loader process is running."""
        try:
            result = subprocess.run(
                ['pgrep', '-f', 'pr0loader.*(fetch|sync)'],
                capture_output=True, text=True
            )
            if result.returncode == 0 and result.stdout.strip():
                return True
        except Exception:
            pass

        wal_file = Path(str(self.settings.db_path) + '-wal')
        if wal_file.exists():
            try:
                if wal_file.stat().st_size > 10 * 1024 * 1024:
                    return True
            except Exception:
                pass
        return False

    def run(self, output_file: Optional[Path] = None) -> tuple[PipelineStats, Path]:
        """Run the prepare pipeline with fully data-driven analytics.

        Images are always preprocessed and embedded for training.

        Args:
            output_file: Output Parquet file path
        """
        print_header(
            "📊 Prepare Dataset",
            "Data-driven analytics with pandas"
        )

        if self._check_concurrent_access():
            print_warning("⚠️  Concurrent fetch/sync detected - reads may be slow")

        min_tags = self.settings.min_valid_tags
        image_size = self.settings.image_size  # (224, 224) typically

        if output_file is None:
            self.settings.output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = self.settings.output_dir / f"{timestamp}_dataset.parquet"

        if output_file.suffix != '.parquet':
            output_file = output_file.with_suffix('.parquet')

        print_info(f"Output: {output_file}")
        print_info(f"Min tags: {min_tags}")

        # =================================================================
        # STEP 1: Load raw data
        # =================================================================
        with SQLiteStorage(self.settings.db_path) as storage:
            print_info("Loading data...")

            if self.settings.dev_mode:
                sql_limit = self.settings.dev_limit * 2
                query = f'SELECT id, image, tags_data FROM items WHERE tags_data IS NOT NULL LIMIT {sql_limit}'
                print_warning(f"Dev mode: LIMIT {sql_limit}")
            else:
                query = 'SELECT id, image, tags_data FROM items WHERE tags_data IS NOT NULL'

            df = pd.read_sql_query(query, storage.conn)
            print_info(f"Loaded {len(df):,} rows ({df.memory_usage(deep=True).sum() / 1024**2:.1f} MB)")

        # =================================================================
        # STEP 2: Filter to images only
        # =================================================================
        print_info("Filtering images...")
        df = df[df['image'].str.match(r'.*\.(jpg|jpeg|png)$', case=False, na=False)].reset_index(drop=True)
        print_info(f"Images: {len(df):,}")

        if self.settings.dev_mode:
            df = df.head(self.settings.dev_limit)
            print_warning(f"Dev limit: {len(df):,}")

        # =================================================================
        # STEP 3: Parse JSON and explode into tag DataFrame
        # =================================================================
        print_info("Parsing and exploding tags...")
        df['tags_parsed'] = [json.loads(x) if x else [] for x in df['tags_data'].values]
        df = df.drop(columns=['tags_data'])

        # Filter by minimum tag count
        df['tag_count'] = df['tags_parsed'].apply(len)
        df = df[df['tag_count'] >= min_tags].reset_index(drop=True)
        print_info(f"After min_tags filter: {len(df):,}")

        if df.empty:
            print_warning("No eligible items!")
            return self.stats, output_file

        # =================================================================
        # STEP 4: Build tag statistics DataFrame (fully vectorized)
        # =================================================================
        print_info("Building tag statistics...")
        self.analysis = self._analyze_tags_datadriven(df)

        # =================================================================
        # STEP 5: Process items using data-driven thresholds
        # =================================================================
        print_info("Processing items...")
        df_result = self._process_items_datadriven(df, min_tags)

        self.stats.items_processed = len(df_result)
        self.stats.items_skipped = len(df) - len(df_result)
        trash_used = df_result['trash_used'].sum() if 'trash_used' in df_result.columns else 0

        # =================================================================
        # STEP 6: Embed preprocessed images (always done for training)
        # =================================================================
        print_info(f"Embedding images ({image_size[0]}x{image_size[1]})...")
        df_embedded = self._embed_images(df_result, image_size)
        if df_embedded is not None:
            df_result = df_embedded
        else:
            print_error("Image embedding failed - cannot create training dataset")
            raise RuntimeError("Image embedding failed")

        # =================================================================
        # STEP 7: Write output
        # =================================================================
        print_info(f"Writing {len(df_result):,} records...")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        cols = ['id', 'image', 'is_nsfw', 'is_nsfl', 'is_nsfp', 'tags', 'confidences', 'image_data']

        table = pa.Table.from_pandas(df_result[cols], preserve_index=False)
        pq.write_table(table, output_file, compression='snappy')

        self._save_metadata(output_file, min_tags, int(trash_used), image_size)

        print_stats_table("Results", {
            "Processed": self.stats.items_processed,
            "Skipped": self.stats.items_skipped,
            "Unique tags": self.analysis.unique_tags,
            "Trash tags": len(self.analysis.trash_tags),
            "Trash kept": int(trash_used),
            "High conf threshold": f"{self.analysis.high_confidence_threshold:.3f}",
            "Min corpus freq": self.analysis.min_corpus_frequency,
        })

        print_success("Done!")
        return self.stats, output_file

    def _analyze_tags_datadriven(self, df: pd.DataFrame) -> DataDrivenAnalysis:
        """
        Analyze tags using pandas - ALL thresholds derived from data.

        Key metrics computed:
        - Tag frequency (how often a tag appears across all items)
        - Tag document frequency (in how many items does the tag appear)
        - Tag confidence distribution (mean, std, percentiles)
        - IDF score (inverse document frequency - rarer = higher)
        """
        # Explode tags into rows for vectorized analysis
        tags_exploded = []
        for idx, row in df.iterrows():
            item_id = row['id']
            for t in row['tags_parsed']:
                tag_lower = t['tag'].lower().strip()
                conf = t.get('confidence', 0)
                tags_exploded.append({
                    'item_id': item_id,
                    'tag': tag_lower,
                    'confidence': conf
                })

        tags_df = pd.DataFrame(tags_exploded)
        total_items = df['id'].nunique()

        print_info(f"Total tag occurrences: {len(tags_df):,}")

        # =================================================================
        # Compute per-tag statistics using groupby
        # =================================================================
        tag_stats = tags_df.groupby('tag').agg(
            frequency=('tag', 'count'),  # Total occurrences
            doc_frequency=('item_id', 'nunique'),  # Items containing this tag
            conf_mean=('confidence', 'mean'),
            conf_std=('confidence', 'std'),
            conf_min=('confidence', 'min'),
            conf_max=('confidence', 'max'),
        ).reset_index()

        # Fill NaN std with 0 (tags that appear once)
        tag_stats['conf_std'] = tag_stats['conf_std'].fillna(0)

        # Compute IDF: log(total_items / doc_frequency)
        # Higher IDF = rarer tag = more informative
        tag_stats['idf'] = np.log(total_items / tag_stats['doc_frequency'])

        # Compute document frequency ratio (what % of items have this tag)
        tag_stats['doc_freq_ratio'] = tag_stats['doc_frequency'] / total_items

        print_info(f"Unique tags: {len(tag_stats):,}")

        # =================================================================
        # Compute confidence thresholds from distribution
        # =================================================================
        conf_percentiles = tags_df['confidence'].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        confidence_stats = {
            'mean': tags_df['confidence'].mean(),
            'std': tags_df['confidence'].std(),
            'median': conf_percentiles[0.5],
            'p75': conf_percentiles[0.75],
            'p90': conf_percentiles[0.9],
        }

        # High confidence threshold: 75th percentile of all confidences
        high_conf_threshold = conf_percentiles[0.75]

        print_info(f"Confidence distribution: median={confidence_stats['median']:.3f}, p75={high_conf_threshold:.3f}")

        # =================================================================
        # Compute minimum corpus frequency threshold from distribution
        # Tags appearing less than this are considered too rare
        # Use 25th percentile of tag frequencies (bottom 25% are "rare")
        # =================================================================
        freq_p25 = tag_stats['frequency'].quantile(0.25)
        min_corpus_freq = max(3, int(freq_p25))  # At least 3 to avoid singleton noise

        print_info(f"Min corpus frequency: {min_corpus_freq} (p25={freq_p25:.0f})")

        # =================================================================
        # Identify trash tags using DATA-DRIVEN criteria
        # =================================================================
        trash_tags = set()
        trash_reasons = {}

        # Criterion 1: Very high document frequency (>20% of items)
        # These are likely meta-tags, not content descriptors
        # The 20% threshold is the 95th percentile of doc_freq_ratio
        high_freq_threshold = tag_stats['doc_freq_ratio'].quantile(0.95)
        high_freq_tags = set(tag_stats[tag_stats['doc_freq_ratio'] > high_freq_threshold]['tag'])
        # Exclude content flags from this
        high_freq_tags -= CONTENT_FLAGS
        for t in high_freq_tags:
            trash_tags.add(t)
            trash_reasons[t] = f"high_doc_freq (>{high_freq_threshold:.1%})"

        # Criterion 2: Very low IDF AND low confidence
        # Tags that are common but not trusted by users
        low_idf_threshold = tag_stats['idf'].quantile(0.10)  # Bottom 10% IDF
        low_conf_threshold = confidence_stats['median']  # Below median confidence
        low_quality_mask = (
            (tag_stats['idf'] < low_idf_threshold) &
            (tag_stats['conf_mean'] < low_conf_threshold)
        )
        low_quality_tags = set(tag_stats[low_quality_mask]['tag']) - CONTENT_FLAGS
        for t in low_quality_tags:
            if t not in trash_tags:
                trash_tags.add(t)
                trash_reasons[t] = "low_idf_low_conf"

        # Criterion 3: High confidence variance indicates inconsistent tagging
        # Tags where users disagree strongly are less reliable
        # Use tags with std > 2x mean std
        if len(tag_stats) > 100:  # Only if we have enough data
            mean_std = tag_stats['conf_std'].mean()
            high_variance_mask = tag_stats['conf_std'] > (2 * mean_std)
            high_var_tags = set(tag_stats[high_variance_mask]['tag']) - CONTENT_FLAGS
            for t in high_var_tags:
                if t not in trash_tags:
                    trash_tags.add(t)
                    trash_reasons[t] = "high_conf_variance"

        # Criterion 4: Single-item tags with low confidence
        # Tags that appear in only one item AND have low confidence
        singleton_low_conf = tag_stats[
            (tag_stats['doc_frequency'] == 1) &
            (tag_stats['conf_mean'] < confidence_stats['median'])
        ]['tag']
        for t in singleton_low_conf:
            if t not in trash_tags:
                trash_tags.add(t)
                trash_reasons[t] = "singleton_low_conf"

        print_info(f"Trash tags identified: {len(trash_tags):,}")
        print_info(f"  - High doc frequency: {len(high_freq_tags)}")
        print_info(f"  - Low IDF + low conf: {len(low_quality_tags)}")

        return DataDrivenAnalysis(
            total_items=total_items,
            total_tags=len(tags_df),
            unique_tags=len(tag_stats),
            tag_stats=tag_stats,
            confidence_stats=confidence_stats,
            high_confidence_threshold=high_conf_threshold,
            min_corpus_frequency=min_corpus_freq,
            trash_tags=frozenset(trash_tags),
            trash_tag_reasons=trash_reasons,
        )

    def _process_items_datadriven(self, df: pd.DataFrame, min_tags: int) -> pd.DataFrame:
        """Process items using data-driven thresholds."""
        trash_tags = self.analysis.trash_tags
        high_conf = self.analysis.high_confidence_threshold
        min_freq = self.analysis.min_corpus_frequency

        # Build tag frequency lookup from analysis
        tag_freq = dict(zip(
            self.analysis.tag_stats['tag'],
            self.analysis.tag_stats['frequency']
        ))

        ids = df['id'].values
        images = df['image'].values
        tags_parsed = df['tags_parsed'].values

        results = []

        for i in range(len(df)):
            tags_list = tags_parsed[i]

            is_nsfw = is_nsfl = is_nsfp = False
            normalized: dict[str, float] = {}

            for t in tags_list:
                tag_lower = t['tag'].lower().strip()
                conf = t.get('confidence', 0)

                if tag_lower == 'nsfw':
                    is_nsfw = True
                elif tag_lower == 'nsfl':
                    is_nsfl = True
                elif tag_lower == 'nsfp':
                    is_nsfp = True
                elif tag_lower != 'sfw':
                    normalized[tag_lower] = normalized.get(tag_lower, 0) + conf

            good = []
            trash_kept = []

            for tag, total_conf in normalized.items():
                if tag in trash_tags:
                    trash_kept.append((tag, total_conf))
                elif total_conf >= high_conf or tag_freq.get(tag, 0) >= min_freq:
                    good.append((tag, total_conf))

            good.sort(key=lambda x: x[1], reverse=True)
            trash_kept.sort(key=lambda x: x[1], reverse=True)

            valid = good.copy()
            trash_used = 0
            if len(valid) < min_tags and trash_kept:
                needed = min_tags - len(valid)
                valid.extend(trash_kept[:needed])
                trash_used = min(needed, len(trash_kept))
                valid.sort(key=lambda x: x[1], reverse=True)

            if len(valid) >= min_tags:
                results.append({
                    'id': ids[i],
                    'image': images[i],
                    'is_nsfw': is_nsfw,
                    'is_nsfl': is_nsfl,
                    'is_nsfp': is_nsfp,
                    'tags': [t[0] for t in valid[:min_tags]],
                    'confidences': [t[1] for t in valid[:min_tags]],
                    'trash_used': trash_used,
                })

        return pd.DataFrame(results)

    def _embed_images(self, df: pd.DataFrame, image_size: tuple[int, int]) -> Optional[pd.DataFrame]:
        """
        Load and preprocess images for ResNet50, embedding as numpy arrays.

        Uses multiprocessing to bypass GIL - each worker runs in a separate process.
        Images are preprocessed exactly as ResNet50 expects:
        - Resized to target size
        - Converted RGB -> BGR
        - Zero-centered by ImageNet mean

        The resulting arrays can be fed directly to the model without further preprocessing.

        Args:
            df: DataFrame with 'image' column containing relative paths
            image_size: Target size (width, height)

        Returns:
            DataFrame with 'image_data' column containing preprocessed float32 arrays
        """
        try:
            from PIL import Image
        except ImportError:
            print_warning("Pillow not installed - cannot embed images")
            print_info("Install with: pip install pillow")
            return None

        media_prefix = str(self.settings.filesystem_prefix)
        image_paths = df['image'].tolist()

        # Prepare arguments for multiprocessing
        args_list = [(path, media_prefix, image_size) for path in image_paths]

        # Determine number of workers (leave some cores for system)
        num_workers = max(1, mp.cpu_count() - 2)
        print_info(f"Loading {len(df):,} images with {num_workers} parallel workers...")

        # Use multiprocessing Pool (bypasses GIL completely)
        # Each worker is a separate process with its own Python interpreter
        progress = create_progress("Embedding")
        with progress:
            task = progress.add_task("[cyan]Processing images...", total=len(args_list))

            with mp.Pool(processes=num_workers) as pool:
                image_arrays = []
                # Use imap for progress tracking
                chunksize = max(1, len(args_list) // (num_workers * 4))
                for result in pool.imap(_preprocess_image_for_resnet, args_list, chunksize=chunksize):
                    image_arrays.append(result)
                    progress.update(task, advance=1)

        # Filter out failed loads
        valid_mask = [arr is not None for arr in image_arrays]
        valid_count = sum(valid_mask)
        failed_count = len(valid_mask) - valid_count

        if failed_count > 0:
            print_warning(f"Failed to load {failed_count:,} images")

        if valid_count == 0:
            print_warning("No images could be loaded!")
            return None

        # Keep only rows with valid images
        df = df[valid_mask].copy()
        valid_arrays = [arr for arr, valid in zip(image_arrays, valid_mask) if valid]

        # Stack into single numpy array for efficient storage
        # Shape: (N, H, W, 3) - float32
        stacked = np.stack(valid_arrays, axis=0)

        # Store as bytes (numpy array serialized) for Parquet
        # We use tobytes() which is very fast and stores raw float32 data
        df['image_data'] = [arr.tobytes() for arr in valid_arrays]

        # Calculate total size
        total_mb = stacked.nbytes / 1024**2
        print_info(f"Embedded {len(valid_arrays):,} images ({total_mb:.0f} MB as float32)")
        print_info(f"Array shape per image: {valid_arrays[0].shape}, dtype: {valid_arrays[0].dtype}")

        return df

    def _save_metadata(self, output_file: Path, min_tags: int, trash_used: int,
                       image_size: tuple[int, int]):
        """Save analysis metadata including data-driven thresholds."""
        metadata = {
            'created': datetime.now().isoformat(),
            'total_items': self.stats.items_processed,
            'items_skipped': self.stats.items_skipped,
            'min_tags': min_tags,
            'unique_tags': self.analysis.unique_tags,
            'total_tag_occurrences': self.analysis.total_tags,
            # Data-driven thresholds (for reproducibility)
            'thresholds': {
                'high_confidence': self.analysis.high_confidence_threshold,
                'min_corpus_frequency': self.analysis.min_corpus_frequency,
            },
            'confidence_stats': self.analysis.confidence_stats,
            'trash_tags_count': len(self.analysis.trash_tags),
            'trash_tags_kept_for_min': trash_used,
            # Image embedding info (always embedded)
            'images_embedded': True,
            'image_size': list(image_size),
            'image_format': {
                'dtype': 'float32',
                'shape': [image_size[0], image_size[1], 3],
                'preprocessing': 'resnet50',  # RGB->BGR, zero-centered by ImageNet mean
                'color_order': 'BGR',
                'imagenet_mean': [103.939, 116.779, 123.68],
            },
            # Sample of trash tags with reasons
            'trash_tags_sample': {
                t: self.analysis.trash_tag_reasons.get(t, 'unknown')
                for t in sorted(self.analysis.trash_tags)[:50]
            },
        }

        metadata_path = output_file.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

