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


def _preprocess_image_for_resnet(args: tuple) -> tuple[Optional[np.ndarray], Optional[str]]:
    """
    Load and preprocess a single image for ResNet50.

    This is a module-level function to work with multiprocessing (avoids GIL).
    All operations use numpy/PIL which release the GIL.

    Designed to be FAILSAFE - returns (None, error_reason) on any failure,
    never raises an exception. This is critical for long-running prepare jobs.

    Args:
        args: Tuple of (image_path, media_prefix, image_size)

    Returns:
        Tuple of (preprocessed_array, error_reason)
        - On success: (float32 array (H, W, 3), None)
        - On failure: (None, "reason string")
    """
    image_path, media_prefix_str, image_size = args

    try:
        from PIL import Image, UnidentifiedImageError
    except ImportError:
        return None, "pillow_not_installed"

    full_path = Path(media_prefix_str) / image_path

    # Check file exists
    if not full_path.exists():
        return None, "file_not_found"

    # Check file is not empty
    try:
        file_size = full_path.stat().st_size
        if file_size == 0:
            return None, "file_empty"
        if file_size < 100:  # Suspiciously small for an image
            return None, "file_too_small"
    except OSError as e:
        return None, f"stat_error:{e}"

    try:
        with Image.open(full_path) as img:
            # Verify the image can be read (catches truncated files)
            try:
                img.verify()
            except Exception:
                return None, "image_verify_failed"

        # Re-open after verify (verify() can only be called once)
        with Image.open(full_path) as img:
            # Force load to catch truncated/corrupted images
            try:
                img.load()
            except Exception:
                return None, "image_load_failed"

            # Check image has valid dimensions
            if img.width < 1 or img.height < 1:
                return None, "invalid_dimensions"

            # Convert to RGB (handles grayscale, RGBA, palette, etc.)
            try:
                img = img.convert('RGB')
            except Exception:
                return None, "rgb_convert_failed"

            # Resize with high-quality resampling
            try:
                img = img.resize(image_size, Image.LANCZOS)
            except Exception:
                return None, "resize_failed"

            # Convert to numpy array (uint8, RGB)
            try:
                arr = np.array(img, dtype=np.float32)
            except Exception:
                return None, "numpy_convert_failed"

        # Validate array shape
        if arr.shape != (image_size[1], image_size[0], 3):
            return None, f"wrong_shape:{arr.shape}"

        # ResNet50 preprocessing (matches keras.applications.resnet50.preprocess_input):
        # 1. Convert RGB to BGR
        arr = arr[..., ::-1]
        # 2. Zero-center by ImageNet mean
        arr -= IMAGENET_MEAN

        # Final sanity check
        if not np.isfinite(arr).all():
            return None, "non_finite_values"

        return arr, None

    except UnidentifiedImageError:
        return None, "unidentified_image"
    except MemoryError:
        return None, "memory_error"
    except Exception as e:
        # Catch-all for unexpected errors
        return None, f"unexpected:{type(e).__name__}"


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
    FAILSAFE: Individual image failures do not crash the pipeline.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.stats = PipelineStats()
        self.analysis: Optional[DataDrivenAnalysis] = None
        self._image_error_counts: dict[str, int] = {}  # Tracks image processing failures

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
        # FAILSAFE: Individual image failures don't crash the pipeline
        # =================================================================
        print_info(f"Embedding images ({image_size[0]}x{image_size[1]})...")

        items_before_embedding = len(df_result)
        df_embedded = self._embed_images(df_result, image_size)

        if df_embedded is None:
            print_error("Image embedding completely failed - no images could be processed")
            print_error("Check that:")
            print_error("  1. Media files exist in the media directory")
            print_error("  2. Image files are not corrupted")
            print_error("  3. Pillow is installed: pip install pillow")
            raise RuntimeError("Image embedding failed: no valid images")

        df_result = df_embedded
        items_after_embedding = len(df_result)

        # Update stats to reflect actual embedded count
        self.stats.items_processed = items_after_embedding

        if items_after_embedding < items_before_embedding:
            dropped = items_before_embedding - items_after_embedding
            print_warning(
                f"Dropped {dropped:,} items due to image failures "
                f"({dropped * 100 / items_before_embedding:.1f}%)"
            )

        if items_after_embedding == 0:
            print_error("No items remain after image embedding - cannot create dataset")
            raise RuntimeError("No valid items for dataset")

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

        Uses vectorized pandas operations for performance.
        """
        total_items = df['id'].nunique()

        # Vectorized tag explosion using pandas explode()
        # This is much faster than iterrows()
        df_tags = df[['id', 'tags_parsed']].copy()
        df_tags = df_tags.explode('tags_parsed').dropna(subset=['tags_parsed'])

        # Extract tag and confidence from the dict column
        df_tags['tag'] = df_tags['tags_parsed'].apply(lambda t: t['tag'].lower().strip())
        df_tags['confidence'] = df_tags['tags_parsed'].apply(lambda t: t.get('confidence', 0))
        df_tags = df_tags.drop(columns=['tags_parsed'])
        df_tags = df_tags.rename(columns={'id': 'item_id'})

        tags_df = df_tags

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

        FAILSAFE DESIGN:
        - Individual image failures do NOT crash the pipeline
        - Failed images are tracked and reported
        - Only successfully processed images are included in output

        Args:
            df: DataFrame with 'image' column containing relative paths
            image_size: Target size (width, height)

        Returns:
            DataFrame with 'image_data' column containing preprocessed float32 arrays,
            or None if no images could be loaded
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
        results = []
        progress = create_progress("Embedding")
        with progress:
            task = progress.add_task("[cyan]Processing images...", total=len(args_list))

            with mp.Pool(processes=num_workers) as pool:
                # Use imap for progress tracking
                chunksize = max(1, len(args_list) // (num_workers * 4))
                for result in pool.imap(_preprocess_image_for_resnet, args_list, chunksize=chunksize):
                    results.append(result)
                    progress.update(task, advance=1)

        # Separate successes from failures and collect error statistics
        valid_mask = []
        valid_arrays = []
        error_counts: dict[str, int] = {}

        for i, (arr, error) in enumerate(results):
            if arr is not None:
                valid_mask.append(True)
                valid_arrays.append(arr)
            else:
                valid_mask.append(False)
                error_reason = error or "unknown"
                error_counts[error_reason] = error_counts.get(error_reason, 0) + 1

        valid_count = len(valid_arrays)
        failed_count = len(results) - valid_count

        # Report failures with breakdown by reason
        if failed_count > 0:
            print_warning(f"Failed to load {failed_count:,} images ({failed_count * 100 / len(results):.1f}%)")
            # Show top error reasons
            sorted_errors = sorted(error_counts.items(), key=lambda x: -x[1])
            for reason, count in sorted_errors[:10]:  # Top 10 reasons
                print_info(f"  - {reason}: {count:,}")
            if len(sorted_errors) > 10:
                others = sum(c for _, c in sorted_errors[10:])
                print_info(f"  - (other reasons): {others:,}")

        if valid_count == 0:
            print_error("No images could be loaded! Check media directory and file integrity.")
            return None

        # Keep only rows with valid images
        df = df[valid_mask].copy()

        # Store as bytes (numpy array serialized) for Parquet
        # We use tobytes() which is very fast and stores raw float32 data
        df['image_data'] = [arr.tobytes() for arr in valid_arrays]

        # Calculate total size
        total_mb = sum(arr.nbytes for arr in valid_arrays) / 1024**2
        print_info(f"Embedded {valid_count:,} images ({total_mb:.0f} MB as float32)")
        print_info(f"Array shape per image: {valid_arrays[0].shape}, dtype: {valid_arrays[0].dtype}")

        # Store failure stats for metadata
        self.stats.items_failed = failed_count
        self._image_error_counts = error_counts

        return df

    def _save_metadata(self, output_file: Path, min_tags: int, trash_used: int,
                       image_size: tuple[int, int]):
        """Save analysis metadata including data-driven thresholds."""
        metadata = {
            'created': datetime.now().isoformat(),
            'total_items': self.stats.items_processed,
            'items_skipped': self.stats.items_skipped,
            'items_failed_images': self.stats.items_failed,
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
            # Image processing error breakdown (for debugging)
            'image_errors': self._image_error_counts if self._image_error_counts else None,
            # Sample of trash tags with reasons
            'trash_tags_sample': {
                t: self.analysis.trash_tag_reasons.get(t, 'unknown')
                for t in sorted(self.analysis.trash_tags)[:50]
            },
        }

        metadata_path = output_file.with_suffix('.meta.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    def export_huggingface(
        self,
        source_parquet: Path,
        output_dir: Path,
        dataset_name: str = "pr0gramm-sfw-tags",
        max_samples: Optional[int] = None,
    ) -> Path:
        """
        Export a SFW-only dataset in Hugging Face datasets format.

        IMPORTANT: This export ONLY includes SFW (Safe For Work) content.
        Any items flagged as NSFW, NSFL, or NSFP are excluded.

        Creates:
        - data/train.parquet - Training split (90%)
        - data/test.parquet - Test split (10%)
        - dataset_dict.json - Dataset metadata
        - README.md - Dataset card with documentation

        Args:
            source_parquet: Path to the source parquet file from prepare
            output_dir: Directory to create the HuggingFace dataset
            dataset_name: Name for the dataset
            max_samples: Optional limit on number of samples (for testing)

        Returns:
            Path to the created dataset directory
        """
        print_header(
            "🤗 Hugging Face Export",
            "Exporting SFW-only dataset for community sharing"
        )

        # Load source data
        print_info(f"Loading source: {source_parquet}")
        df = pd.read_parquet(source_parquet)
        total_items = len(df)
        print_info(f"Total items in source: {total_items:,}")

        # =================================================================
        # CRITICAL: Filter to SFW ONLY
        # This is non-negotiable for public sharing
        # =================================================================
        print_info("Filtering to SFW content only...")

        # Exclude ANY content that has NSFW, NSFL, or NSFP flags
        sfw_mask = (
            (df['is_nsfw'] == False) &
            (df['is_nsfl'] == False) &
            (df['is_nsfp'] == False)
        )
        df_sfw = df[sfw_mask].copy()

        excluded = total_items - len(df_sfw)
        print_info(f"SFW items: {len(df_sfw):,} ({len(df_sfw) * 100 / total_items:.1f}%)")
        print_info(f"Excluded (NSFW/NSFL/NSFP): {excluded:,}")

        if len(df_sfw) == 0:
            print_error("No SFW items found - cannot create export")
            raise ValueError("No SFW content available for export")

        # Optional sample limit (for testing)
        if max_samples and len(df_sfw) > max_samples:
            df_sfw = df_sfw.sample(n=max_samples, random_state=42)
            print_info(f"Sampled to {max_samples:,} items")

        # =================================================================
        # Prepare data for HuggingFace format
        # =================================================================
        print_info("Preparing HuggingFace format...")

        # Remove the NSFW flags from export (they're all False anyway)
        # Keep: id, image, tags, confidences
        # Convert image_data back to a usable format
        export_cols = ['id', 'image', 'tags', 'confidences']

        # Check if image_data exists and handle it
        if 'image_data' in df_sfw.columns:
            # Convert bytes back to base64 for portability
            import base64
            df_sfw['image_bytes_b64'] = df_sfw['image_data'].apply(
                lambda x: base64.b64encode(x).decode('utf-8') if x else None
            )
            export_cols.append('image_bytes_b64')

        df_export = df_sfw[export_cols].copy()

        # Ensure tags and confidences are lists (not numpy arrays)
        df_export['tags'] = df_export['tags'].apply(
            lambda x: list(x) if hasattr(x, '__iter__') and not isinstance(x, str) else x
        )
        df_export['confidences'] = df_export['confidences'].apply(
            lambda x: [float(c) for c in x] if hasattr(x, '__iter__') else x
        )

        # =================================================================
        # Split into train/test
        # =================================================================
        print_info("Creating train/test splits...")

        try:
            from sklearn.model_selection import train_test_split
            df_train, df_test = train_test_split(
                df_export,
                test_size=0.1,
                random_state=42
            )
        except ImportError:
            # Fallback: manual split if sklearn not installed
            print_warning("sklearn not installed, using simple split")
            df_shuffled = df_export.sample(frac=1, random_state=42)
            split_idx = int(len(df_shuffled) * 0.9)
            df_train = df_shuffled.iloc[:split_idx]
            df_test = df_shuffled.iloc[split_idx:]

        print_info(f"Train: {len(df_train):,} | Test: {len(df_test):,}")

        # =================================================================
        # Create output directory structure
        # =================================================================
        output_dir = Path(output_dir)
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Write parquet files
        print_info("Writing parquet files...")
        train_path = data_dir / "train.parquet"
        test_path = data_dir / "test.parquet"

        df_train.to_parquet(train_path, index=False)
        df_test.to_parquet(test_path, index=False)

        # =================================================================
        # Create dataset_dict.json (HuggingFace metadata)
        # =================================================================
        dataset_dict = {
            "splits": ["train", "test"],
            "data_files": {
                "train": "data/train.parquet",
                "test": "data/test.parquet"
            }
        }

        with open(output_dir / "dataset_dict.json", "w") as f:
            json.dump(dataset_dict, f, indent=2)

        # =================================================================
        # Create README.md (Dataset Card)
        # =================================================================
        print_info("Creating dataset card...")

        # Collect tag statistics for documentation
        all_tags = []
        for tags in df_export['tags']:
            all_tags.extend(tags)
        unique_tags = len(set(all_tags))

        readme_content = self._generate_dataset_card(
            dataset_name=dataset_name,
            total_samples=len(df_export),
            train_samples=len(df_train),
            test_samples=len(df_test),
            unique_tags=unique_tags,
            has_image_data='image_bytes_b64' in export_cols,
        )

        with open(output_dir / "README.md", "w") as f:
            f.write(readme_content)

        # =================================================================
        # Done
        # =================================================================
        print_stats_table("HuggingFace Export", {
            "Dataset": dataset_name,
            "Total samples": f"{len(df_export):,}",
            "Train split": f"{len(df_train):,}",
            "Test split": f"{len(df_test):,}",
            "Unique tags": f"{unique_tags:,}",
            "Content": "SFW ONLY ✓",
            "Output": str(output_dir),
        })

        print_success("HuggingFace export complete! 🤗")
        print_info(f"Upload with: huggingface-cli upload {dataset_name} {output_dir}")

        return output_dir

    def _generate_dataset_card(
        self,
        dataset_name: str,
        total_samples: int,
        train_samples: int,
        test_samples: int,
        unique_tags: int,
        has_image_data: bool,
    ) -> str:
        """Generate a HuggingFace dataset card (README.md)."""

        card = f'''---
license: cc-by-nc-4.0
task_categories:
  - image-classification
  - multi-label-classification
language:
  - de
tags:
  - image-tagging
  - multi-label
  - pr0gramm
  - german
size_categories:
  - {"100K<n<1M" if total_samples >= 100000 else "10K<n<100K" if total_samples >= 10000 else "1K<n<10K"}
---

# {dataset_name}

A dataset of **SFW (Safe For Work) images** with multi-label tags from pr0gramm.com, 
suitable for training image tagging models.

## ⚠️ Content Notice

**This dataset contains ONLY SFW (Safe For Work) content.**

All images flagged as NSFW, NSFL, or NSFP have been explicitly excluded.
This dataset is intended for research and educational purposes.

## Dataset Description

- **Total samples:** {total_samples:,}
- **Train split:** {train_samples:,} (90%)
- **Test split:** {test_samples:,} (10%)
- **Unique tags:** {unique_tags:,}
- **Language:** German (tags are in German)

## Dataset Structure

### Data Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | int | Unique item identifier |
| `image` | string | Relative image path (original source) |
| `tags` | list[string] | List of tag labels |
| `confidences` | list[float] | Confidence scores for each tag |
{"| `image_bytes_b64` | string | Base64-encoded preprocessed image (224x224, float32, BGR, ImageNet-normalized) |" if has_image_data else ""}

### Splits

| Split | Samples |
|-------|---------|
| train | {train_samples:,} |
| test | {test_samples:,} |

## Usage

```python
from datasets import load_dataset

# Load from HuggingFace Hub
dataset = load_dataset("{dataset_name}")

# Or load from local directory
dataset = load_dataset("parquet", data_dir="path/to/dataset")

# Access data
for sample in dataset["train"]:
    print(f"ID: {{sample['id']}}")
    print(f"Tags: {{sample['tags']}}")
    print(f"Confidences: {{sample['confidences']}}")
```

### Decode Image Data

{"" if not has_image_data else '''
If the dataset includes `image_bytes_b64`, you can decode it:

```python
import base64
import numpy as np

def decode_image(b64_string):
    """Decode base64 image to numpy array."""
    raw_bytes = base64.b64decode(b64_string)
    arr = np.frombuffer(raw_bytes, dtype=np.float32)
    return arr.reshape(224, 224, 3)  # BGR, ImageNet-normalized

# Usage
img_array = decode_image(sample["image_bytes_b64"])
```
'''}

## Preprocessing

Images have been preprocessed for ResNet50:
- Resized to 224×224
- Converted to float32
- RGB → BGR channel order
- Zero-centered by ImageNet mean [103.939, 116.779, 123.68]

## Intended Use

This dataset is intended for:
- Training multi-label image classification models
- Research on image tagging systems
- Educational purposes

## Limitations

- Tags are user-generated and may contain noise
- Tag distribution follows a long-tail pattern
- Language is primarily German
- Only SFW content is included

## License

This dataset is released under CC-BY-NC-4.0 (Creative Commons Attribution-NonCommercial 4.0).

- ✓ Share and adapt with attribution
- ✓ Non-commercial use only
- ✗ No commercial use without permission

## Citation

```bibtex
@dataset{{{dataset_name.replace("-", "_")}}},
  title = {{{dataset_name}}},
  year = {{{datetime.now().year}}},
  publisher = {{HuggingFace}},
  note = {{SFW-only image tagging dataset from pr0gramm}}
}}
```

## Acknowledgments

Data sourced from pr0gramm.com. This dataset contains only SFW content
and is created for research/educational purposes.

---

*Generated on {datetime.now().strftime("%Y-%m-%d")} by pr0loader*
'''
        return card
