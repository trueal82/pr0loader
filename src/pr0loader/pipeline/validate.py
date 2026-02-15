"""Validate pipeline - evaluate model performance on test set."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from pr0loader.config import Settings
from pr0loader.models import PipelineStats
from pr0loader.utils.console import (
    print_header,
    print_stats_table,
    print_success,
    print_info,
    print_warning,
    print_error,
    console,
)

logger = logging.getLogger(__name__)


class ValidatePipeline:
    """Pipeline stage for validating/evaluating the trained model."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.stats = PipelineStats()

    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available."""
        try:
            import tensorflow as tf
            gpu_devices = tf.config.list_physical_devices('GPU')
            if gpu_devices:
                logger.info(f"GPU available for validation: {len(gpu_devices)} device(s)")
            return True
        except ImportError:
            import sys
            print_error("TensorFlow not installed. Install with: pip install 'pr0loader[ml]'")
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            if sys.version_info >= (3, 13):
                print_warning(f"You're using Python {python_version}, but TensorFlow requires Python 3.10-3.12")
                print_info("Create a venv with Python 3.12: py -3.12 -m venv .venv")
            else:
                print_info("For GPU support: Install CUDA toolkit first, then tensorflow")
            return False

    def split_dataset(
        self,
        csv_path: Path,
        train_ratio: float = 0.8,
        random_seed: int = 42
    ) -> tuple[Path, Path]:
        """
        Split dataset into train and test sets.

        Args:
            csv_path: Path to the full dataset CSV
            train_ratio: Ratio of data for training (default 0.8)
            random_seed: Random seed for reproducibility

        Returns:
            Tuple of (train_path, test_path)
        """
        print_info(f"Splitting dataset: {csv_path}")
        print_info(f"Train ratio: {train_ratio:.0%}, Test ratio: {(1-train_ratio):.0%}")

        # Load dataset
        df = pd.read_csv(csv_path)
        total_samples = len(df)

        # Verbose logging: show dataset info
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Total samples: {total_samples}")
            logger.debug(f"Random seed: {random_seed}")

        # Shuffle and split
        df_shuffled = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        split_idx = int(total_samples * train_ratio)

        train_df = df_shuffled[:split_idx]
        test_df = df_shuffled[split_idx:]

        # Verbose logging: show split details
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Split at index: {split_idx}")
            logger.debug(f"Train samples: {len(train_df)}")
            logger.debug(f"Test samples: {len(test_df)}")

        # Save splits
        train_path = csv_path.parent / f"{csv_path.stem}_train.csv"
        test_path = csv_path.parent / f"{csv_path.stem}_test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print_success(f"Train set: {len(train_df)} samples → {train_path.name}")
        print_success(f"Test set: {len(test_df)} samples → {test_path.name}")

        return train_path, test_path

    def evaluate_model(
        self,
        model_path: Path,
        test_csv_path: Path,
        top_k: int = 5
    ) -> dict:
        """
        Evaluate model on test set.

        Args:
            model_path: Path to trained model
            test_csv_path: Path to test CSV
            top_k: Number of top predictions to consider

        Returns:
            Dictionary with evaluation metrics
        """
        print_header("✅ Model Validation", "Evaluating model on test set")

        if not self._check_tensorflow():
            return {}

        import tensorflow as tf
        from tensorflow import keras

        # Load model
        print_info(f"Loading model: {model_path}")
        try:
            model = keras.models.load_model(model_path)
        except Exception as e:
            print_error(f"Failed to load model: {e}")
            return {}

        # Load tag mapping
        mapping_path = model_path.with_suffix('.tags.json')
        if not mapping_path.exists():
            print_error(f"Tag mapping not found: {mapping_path}")
            return {}

        with open(mapping_path) as f:
            mapping_data = json.load(f)
            idx_to_tag = {int(k): v for k, v in mapping_data['idx_to_tag'].items()}
            num_classes = mapping_data['num_classes']

        print_info(f"Number of classes: {num_classes}")

        # Load test data
        print_info(f"Loading test data: {test_csv_path}")
        df = pd.read_csv(test_csv_path)
        df = df[df['image'].notnull()]
        print_info(f"Test samples: {len(df)}")

        # Build tag vocabulary
        tag_columns = [f'tag{i}' for i in range(1, 6)]
        all_tags = df[tag_columns].values.flatten()
        unique_tags = sorted(set(tag for tag in all_tags if pd.notnull(tag)))
        tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}

        # Encode ground truth labels
        def encode_tags(row):
            label = np.zeros(num_classes, dtype=np.float32)
            for col in tag_columns:
                tag = row[col]
                if pd.notnull(tag) and tag in tag_to_idx:
                    label[tag_to_idx[tag]] = 1.0
            return label

        y_true = np.array([encode_tags(row) for _, row in df.iterrows()])

        # Prepare image paths
        image_paths = [
            str(self.settings.filesystem_prefix / img)
            for img in df['image'].values
        ]

        # Create dataset
        def load_and_preprocess_image(path):
            img = tf.io.read_file(path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, [224, 224])
            img = tf.cast(img, tf.float32)
            img = keras.applications.resnet50.preprocess_input(img)
            return img

        def load_image_wrapper(path_tensor):
            return tf.py_function(
                func=lambda p: load_and_preprocess_image(p.numpy().decode('utf-8')),
                inp=[path_tensor],
                Tout=tf.float32
            )

        test_ds = tf.data.Dataset.from_tensor_slices(image_paths)
        test_ds = test_ds.map(load_image_wrapper, num_parallel_calls=tf.data.AUTOTUNE)
        test_ds = test_ds.batch(self.settings.batch_size)
        test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

        # Predict
        print_info("Running predictions...")

        # Verbose logging: show test set info
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Test set size: {len(df)} samples")
            logger.debug(f"Number of classes: {num_classes}")

        y_pred = model.predict(test_ds, verbose=1)

        # Verbose logging: show some predictions
        if logger.isEnabledFor(logging.DEBUG):
            for i in range(min(3, len(y_pred))):
                true_tags = [idx_to_tag.get(idx, f"tag_{idx}") for idx in np.where(y_true[i] == 1)[0]]
                top_pred_indices = np.argsort(y_pred[i])[-5:][::-1]
                pred_tags = [
                    f"{idx_to_tag.get(idx, f'tag_{idx}')}({y_pred[i][idx]:.3f})"
                    for idx in top_pred_indices
                ]
                logger.debug(f"Sample {i+1}: True={', '.join(true_tags[:3])} | Pred={', '.join(pred_tags[:3])}")

        # Calculate metrics
        metrics = self._calculate_metrics(y_true, y_pred, top_k)

        # Display results
        self._display_results(metrics, idx_to_tag, y_true, y_pred, top_k)

        return metrics

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_k: int = 5
    ) -> dict:
        """Calculate evaluation metrics."""
        print_info("Calculating metrics...")

        metrics = {}

        # Binary classification metrics (per-tag)
        # Consider prediction positive if probability > 0.5
        y_pred_binary = (y_pred > 0.5).astype(int)

        # True positives, false positives, false negatives per sample
        tp = np.sum((y_true == 1) & (y_pred_binary == 1), axis=1)
        fp = np.sum((y_true == 0) & (y_pred_binary == 1), axis=1)
        fn = np.sum((y_true == 1) & (y_pred_binary == 0), axis=1)

        # Precision, recall, F1 per sample (micro-averaged)
        precision = np.mean(tp / (tp + fp + 1e-10))
        recall = np.mean(tp / (tp + fn + 1e-10))
        f1 = 2 * precision * recall / (precision + recall + 1e-10)

        metrics['precision'] = float(precision)
        metrics['recall'] = float(recall)
        metrics['f1_score'] = float(f1)

        # Hamming accuracy (exact match per tag)
        hamming_acc = np.mean(y_true == y_pred_binary)
        metrics['hamming_accuracy'] = float(hamming_acc)

        # Top-K accuracy: at least one of top-K predictions is in ground truth
        top_k_correct = 0
        for i in range(len(y_true)):
            true_labels = set(np.where(y_true[i] == 1)[0])
            top_k_preds = set(np.argsort(y_pred[i])[-top_k:])
            if len(true_labels & top_k_preds) > 0:
                top_k_correct += 1

        metrics[f'top_{top_k}_accuracy'] = float(top_k_correct / len(y_true))

        # Mean average precision (mAP)
        # For each sample, calculate average precision
        aps = []
        for i in range(len(y_true)):
            true_labels = set(np.where(y_true[i] == 1)[0])
            if len(true_labels) == 0:
                continue

            # Sort predictions by score
            sorted_indices = np.argsort(y_pred[i])[::-1]

            # Calculate AP
            num_correct = 0
            precision_sum = 0.0
            for k, idx in enumerate(sorted_indices[:top_k], 1):
                if idx in true_labels:
                    num_correct += 1
                    precision_sum += num_correct / k

            ap = precision_sum / min(len(true_labels), top_k)
            aps.append(ap)

        metrics['mean_average_precision'] = float(np.mean(aps))

        return metrics

    def _display_results(
        self,
        metrics: dict,
        idx_to_tag: dict,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        top_k: int
    ):
        """Display evaluation results."""
        from rich.table import Table
        from rich import box

        console.print()

        # Main metrics table
        table = Table(title="📊 Validation Results", box=box.ROUNDED)
        table.add_column("Metric", style="cyan", width=30)
        table.add_column("Value", style="green", justify="right", width=15)

        table.add_row("Precision", f"{metrics['precision']:.4f}")
        table.add_row("Recall", f"{metrics['recall']:.4f}")
        table.add_row("F1 Score", f"{metrics['f1_score']:.4f}")
        table.add_row("Hamming Accuracy", f"{metrics['hamming_accuracy']:.4f}")
        table.add_row(f"Top-{top_k} Accuracy", f"{metrics[f'top_{top_k}_accuracy']:.4f}")
        table.add_row("Mean Average Precision", f"{metrics['mean_average_precision']:.4f}")

        console.print(table)
        console.print()

        # Sample predictions (first 3)
        console.print("[bold cyan]Sample Predictions:[/bold cyan]")
        for i in range(min(3, len(y_true))):
            true_tags = [idx_to_tag.get(idx, f"tag_{idx}") for idx in np.where(y_true[i] == 1)[0]]
            top_pred_indices = np.argsort(y_pred[i])[-top_k:][::-1]
            pred_tags = [
                f"{idx_to_tag.get(idx, f'tag_{idx}')} ({y_pred[i][idx]:.3f})"
                for idx in top_pred_indices
            ]

            console.print(f"\n[dim]Sample {i+1}:[/dim]")
            console.print(f"  True: {', '.join(true_tags[:5])}")
            console.print(f"  Pred: {', '.join(pred_tags)}")

    def run(
        self,
        model_path: Optional[Path] = None,
        test_csv_path: Optional[Path] = None,
        top_k: int = 5
    ) -> dict:
        """
        Run the validation pipeline.

        Args:
            model_path: Path to trained model (default: settings.model_path)
            test_csv_path: Path to test CSV (default: latest test set)
            top_k: Number of top predictions to evaluate

        Returns:
            Dictionary with evaluation metrics
        """
        # Use defaults if not provided
        if model_path is None:
            model_path = self.settings.model_path

        if not model_path.exists():
            print_error(f"Model not found: {model_path}")
            return {}

        if test_csv_path is None:
            # Find most recent test dataset
            test_datasets = sorted(
                self.settings.output_dir.glob("*_test.csv"),
                key=lambda p: p.stat().st_mtime,
                reverse=True
            )
            if test_datasets:
                test_csv_path = test_datasets[0]
            else:
                print_error("No test dataset found. Run prepare with split first.")
                return {}

        return self.evaluate_model(model_path, test_csv_path, top_k)
