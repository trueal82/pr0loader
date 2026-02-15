"""Train pipeline - train tag prediction model.

Reads Parquet datasets from prepare pipeline with pre-processed images.
Images are already:
- Resized to target size (224x224)
- Preprocessed for ResNet50 (RGB->BGR, ImageNet mean subtracted)
- Stored as float32 arrays

No image manipulation happens here - that's all done in prepare.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq

from pr0loader.config import Settings
from pr0loader.utils.console import (
    print_header,
    print_stats_table,
    print_success,
    print_info,
    print_warning,
    print_error,
    is_headless,
)

logger = logging.getLogger(__name__)


class TrainPipeline:
    """Pipeline stage for training the tag prediction model.

    Reads Parquet datasets prepared by PreparePipeline.
    Images must be pre-processed and embedded in the Parquet file.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.tag_to_idx: dict[str, int] = {}
        self.idx_to_tag: dict[int, str] = {}

    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available."""
        try:
            import tensorflow as tf
            gpu_devices = tf.config.list_physical_devices('GPU')
            gpu_available = bool(gpu_devices)

            print_info(f"TensorFlow version: {tf.__version__}")
            print_info(f"GPU available: {gpu_available}")

            if gpu_available:
                for gpu in gpu_devices:
                    try:
                        gpu_name = tf.config.experimental.get_device_details(gpu).get('device_name', 'Unknown GPU')
                        print_info(f"  GPU: {gpu_name}")
                        logger.info(f"Using GPU: {gpu_name}")
                    except:
                        print_info(f"  GPU: {gpu.name}")
                        logger.info(f"Using GPU: {gpu.name}")
            else:
                print_warning("No GPU detected - training will use CPU (slower)")
                print_info("For GPU support, ensure CUDA toolkit is installed")
                logger.warning("Training on CPU - this will be significantly slower")

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

    def _load_parquet_dataset(self, parquet_path: Path) -> tuple[np.ndarray, np.ndarray, int, tuple[int, int]]:
        """
        Load dataset from Parquet file with embedded images.

        Returns:
            Tuple of (images_array, labels_array, num_classes, image_size)
        """
        print_info(f"Loading Parquet dataset: {parquet_path}")

        # Check metadata for image format
        meta_path = parquet_path.with_suffix('.meta.json')
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)

            if not metadata.get('images_embedded', False):
                print_error("Dataset does not contain embedded images!")
                print_error("Run 'pr0loader prepare' to create a dataset with embedded images.")
                raise ValueError("Dataset missing embedded images")

            image_format = metadata.get('image_format', {})
            image_size = tuple(metadata.get('image_size', [224, 224]))
            print_info(f"Image format: {image_format.get('dtype', 'unknown')}, "
                      f"size: {image_size}, preprocessing: {image_format.get('preprocessing', 'unknown')}")
        else:
            print_warning("No metadata file found - assuming default image format")
            image_size = (224, 224)

        # Read Parquet file
        table = pq.read_table(parquet_path)

        # Check for required columns
        columns = table.column_names
        if 'image_data' not in columns:
            print_error("Dataset does not contain 'image_data' column!")
            print_error("Run 'pr0loader prepare' to create a dataset with embedded images.")
            raise ValueError("Dataset missing image_data column")

        tags_list = table.column('tags').to_pylist()
        image_data_list = table.column('image_data').to_pylist()

        print_info(f"Loaded {len(tags_list):,} samples")

        # Build tag vocabulary from all tags in dataset
        all_tags = set()
        for tags in tags_list:
            all_tags.update(tags)

        unique_tags = sorted(all_tags)
        self.tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}
        num_classes = len(unique_tags)

        print_info(f"Number of unique tags (classes): {num_classes}")

        # Encode tags to multi-hot vectors
        labels = np.zeros((len(tags_list), num_classes), dtype=np.float32)
        for i, tags in enumerate(tags_list):
            for tag in tags:
                if tag in self.tag_to_idx:
                    labels[i, self.tag_to_idx[tag]] = 1.0

        # Reconstruct images from bytes
        # Images are stored as raw float32 bytes, shape (H, W, 3)
        print_info("Reconstructing images from embedded data...")
        h, w = image_size
        images = np.zeros((len(image_data_list), h, w, 3), dtype=np.float32)

        for i, img_bytes in enumerate(image_data_list):
            # Reconstruct from raw bytes
            arr = np.frombuffer(img_bytes, dtype=np.float32).reshape(h, w, 3)
            images[i] = arr

        print_info(f"Images array shape: {images.shape}, dtype: {images.dtype}")
        print_info(f"Memory usage: {images.nbytes / 1024**3:.2f} GB")

        return images, labels, num_classes, image_size

    def run(self, dataset_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Run the training pipeline.

        Args:
            dataset_path: Path to training dataset (Parquet with embedded images)
            output_path: Optional path for model output

        Returns:
            Path to saved model, or None if training failed
        """
        print_header(
            "🧠 Train Model",
            "Training tag prediction model with ResNet50"
        )

        if not self._check_tensorflow():
            return None

        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers

        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)

        # Load dataset - only Parquet with embedded images is supported
        if dataset_path.suffix != '.parquet':
            print_error("Only Parquet datasets with embedded images are supported.")
            print_error("Run 'pr0loader prepare' to create a dataset.")
            return None

        try:
            images, labels, num_classes, image_size = self._load_parquet_dataset(dataset_path)
        except Exception as e:
            print_error(f"Failed to load dataset: {e}")
            return None

        # Apply dev mode limit
        if self.settings.dev_mode:
            limit = min(self.settings.dev_limit, len(images))
            images = images[:limit]
            labels = labels[:limit]
            print_warning(f"Dev mode: using {len(images)} samples")

        # Create TensorFlow dataset directly from numpy arrays
        # No image preprocessing needed - it's already done in prepare!
        print_info("Creating TensorFlow dataset...")
        dataset = tf.data.Dataset.from_tensor_slices((images, labels))
        dataset = dataset.shuffle(buffer_size=min(1000, len(images)), seed=42)
        dataset = dataset.batch(self.settings.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        # Build model
        print_info("Building model...")
        base_model = keras.applications.ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=(*image_size, 3)
        )
        base_model.trainable = False  # Freeze base model

        inputs = keras.Input(shape=(*image_size, 3))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='sigmoid')(x)
        model = keras.Model(inputs, outputs)

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.settings.learning_rate),
            loss='binary_crossentropy',
            metrics=[keras.metrics.BinaryAccuracy(threshold=0.5)]
        )

        print_info("Model architecture:")
        if not is_headless():
            model.summary(print_fn=lambda x: print_info(x))

        # Setup callbacks
        self.settings.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.settings.checkpoint_dir / 'ckpt_{epoch}.weights.h5'),
                save_weights_only=True,
                save_freq='epoch'
            ),
            keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=2,
                restore_best_weights=True
            ),
        ]

        # Train
        print_info(f"Starting training for {self.settings.num_epochs} epochs...")
        print_info(f"Batch size: {self.settings.batch_size}")
        print_info(f"Learning rate: {self.settings.learning_rate}")
        print_info(f"Image size: {image_size}")

        try:
            verbosity = 2 if logger.isEnabledFor(logging.DEBUG) else (1 if not is_headless() else 2)

            history = model.fit(
                dataset,
                epochs=self.settings.num_epochs,
                callbacks=callbacks,
                verbose=verbosity
            )
        except Exception as e:
            print_error(f"Training failed: {e}")
            return None

        # Save model and tag mapping
        if output_path is None:
            output_path = self.settings.model_path

        output_path.parent.mkdir(parents=True, exist_ok=True)

        print_info(f"Saving model to {output_path}")
        model.save(str(output_path))

        # Save tag mapping
        mapping_path = output_path.with_suffix('.tags.json')
        with open(mapping_path, 'w') as f:
            json.dump({
                'tag_to_idx': self.tag_to_idx,
                'idx_to_tag': {str(k): v for k, v in self.idx_to_tag.items()},
                'num_classes': num_classes,
                'image_size': list(image_size),
            }, f, indent=2)

        print_info(f"Saved tag mapping to {mapping_path}")

        # Print training stats
        print_stats_table("Training Results", {
            "Final loss": f"{history.history['loss'][-1]:.4f}",
            "Final accuracy": f"{history.history['binary_accuracy'][-1]:.4f}",
            "Epochs completed": len(history.history['loss']),
            "Model saved": str(output_path),
        })

        print_success("Training complete!")

        return output_path

