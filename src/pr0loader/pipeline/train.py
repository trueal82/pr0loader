"""Train pipeline - train tag prediction model."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

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
    """Pipeline stage for training the tag prediction model."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.tag_to_idx: dict[str, int] = {}
        self.idx_to_tag: dict[int, str] = {}

    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available."""
        try:
            import tensorflow as tf
            gpu_available = bool(tf.config.list_physical_devices('GPU'))
            print_info(f"TensorFlow version: {tf.__version__}")
            print_info(f"GPU available: {gpu_available}")
            return True
        except ImportError:
            print_error("TensorFlow not installed. Install with: pip install pr0loader[ml]")
            return False

    def run(self, csv_path: Path, output_path: Optional[Path] = None) -> Optional[Path]:
        """
        Run the training pipeline.

        Args:
            csv_path: Path to training CSV file
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

        # Load dataset
        print_info(f"Loading dataset: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print_error(f"Failed to load CSV: {e}")
            return None

        df = df[df['image'].notnull()]

        # Apply dev mode limit
        if self.settings.dev_mode:
            df = df.head(self.settings.dev_limit)
            print_warning(f"Dev mode: using {len(df)} samples")

        print_info(f"Loaded {len(df)} samples")

        # Build tag vocabulary
        tag_columns = [f'tag{i}' for i in range(1, 6)]
        all_tags = df[tag_columns].values.flatten()
        unique_tags = sorted(set(tag for tag in all_tags if pd.notnull(tag)))

        self.tag_to_idx = {tag: idx for idx, tag in enumerate(unique_tags)}
        self.idx_to_tag = {idx: tag for tag, idx in self.tag_to_idx.items()}
        num_classes = len(unique_tags)

        print_info(f"Number of unique tags (classes): {num_classes}")

        # Encode tags to multi-hot vectors
        def encode_tags(row):
            tags = [row[f'tag{i}'] for i in range(1, 6) if pd.notnull(row.get(f'tag{i}'))]
            indices = [self.tag_to_idx[tag] for tag in tags if tag in self.tag_to_idx]
            multi_hot = np.zeros(num_classes, dtype=np.float32)
            multi_hot[indices] = 1.0
            return multi_hot

        print_info("Encoding tags...")
        df['labels'] = df.apply(encode_tags, axis=1)

        # Prepare image paths
        image_paths = df['image'].apply(
            lambda x: str(self.settings.filesystem_prefix / x)
        ).values
        labels = np.stack(df['labels'].values)

        # Create TensorFlow dataset
        print_info("Creating TensorFlow dataset...")
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

        image_size = self.settings.image_size

        def load_and_preprocess(path, label):
            try:
                image = tf.io.read_file(path)
                image = tf.image.decode_jpeg(image, channels=3)
                image = tf.image.resize(image, image_size)
                image = keras.applications.resnet50.preprocess_input(image)
                return image, label
            except Exception:
                # Return a black image if loading fails
                return tf.zeros((*image_size, 3)), label

        dataset = dataset.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
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
                filepath=str(self.settings.checkpoint_dir / 'ckpt_{epoch}'),
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

        try:
            history = model.fit(
                dataset,
                epochs=self.settings.num_epochs,
                callbacks=callbacks,
                verbose=1 if not is_headless() else 2
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

