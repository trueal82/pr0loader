"""Predict pipeline - predict tags for images."""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from pr0loader.config import Settings
from pr0loader.models import PredictionResult
from pr0loader.utils.console import (
    print_header,
    print_success,
    print_info,
    print_error,
    console,
)

logger = logging.getLogger(__name__)


class PredictPipeline:
    """Pipeline stage for predicting tags on images."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = None
        self.tag_to_idx: dict[str, int] = {}
        self.idx_to_tag: dict[int, str] = {}

    def _check_tensorflow(self) -> bool:
        """Check if TensorFlow is available."""
        try:
            import tensorflow as tf
            return True
        except ImportError:
            print_error("TensorFlow not installed. Install with: pip install pr0loader[ml]")
            return False

    def load_model(self, model_path: Optional[Path] = None) -> bool:
        """
        Load a trained model.

        Args:
            model_path: Path to model file. Defaults to settings.model_path.

        Returns:
            True if model loaded successfully.
        """
        if not self._check_tensorflow():
            return False

        from tensorflow import keras

        if model_path is None:
            model_path = self.settings.model_path

        if not model_path.exists():
            print_error(f"Model not found: {model_path}")
            return False

        # Load model
        print_info(f"Loading model from {model_path}")
        try:
            self.model = keras.models.load_model(str(model_path))
        except Exception as e:
            print_error(f"Failed to load model: {e}")
            return False

        # Load tag mapping
        mapping_path = model_path.with_suffix('.tags.json')
        if not mapping_path.exists():
            print_error(f"Tag mapping not found: {mapping_path}")
            return False

        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            self.tag_to_idx = mapping['tag_to_idx']
            self.idx_to_tag = {int(k): v for k, v in mapping['idx_to_tag'].items()}

        print_info(f"Loaded {len(self.idx_to_tag)} tags")
        print_success("Model loaded successfully!")

        return True

    def predict(self, image_path: Path, top_k: int = 5) -> PredictionResult:
        """
        Predict tags for a single image.

        Args:
            image_path: Path to the image file.
            top_k: Number of top tags to return.

        Returns:
            PredictionResult with predicted tags and confidences.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        import tensorflow as tf
        from tensorflow import keras

        # Load and preprocess image
        image_size = self.settings.image_size

        image = tf.io.read_file(str(image_path))
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, image_size)
        image = keras.applications.resnet50.preprocess_input(image)
        image = tf.expand_dims(image, 0)  # Add batch dimension

        # Predict
        predictions = self.model.predict(image, verbose=0)[0]

        # Get top-k indices
        top_indices = np.argsort(predictions)[-top_k:][::-1]

        # Build result
        tags = [
            (self.idx_to_tag[idx], float(predictions[idx]))
            for idx in top_indices
            if idx in self.idx_to_tag
        ]

        return PredictionResult(
            image_path=str(image_path),
            tags=tags,
        )

    def predict_batch(self, image_paths: list[Path], top_k: int = 5) -> list[PredictionResult]:
        """
        Predict tags for multiple images.

        Args:
            image_paths: List of paths to image files.
            top_k: Number of top tags to return per image.

        Returns:
            List of PredictionResults.
        """
        results = []
        for path in image_paths:
            try:
                result = self.predict(path, top_k=top_k)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to predict for {path}: {e}")
                results.append(PredictionResult(image_path=str(path), tags=[]))

        return results

    def run(
        self,
        image_paths: list[Path],
        model_path: Optional[Path] = None,
        top_k: int = 5
    ) -> list[PredictionResult]:
        """
        Run prediction pipeline.

        Args:
            image_paths: List of image paths to predict.
            model_path: Optional model path override.
            top_k: Number of top tags to return.

        Returns:
            List of PredictionResults.
        """
        print_header(
            "🔮 Predict Tags",
            f"Predicting top {top_k} tags for {len(image_paths)} image(s)"
        )

        if not self.load_model(model_path):
            return []

        results = []

        for path in image_paths:
            print_info(f"Processing: {path}")

            try:
                result = self.predict(Path(path), top_k=top_k)
                results.append(result)

                # Display results
                console.print(f"\n[bold]Top {top_k} predicted tags:[/bold]")
                for i, (tag, confidence) in enumerate(result.top_5_tags, 1):
                    bar_width = int(confidence * 30)
                    bar = "█" * bar_width + "░" * (30 - bar_width)
                    console.print(
                        f"  {i}. [cyan]{tag:20}[/cyan] [{bar}] {confidence:.1%}"
                    )

            except Exception as e:
                print_error(f"Failed to predict: {e}")
                results.append(PredictionResult(image_path=str(path), tags=[]))

        print_success("Prediction complete!")

        return results

