"""Web API for tag prediction inference."""

import io
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Pydantic models for API responses
class TagPrediction(BaseModel):
    """A single tag prediction."""
    tag: str
    confidence: float


class PredictionResponse(BaseModel):
    """Response for a single image prediction."""
    filename: str
    tags: list[TagPrediction]


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    num_tags: Optional[int] = None


class APIServer:
    """FastAPI server for tag prediction."""

    def __init__(self, model_path: Optional[Path] = None):
        self.app = FastAPI(
            title="pr0loader Tag Prediction API",
            description="Predict the most likely tags for images",
            version="2.0.0",
        )
        self.model_path = model_path
        self.predictor = None
        self._setup_routes()
        self._setup_cors()

    def _setup_cors(self):
        """Setup CORS middleware for external access."""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure as needed for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes."""

        @self.app.on_event("startup")
        async def startup():
            """Load model on startup."""
            await self._load_model()

        @self.app.get("/", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint."""
            return HealthResponse(
                status="healthy" if self.predictor and self.predictor.model else "no_model",
                model_loaded=self.predictor is not None and self.predictor.model is not None,
                model_path=str(self.model_path) if self.model_path else None,
                num_tags=len(self.predictor.idx_to_tag) if self.predictor else None,
            )

        @self.app.get("/health", response_model=HealthResponse)
        async def health():
            """Alias for health check."""
            return await health_check()

        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict_single(
            file: UploadFile = File(..., description="Image file to predict tags for"),
            top_k: int = 5,
        ):
            """
            Predict tags for a single image.

            - **file**: Image file (jpg, jpeg, png, gif)
            - **top_k**: Number of top tags to return (default: 5)
            """
            if not self.predictor or not self.predictor.model:
                raise HTTPException(status_code=503, detail="Model not loaded")

            # Validate file type
            if not file.content_type or not file.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="File must be an image")

            try:
                # Read file content
                content = await file.read()

                # Predict
                tags = await self._predict_from_bytes(content, top_k)

                return PredictionResponse(
                    filename=file.filename or "unknown",
                    tags=[TagPrediction(tag=t, confidence=c) for t, c in tags],
                )
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        @self.app.post("/predict/batch", response_model=BatchPredictionResponse)
        async def predict_batch(
            files: list[UploadFile] = File(..., description="Image files to predict tags for"),
            top_k: int = 5,
        ):
            """
            Predict tags for multiple images.

            - **files**: List of image files
            - **top_k**: Number of top tags to return per image (default: 5)
            """
            if not self.predictor or not self.predictor.model:
                raise HTTPException(status_code=503, detail="Model not loaded")

            predictions = []
            for file in files:
                try:
                    if not file.content_type or not file.content_type.startswith("image/"):
                        predictions.append(PredictionResponse(
                            filename=file.filename or "unknown",
                            tags=[],
                        ))
                        continue

                    content = await file.read()
                    tags = await self._predict_from_bytes(content, top_k)

                    predictions.append(PredictionResponse(
                        filename=file.filename or "unknown",
                        tags=[TagPrediction(tag=t, confidence=c) for t, c in tags],
                    ))
                except Exception as e:
                    logger.error(f"Prediction failed for {file.filename}: {e}")
                    predictions.append(PredictionResponse(
                        filename=file.filename or "unknown",
                        tags=[],
                    ))

            return BatchPredictionResponse(predictions=predictions)

    async def _load_model(self):
        """Load the prediction model."""
        from pr0loader.config import load_settings
        from pr0loader.pipeline import PredictPipeline

        settings = load_settings()
        self.predictor = PredictPipeline(settings)

        model_path = self.model_path or settings.model_path

        if model_path and model_path.exists():
            logger.info(f"Loading model from {model_path}")
            success = self.predictor.load_model(model_path)
            if success:
                logger.info(f"Model loaded successfully with {len(self.predictor.idx_to_tag)} tags")
                self.model_path = model_path
            else:
                logger.warning("Failed to load model")
        else:
            logger.warning(f"Model not found at {model_path}")

    async def _predict_from_bytes(self, image_bytes: bytes, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Predict tags from image bytes.

        Returns:
            List of (tag, confidence) tuples.
        """
        import tempfile
        import numpy as np

        # TensorFlow imports
        import tensorflow as tf
        from tensorflow import keras

        # Write to temp file (TensorFlow needs a file path)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = Path(tmp.name)

        try:
            # Load and preprocess image
            image_size = self.predictor.settings.image_size

            image = tf.io.read_file(str(tmp_path))
            image = tf.image.decode_image(image, channels=3, expand_animations=False)
            image = tf.image.resize(image, image_size)
            image = keras.applications.resnet50.preprocess_input(image)
            image = tf.expand_dims(image, 0)

            # Predict
            predictions = self.predictor.model.predict(image, verbose=0)[0]

            # Get top-k indices
            top_indices = np.argsort(predictions)[-top_k:][::-1]

            # Build result
            tags = [
                (self.predictor.idx_to_tag[idx], float(predictions[idx]))
                for idx in top_indices
                if idx in self.predictor.idx_to_tag
            ]

            return tags
        finally:
            # Clean up temp file
            tmp_path.unlink(missing_ok=True)


def create_app(model_path: Optional[Path] = None) -> FastAPI:
    """Create and return the FastAPI application."""
    server = APIServer(model_path)
    return server.app

