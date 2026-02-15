"""Gradio web interface for tag prediction testing."""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def create_gradio_interface(model_path: Optional[Path] = None, api_url: Optional[str] = None):
    """
    Create a Gradio interface for tag prediction.

    Args:
        model_path: Path to the model (for local inference)
        api_url: URL of the API server (for remote inference)

    Returns:
        Gradio Blocks interface
    """
    import gradio as gr

    # Determine inference mode
    use_api = api_url is not None
    predictor = None

    if not use_api:
        # Load model locally
        from pr0loader.config import load_settings
        from pr0loader.pipeline import PredictPipeline

        settings = load_settings()
        predictor = PredictPipeline(settings)

        model_path = model_path or settings.model_path
        if model_path and model_path.exists():
            predictor.load_model(model_path)
        else:
            logger.warning(f"Model not found at {model_path}")

    def predict_tags(image, top_k: int = 5) -> str:
        """Predict tags for an uploaded image."""
        if image is None:
            return "Please upload an image"

        try:
            if use_api:
                # Use API for inference
                import requests

                # Save image to bytes
                import io
                from PIL import Image

                img = Image.fromarray(image)
                buf = io.BytesIO()
                img.save(buf, format='JPEG')
                buf.seek(0)

                response = requests.post(
                    f"{api_url}/predict",
                    files={"file": ("image.jpg", buf, "image/jpeg")},
                    params={"top_k": top_k},
                )

                if response.status_code == 200:
                    data = response.json()
                    tags = data.get("tags", [])
                    return format_predictions(tags)
                else:
                    return f"API Error: {response.status_code} - {response.text}"

            else:
                # Local inference
                if not predictor or not predictor.model:
                    return "Model not loaded. Please ensure a trained model exists."

                import tempfile
                import numpy as np
                import tensorflow as tf
                from tensorflow import keras
                from PIL import Image

                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    img = Image.fromarray(image)
                    img.save(tmp.name, format='JPEG')
                    tmp_path = Path(tmp.name)

                try:
                    # Load and preprocess
                    image_size = predictor.settings.image_size

                    img_tensor = tf.io.read_file(str(tmp_path))
                    img_tensor = tf.image.decode_image(img_tensor, channels=3, expand_animations=False)
                    img_tensor = tf.image.resize(img_tensor, image_size)
                    img_tensor = keras.applications.resnet50.preprocess_input(img_tensor)
                    img_tensor = tf.expand_dims(img_tensor, 0)

                    # Predict
                    predictions = predictor.model.predict(img_tensor, verbose=0)[0]

                    # Get top-k
                    top_indices = np.argsort(predictions)[-top_k:][::-1]

                    tags = [
                        {"tag": predictor.idx_to_tag[idx], "confidence": float(predictions[idx])}
                        for idx in top_indices
                        if idx in predictor.idx_to_tag
                    ]

                    return format_predictions(tags)
                finally:
                    tmp_path.unlink(missing_ok=True)

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return f"Error: {str(e)}"

    def predict_batch(files, top_k: int = 5) -> str:
        """Predict tags for multiple images."""
        if not files:
            return "Please upload at least one image"

        results = []
        for file in files:
            try:
                if use_api:
                    import requests

                    with open(file.name, "rb") as f:
                        response = requests.post(
                            f"{api_url}/predict",
                            files={"file": (Path(file.name).name, f, "image/jpeg")},
                            params={"top_k": top_k},
                        )

                    if response.status_code == 200:
                        data = response.json()
                        tags = data.get("tags", [])
                        results.append(f"**{Path(file.name).name}**\n{format_predictions(tags)}")
                    else:
                        results.append(f"**{Path(file.name).name}**: Error {response.status_code}")

                else:
                    # Local inference
                    from PIL import Image
                    import numpy as np

                    img = Image.open(file.name)
                    img_array = np.array(img)

                    pred_result = predict_tags(img_array, top_k)
                    results.append(f"**{Path(file.name).name}**\n{pred_result}")

            except Exception as e:
                results.append(f"**{Path(file.name).name}**: Error - {str(e)}")

        return "\n\n---\n\n".join(results)

    def format_predictions(tags: list) -> str:
        """Format predictions as a nice string."""
        if not tags:
            return "No predictions available"

        lines = []
        for i, tag_info in enumerate(tags, 1):
            tag = tag_info.get("tag", tag_info.get("tag", ""))
            conf = tag_info.get("confidence", 0)
            bar_width = int(conf * 20)
            bar = "█" * bar_width + "░" * (20 - bar_width)
            lines.append(f"{i}. **{tag}** [{bar}] {conf:.1%}")

        return "\n".join(lines)

    # Create Gradio interface
    with gr.Blocks(
        title="pr0loader Tag Prediction",
        theme=gr.themes.Soft(),
    ) as interface:
        gr.Markdown("""
        # 🔮 pr0loader Tag Prediction
        
        Upload an image to predict the most likely tags.
        """)

        with gr.Tab("Single Image"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Upload Image",
                        type="numpy",
                    )
                    top_k_single = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of tags to predict",
                    )
                    predict_btn = gr.Button("🔮 Predict Tags", variant="primary")

                with gr.Column():
                    output_single = gr.Markdown(label="Predictions")

            predict_btn.click(
                fn=predict_tags,
                inputs=[image_input, top_k_single],
                outputs=output_single,
            )

        with gr.Tab("Batch Upload"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Images",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    top_k_batch = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of tags per image",
                    )
                    batch_btn = gr.Button("🔮 Predict All", variant="primary")

                with gr.Column():
                    output_batch = gr.Markdown(label="Predictions")

            batch_btn.click(
                fn=predict_batch,
                inputs=[file_input, top_k_batch],
                outputs=output_batch,
            )

        with gr.Tab("API Info"):
            mode = "Remote API" if use_api else "Local Model"
            endpoint = api_url if use_api else "N/A (local inference)"

            gr.Markdown(f"""
            ## API Information
            
            **Mode:** {mode}
            
            **API Endpoint:** `{endpoint}`
            
            ### API Usage
            
            #### Single Image Prediction
            ```bash
            curl -X POST "{api_url or 'http://localhost:8000'}/predict" \\
                -F "file=@image.jpg" \\
                -F "top_k=5"
            ```
            
            #### Batch Prediction
            ```bash
            curl -X POST "{api_url or 'http://localhost:8000'}/predict/batch" \\
                -F "files=@image1.jpg" \\
                -F "files=@image2.jpg" \\
                -F "top_k=5"
            ```
            
            #### Response Format
            ```json
            {{
                "filename": "image.jpg",
                "tags": [
                    {{"tag": "example", "confidence": 0.95}},
                    {{"tag": "test", "confidence": 0.87}}
                ]
            }}
            ```
            """)

    return interface


def launch_gradio(
    model_path: Optional[Path] = None,
    api_url: Optional[str] = None,
    host: str = "0.0.0.0",
    port: int = 7860,
    share: bool = False,
):
    """
    Launch the Gradio interface.

    Args:
        model_path: Path to the model for local inference
        api_url: URL of API server for remote inference
        host: Host to bind to
        port: Port to listen on
        share: Create a public Gradio link
    """
    interface = create_gradio_interface(model_path, api_url)
    interface.launch(
        server_name=host,
        server_port=port,
        share=share,
    )

