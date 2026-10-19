"""Load a trained model and classify an individual image."""

from pathlib import Path

import numpy as np
import tensorflow as tf

from covid_xray.data import preprocess_image


def load_model(model_path: str | Path) -> tf.keras.Model:
    """Load the saved model for inference."""
    model_path = Path(model_path)

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = tf.keras.models.load_model(
        model_path,
        compile=False,
    )

    if model.input_shape != (None, 224, 224, 1):
        raise ValueError(f"Unexpected model input: {model.input_shape}")

    if model.output_shape != (None, 1):
        raise ValueError(f"Unexpected model output: {model.output_shape}")

    return model


def predict_image(
    model: tf.keras.Model,
    image_path: str | Path,
    threshold: float = 0.5,
) -> dict:
    """Return the positive-class score and thresholded label."""
    if not np.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1.")

    image = preprocess_image(image_path)
    batch = np.expand_dims(image, axis=0)

    output = np.asarray(model(batch, training=False))

    if output.shape != (1, 1):
        raise ValueError(f"Unexpected prediction shape: {output.shape}")

    score = float(output[0, 0])

    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Model returned an invalid score.")

    # Match the threshold convention used in the evaluation notebook.
    predicted_class = int(score > threshold)

    return {
        "positive_score": score,
        "predicted_class": predicted_class,
        "predicted_label": (
            "positive" if predicted_class == 1 else "negative"
        ),
        "threshold": float(threshold),
    }