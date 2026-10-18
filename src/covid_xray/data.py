"""Shared image preprocessing for training and inference."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from covid_xray.config import IMAGE_SIZE


def preprocess_image(
    path: str | Path,
    image_size: tuple[int, int] = IMAGE_SIZE,
) -> np.ndarray:
    """Return a padded grayscale image normalized to [0, 1].

    image_size uses Pillow's (width, height) convention.
    Output shape is (height, width, 1).
    """
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)

        # Composite transparent images onto a black background.
        if image.mode in ("RGBA", "LA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            background = Image.new(
                "RGBA",
                rgba.size,
                (0, 0, 0, 255),
            )
            image = Image.alpha_composite(background, rgba)

        image = image.convert("L")

        # Preserve aspect ratio and pad to the target dimensions.
        image = ImageOps.pad(
            image,
            image_size,
            method=Image.Resampling.BILINEAR,
            color=0,
        )

        pixels = np.asarray(image, dtype=np.float32) / 255.0

    return pixels[..., np.newaxis]