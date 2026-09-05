from pathlib import Path

from PIL import Image


def read_image_size(path):
    """Read image dimensions without decoding the full pixel buffer."""
    image_path = Path(path)
    with Image.open(image_path) as image:
        width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Image has invalid dimensions: {image_path}")
    return int(width), int(height)
