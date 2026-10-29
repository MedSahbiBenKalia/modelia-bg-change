"""Alpha compositing of the foreground over the new background colour."""
import numpy as np
from PIL import Image


def over_colour(img: Image.Image, alpha: np.ndarray, rgb: tuple[int, int, int]) -> Image.Image:
    """Porter-Duff 'over': out = alpha * image + (1 - alpha) * colour."""
    a = alpha[..., None]
    out = a * np.asarray(img, np.float32) + (1 - a) * np.array(rgb, np.float32)
    return Image.fromarray(out.round().astype(np.uint8))
