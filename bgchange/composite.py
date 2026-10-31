"""Alpha compositing of the foreground over the new background colour."""
import numpy as np
from PIL import Image


def over_colour(img: Image.Image, alpha: np.ndarray, rgb: tuple[int, int, int],
                shade: np.ndarray | None = None) -> Image.Image:
    """Porter-Duff 'over': out = alpha * image + (1 - alpha) * shade * colour.

    shade: the shadow map T of bgchange/shadow.py (1 = lit), or None for a plain colour.
    """
    a = alpha[..., None]
    background = np.array(rgb, np.float32) if shade is None else shade[..., None] * np.array(rgb, np.float32)
    out = a * np.asarray(img, np.float32) + (1 - a) * background
    return Image.fromarray(out.round().astype(np.uint8))
