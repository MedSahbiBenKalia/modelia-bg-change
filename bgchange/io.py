"""Image loading/saving (EXIF rotation, RGBA/grayscale to RGB) and hex colour parsing."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_rgb(path: str | Path) -> Image.Image:
    """Open an image, apply its EXIF rotation, and convert it to RGB."""
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def load_alpha(path: str | Path) -> np.ndarray:
    """Read a mask / matte / saved alpha as float32 in [0, 1]."""
    return np.asarray(Image.open(path).convert("L"), np.float32) / 255


def save_alpha(alpha: np.ndarray, path: str | Path) -> None:
    """Save a [0, 1] alpha as an 8-bit grayscale PNG (rounding error <= 1/510)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(alpha, 0, 1) * 255).round().astype(np.uint8)).save(path)


def parse_hex(s: str) -> tuple[int, int, int]:
    """'#1E90FF', '1e90ff', '#fff' -> (r, g, b)."""
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)  # 'fff' -> 'ffffff'
    if len(s) != 6:
        raise ValueError(f"bad hex colour: {s!r}")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
