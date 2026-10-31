"""Image loading/saving (EXIF rotation, colour profile, RGBA/grayscale to RGB) and hex colour parsing."""
import io
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageCms, ImageOps

SRGB = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB"))


def load_rgb(path: str | Path) -> Image.Image:
    """Open an image, apply its EXIF rotation, and convert it to sRGB.

    Hex colours are sRGB and output is saved profile-less, so embedded profiles get
    converted, not relabelled
    """
    img = ImageOps.exif_transpose(Image.open(path))
    icc = img.info.get("icc_profile")
    if icc:
        try:
            img = ImageCms.profileToProfile(img, ImageCms.ImageCmsProfile(io.BytesIO(icc)), SRGB, outputMode="RGB")
        except (OSError, ImageCms.PyCMSError):
            pass  # broken profile
    return img.convert("RGB")


def load_alpha(path: str | Path) -> np.ndarray:
    """Read a mask / matte / saved alpha as float32 in [0, 1]."""
    return np.asarray(Image.open(path).convert("L"), np.float32) / 255


def save_rgb(img: Image.Image, path: str | Path) -> None:
    """Save an image, creating the folder; JPEG at quality 95 (Pillow's default is 75; PNG ignores it)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=95)


def save_alpha(alpha: np.ndarray, path: str | Path) -> None:
    """Save a [0, 1] alpha as an 8-bit grayscale PNG (rounding error <= 0.5/255)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(alpha, 0, 1) * 255).round().astype(np.uint8)).save(path)


def parse_hex(s: str) -> tuple[int, int, int]:
    """'#1E90FF', '1e90ff', '#fff' -> (r, g, b)."""
    h = s.strip().removeprefix("#")
    # a regex, because int(x, 16) alone also accepts '-1' and '+f': '#-1ffff' gave (-1, 255, 255)
    if not re.fullmatch(r"[0-9a-fA-F]{3}|[0-9a-fA-F]{6}", h):
        raise ValueError(f"bad hex colour {s!r}: expected 3 or 6 hex digits, e.g. '#1E90FF' or '#fff'")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)  # 'fff' -> 'ffffff'
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
