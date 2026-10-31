"""Shadow map, so the subject's shadows can be kept on the new colour (compared in E3, experiments/e3_shadows.ipynb).

Chuang et al. (Shadow Matting and Compositing, SIGGRAPH 2003): a background pixel is I = beta*L + (1 - beta)*S,
with L the lit background and S the same background in full shadow. If S is the lit background dimmed
(S = k*L), then I / L = beta + (1 - beta)*k = T, and on a new colour N the right pixel is T*N. So only the
lit background L has to be estimated, and T is the ratio of the photo to it.
"""
import cv2
import numpy as np
from PIL import Image

LIT_METHODS = ("flat", "inpaint")
WORK_SIZE = 512  # long side the map is computed at: shadows are smooth, and inpainting is slow at full size
LUMA = np.array([0.299, 0.587, 0.114], np.float32)  # Rec. 601 weights: a grey shadow, one T for R, G and B


def inpaint(x: np.ndarray, mask: np.ndarray, scale: float) -> np.ndarray:
    """Telea inpainting (Telea 2004) of a float map whose values times `scale` fit in 16 bits.

    Through uint16: on float32 input, OpenCV 5.0's Telea leaves about half of a large hole unfilled (stripes).
    """
    x16 = np.clip(x * scale, 0, 65535).round().astype(np.uint16)
    return cv2.inpaint(x16, mask.astype(np.uint8), 3, cv2.INPAINT_TELEA).astype(np.float32) / scale


def lit_luminance(y: np.ndarray, subject: np.ndarray, method: str, margin: float) -> np.ndarray:
    """Luminance of the background without the subject's shadow.

    flat:    one level for the whole image, the median of the background (shadows are a minority of it)
    inpaint: the photo, with the subject and a band of `margin` (fraction of the diagonal) around it
             filled in from the pixels around the band (Telea 2004): a shadow inside the band is removed
    """
    if method == "flat":
        return np.full_like(y, np.median(y[~subject]))
    if method == "inpaint":
        r = max(1, round(margin * np.hypot(*y.shape)))
        band = cv2.dilate(subject.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        return inpaint(y, band, 256)  # luminance 0-255 -> 0-65280
    raise ValueError(f"unknown lit-background method {method!r}: expected one of {LIT_METHODS}")


def shadow_map(img: Image.Image, alpha: np.ndarray, method: str = "inpaint", margin: float = 0.10,
               dead_zone: float = 0.03) -> np.ndarray:
    """T in [0, 1] at the image's size: the photo's luminance over its estimated lit background (1 = lit).

    Darkening below `dead_zone` is set to 0 (noise, JPEG, texture) and the rest is rescaled, so T stays
    continuous. Where the subject is, the photo shows the subject, so T is filled in from around it.
    """
    h, w = alpha.shape
    s = min(1.0, WORK_SIZE / max(h, w))
    size = (max(1, round(w * s)), max(1, round(h * s)))
    y = cv2.resize(np.asarray(img), size, interpolation=cv2.INTER_AREA).astype(np.float32) @ LUMA
    a = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
    # 1 px past the soft edge: pixels with alpha just above 0 still hold some of the subject
    subject = cv2.dilate((a > 0.02).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if subject.all():
        return np.ones((h, w), np.float32)  # no background to measure
    t = np.clip(y / np.maximum(lit_luminance(y, subject, method, margin), 1), 0, 1)
    t = inpaint(t, subject, 65535)
    t = 1 - np.maximum(0, (1 - t) - dead_zone) / (1 - dead_zone)
    return np.clip(cv2.resize(t, (w, h), interpolation=cv2.INTER_LINEAR), 0, 1)


def with_strength(t: np.ndarray, strength: float) -> np.ndarray:
    """0 = no shadow (T = 1 everywhere), 1 = the measured shadow, in between: a lighter shadow."""
    return 1 - strength * (1 - t)
