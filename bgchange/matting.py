"""What to composite at the edge of the subject: a hard mask, the soft alpha, or the soft alpha with
an estimate of the subject's own colour (compared in E2, experiments/e2_edges.ipynb).

At an edge pixel the photo is already a mix of the subject and the old background:
I = a*F + (1 - a)*B. Compositing the photo itself over a new colour C (the "soft" mode) gives
a*I + (1 - a)*C = [a*F + (1 - a)*C] + a*(1 - a)*(B - F): the right answer plus a leak of the old
background, largest at a = 0.5 (hair, soft edges). The "fg" mode composites an estimate of F instead of I.
"""
import numpy as np
from PIL import Image

EDGE_MODES = ("hard", "soft", "fg")


def hard_alpha(alpha: np.ndarray, thr: float = 0.5) -> np.ndarray:
    """Binary mask as float32: 1 where alpha >= thr (the threshold of the IoU metrics), else 0."""
    return (alpha >= thr).astype(np.float32)


def estimate_foreground(img: Image.Image, alpha: np.ndarray) -> Image.Image:
    """The subject's own colour F at every pixel, from the photo and its alpha (RGB, same size).

    pymatting's estimate_foreground_ml with its default settings (Germer et al., Fast Multi-Level
    Foreground Estimation, ICPR 2020). Where alpha is 1, F is the photo, except right next to where
    alpha changes (its smoothness term acts there). Where alpha is 0, F is extrapolated from nearby
    subject pixels; the composite gives it no weight. Runs on CPU (numba): the first call ever
    compiles the code, later runs load it from numba's disk cache.
    """
    from pymatting import estimate_foreground_ml  # imports numba (seconds): only when this mode is used
    fg = estimate_foreground_ml(np.asarray(img, np.float32) / 255, alpha)
    return Image.fromarray((np.clip(fg, 0, 1) * 255).round().astype(np.uint8))


def prepare(img: Image.Image, alpha: np.ndarray, mode: str) -> tuple[Image.Image, np.ndarray]:
    """The (image, alpha) pair to composite over the new colour, for one edge mode.

    hard: the photo with a binary mask (no soft edge at all)
    soft: the photo with the soft alpha (the old background leaks into soft edges)
    fg:   the estimated foreground with the soft alpha
    """
    if mode == "hard":
        return img, hard_alpha(alpha)
    if mode == "soft":
        return img, alpha
    if mode == "fg":
        return estimate_foreground(img, alpha), alpha
    raise ValueError(f"unknown edge mode {mode!r}: expected one of {EDGE_MODES}")
