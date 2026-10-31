"""Tests for bgchange/pipeline.py, with a fake segmenter (no model download)."""
import numpy as np
from PIL import Image

from bgchange.pipeline import change_background


def left_half(img):
    """Fake segmenter: the left half of the image is the subject."""
    alpha = np.zeros((img.height, img.width), np.float32)
    alpha[:, : img.width // 2] = 1
    return alpha


def test_change_background_keeps_size_subject_and_alpha():
    img = Image.new("RGB", (40, 30), (200, 100, 50))  # not square: a swapped width/height would show
    result = change_background(img, (0, 0, 255), left_half)
    out = np.asarray(result.image)
    assert result.image.size == img.size
    assert (out[:, :20] == (200, 100, 50)).all()
    assert (out[:, 20:] == (0, 0, 255)).all()
    assert (result.alpha == left_half(img)).all()  # the alpha that was used is returned
