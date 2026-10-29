"""Tests for bgchange/composite.py."""
import numpy as np
from PIL import Image

from bgchange.composite import over_colour


def test_over_colour_extremes_and_size():
    img = Image.new("RGB", (6, 4), (200, 100, 50))
    alpha = np.zeros((4, 6), np.float32)
    alpha[:, :3] = 1  # left half subject, right half background
    out = np.asarray(over_colour(img, alpha, (0, 0, 255)))
    assert out.shape == (4, 6, 3)
    assert (out[:, :3] == (200, 100, 50)).all()
    assert (out[:, 3:] == (0, 0, 255)).all()


def test_over_colour_half_alpha_blends():
    img = Image.new("RGB", (2, 2), (200, 200, 200))
    out = np.asarray(over_colour(img, np.full((2, 2), 0.5, np.float32), (0, 0, 0)))
    assert (out == 100).all()
