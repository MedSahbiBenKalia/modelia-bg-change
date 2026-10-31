"""Tests for bgchange/matting.py (the first run compiles pymatting's numba code: a few seconds)."""
import numpy as np
import pytest
from PIL import Image

from bgchange.composite import over_colour
from bgchange.matting import EDGE_MODES, estimate_foreground, hard_alpha, prepare


def ramp_alpha(h=48, w=64):
    """Subject on the left (alpha 1), a 10 px soft edge, background on the right (alpha 0)."""
    a = np.zeros((h, w), np.float32)
    a[:, :20] = 1
    a[:, 20:30] = np.linspace(1, 0, 10)
    return a


def test_hard_alpha_is_binary_at_half():
    a = np.array([0.0, 0.49, 0.5, 0.51, 1.0], np.float32)
    assert hard_alpha(a).tolist() == [0, 0, 1, 1, 1]
    assert hard_alpha(a).dtype == np.float32


def test_foreground_removes_a_known_leak():
    # A red subject over a green background: at the edge the photo mixes the two.
    alpha = ramp_alpha()
    a = alpha[..., None]
    red, green, blue = np.array([200, 0, 0]), np.array([0, 255, 0]), (0, 0, 255)
    photo = Image.fromarray((a * red + (1 - a) * green).round().astype(np.uint8))
    truth = np.asarray(over_colour(Image.new("RGB", photo.size, (200, 0, 0)), alpha, blue), np.float32)
    edge = (alpha > 0) & (alpha < 1)
    soft = np.asarray(over_colour(*prepare(photo, alpha, "soft"), blue), np.float32)
    fg = np.asarray(over_colour(*prepare(photo, alpha, "fg"), blue), np.float32)
    assert np.abs(soft - truth)[edge].mean() > 10  # the green leaks into the soft edge
    assert np.abs(fg - truth)[edge].max() <= 1  # the estimated foreground is red again (rounding only)


def test_foreground_keeps_the_opaque_interior():
    # Random texture, the hardest case for a smoothness term: still the photo, 2 px or more from the edge.
    rng = np.random.default_rng(0)
    photo = Image.fromarray(rng.integers(0, 256, (48, 64, 3), dtype=np.uint8))
    fg = np.asarray(estimate_foreground(photo, ramp_alpha()))
    assert fg.shape == (48, 64, 3) and fg.dtype == np.uint8
    assert (fg[:, :18] == np.asarray(photo)[:, :18]).all()  # alpha 1 up to column 19, so columns 0..17


def test_unknown_mode_is_rejected():
    assert EDGE_MODES == ("hard", "soft", "fg")
    with pytest.raises(ValueError, match="unknown edge mode"):
        prepare(Image.new("RGB", (4, 4)), np.ones((4, 4), np.float32), "blur")
