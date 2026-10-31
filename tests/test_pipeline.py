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


def ramp(img):
    """Fake segmenter: subject on the left, an 8 px soft edge, background on the right."""
    alpha = np.zeros((img.height, img.width), np.float32)
    alpha[:, :16] = 1
    alpha[:, 16:24] = np.linspace(1, 0, 8)
    return alpha


def red_on_green():
    # A red subject photographed on green: the soft edge mixes the two colours.
    a = ramp(Image.new("RGB", (40, 30)))[..., None]
    return Image.fromarray((a * (200, 0, 0) + (1 - a) * (0, 255, 0)).round().astype(np.uint8))


def edge_green(**kwargs):
    # Mean green left in the soft edge after moving the subject to blue (the right answer has none).
    return np.asarray(change_background(red_on_green(), (0, 0, 255), ramp, **kwargs).image)[:, 16:24, 1].mean()


def test_default_edges_remove_the_old_background():
    assert edge_green(edges="soft") > 10  # the old green leaks into the edge
    assert edge_green() < 1  # the default (fg) removes it


def test_hard_edges_return_a_binary_mask():
    result = change_background(red_on_green(), (0, 0, 255), ramp, "hard")
    assert set(np.unique(result.alpha)) <= {0.0, 1.0}
