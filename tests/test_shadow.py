"""Tests for bgchange/shadow.py and the shade of composite.over_colour."""
import numpy as np
import pytest
from PIL import Image

from bgchange.composite import over_colour
from bgchange.shadow import LIT_METHODS, shadow_map, with_strength


def scene(h=600, w=800, grey=200, shadow=None, noise=2.0):
    """A dark square subject on a flat grey background, optionally with a shadow T = 0.5 to its right."""
    alpha = np.zeros((h, w), np.float32)
    alpha[200:400, 200:350] = 1
    t = np.ones((h, w), np.float32)
    if shadow:
        t[220:400, 350:430] = 0.5
    a = alpha[..., None]
    x = a * 40 + (1 - a) * t[..., None] * grey + np.random.default_rng(0).normal(0, noise, (h, w, 3))
    return Image.fromarray(np.clip(x, 0, 255).round().astype(np.uint8)), alpha, t


@pytest.mark.parametrize("method", LIT_METHODS)
def test_plain_background_has_no_shadow(method):
    img, alpha, _ = scene()
    t = shadow_map(img, alpha, method)
    assert t.shape == alpha.shape and t.dtype == np.float32
    assert t.min() == 1  # the noise is inside the dead zone, and the subject's own darkness is not a shadow


@pytest.mark.parametrize("method", LIT_METHODS)
def test_known_shadow_is_recovered(method):
    img, alpha, truth = scene(shadow=True)
    t = shadow_map(img, alpha, method)
    assert 0 <= t.min() and t.max() <= 1
    core = np.zeros_like(alpha, bool)
    core[240:380, 365:415] = True  # inside the shadow, away from its edges
    assert np.abs(t[core] - 0.5).mean() < 0.03
    far = np.zeros_like(alpha, bool)
    far[:, 600:] = True  # lit background
    assert (t[far] == 1).all()


def test_strength_zero_is_no_shadow():
    img, alpha, _ = scene(shadow=True)
    t = shadow_map(img, alpha)
    assert (with_strength(t, 0) == 1).all()
    assert np.allclose(with_strength(t, 1), t)
    plain = np.asarray(over_colour(img, alpha, (30, 144, 255)))
    assert (np.asarray(over_colour(img, alpha, (30, 144, 255), with_strength(t, 0))) == plain).all()


def test_shade_darkens_the_background_only():
    img = Image.new("RGB", (4, 2), (10, 20, 30))
    alpha = np.array([[1, 1, 0, 0]] * 2, np.float32)
    out = np.asarray(over_colour(img, alpha, (200, 100, 50), np.full((2, 4), 0.5, np.float32)))
    assert (out[:, :2] == (10, 20, 30)).all()
    assert (out[:, 2:] == (100, 50, 25)).all()


def test_unknown_method_is_rejected():
    img, alpha, _ = scene(h=40, w=40)
    with pytest.raises(ValueError, match="unknown lit-background method"):
        shadow_map(img, alpha, "lama")
