"""Tests for bgchange/segment.py.

The fast tests use a small stand-in model, so they check our pre- and post-processing without
downloading BiRefNet. The last test runs the real model; it is skipped unless BGCHANGE_SLOW=1.
"""
import os

import numpy as np
import pytest
import torch
from PIL import Image, ImageDraw

from bgchange.segment import INPUT_SIZE, Segmenter


class FakeBiRefNet(torch.nn.Module):
    """Returns side outputs like BiRefNet; the last map is high where the input is bright."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(20.0))  # a weight, so .to(device, dtype) has something to move
        self.seen = None

    def forward(self, x):
        self.seen = x
        return [torch.zeros_like(x[:, :1]), self.scale * x.mean(1, keepdim=True)]


def segment_with_fake(img):
    fake = FakeBiRefNet()
    return Segmenter(device="cpu", model=fake)(img), fake


def test_alpha_has_image_size_range_and_dtype():
    alpha, _ = segment_with_fake(Image.new("RGB", (300, 200), "white"))  # not square: a swapped width/height would show
    assert alpha.shape == (200, 300)
    assert alpha.dtype == np.float32
    assert alpha.min() >= 0 and alpha.max() <= 1


def test_model_sees_a_normalised_square():
    _, fake = segment_with_fake(Image.new("RGB", (300, 200), "white"))
    assert fake.seen.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)
    assert fake.seen.dtype == torch.float32  # CPU runs in fp32
    expected = (1 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])  # white, ImageNet stats
    assert torch.allclose(fake.seen[0, :, 0, 0], expected, atol=1e-4)


def test_alpha_follows_the_subject():
    img = Image.new("RGB", (400, 100), "black")
    img.paste((255, 255, 255), (0, 0, 200, 100))  # bright left half = subject for the fake model
    alpha, _ = segment_with_fake(img)
    assert alpha[:, :190].min() > 0.99
    assert alpha[:, 210:].max() < 0.01


@pytest.mark.parametrize("mode", ["RGBA", "L"])
def test_non_rgb_input(mode):
    alpha, fake = segment_with_fake(Image.new(mode, (64, 48)))
    assert alpha.shape == (48, 64)
    assert fake.seen.shape[1] == 3


@pytest.mark.skipif(os.environ.get("BGCHANGE_SLOW") != "1",
                    reason="runs the real BiRefNet (444 MB download, ~30 s on CPU); set BGCHANGE_SLOW=1")
def test_birefnet_finds_a_simple_object():
    img = Image.new("RGB", (640, 480), (235, 235, 235))
    ImageDraw.Draw(img).ellipse((220, 140, 420, 340), fill=(180, 30, 40))  # red disc on light grey
    alpha = Segmenter()(img)
    assert alpha[220:260, 300:340].mean() > 0.9  # centre of the disc
    assert alpha[:60, :60].mean() < 0.1  # background corner
    assert abs((alpha > 0.5).mean() - np.pi * 100 ** 2 / (640 * 480)) < 0.01  # mask area close to the disc's
