"""Tests for bgchange/io.py."""
import numpy as np
import pytest
from PIL import Image

from bgchange.io import load_alpha, load_rgb, parse_hex, save_alpha


@pytest.mark.parametrize("s", ["#1E90FF", "1e90ff", " #1E90FF "])
def test_parse_hex_6_digits(s):
    assert parse_hex(s) == (30, 144, 255)


def test_parse_hex_3_digits():
    assert parse_hex("#fff") == (255, 255, 255)
    assert parse_hex("0a5") == (0, 170, 85)


@pytest.mark.parametrize("s", ["", "#12", "#12345", "#GGGGGG"])
def test_parse_hex_rejects_bad_input(s):
    with pytest.raises(ValueError):
        parse_hex(s)


def test_load_rgb_converts_rgba_and_gray(tmp_path):
    for mode in ("RGBA", "L"):
        Image.new(mode, (4, 3)).save(tmp_path / f"{mode}.png")
        assert load_rgb(tmp_path / f"{mode}.png").mode == "RGB"


def test_load_rgb_applies_exif_rotation(tmp_path):
    img = Image.new("RGB", (40, 20))
    exif = img.getexif()
    exif[274] = 6  # orientation: rotate 90 degrees
    img.save(tmp_path / "rot.jpg", exif=exif)
    assert load_rgb(tmp_path / "rot.jpg").size == (20, 40)


def test_alpha_roundtrip(tmp_path):
    alpha = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
    save_alpha(alpha, tmp_path / "a.png")
    back = load_alpha(tmp_path / "a.png")
    assert back.min() >= 0 and back.max() <= 1
    assert np.abs(back - alpha).max() <= 1 / 510 + 1e-6
