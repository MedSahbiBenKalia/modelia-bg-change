"""Tests for bgchange/io.py."""
import numpy as np
import pytest
from PIL import Image, ImageCms

from bgchange.io import load_alpha, load_rgb, parse_hex, save_alpha, save_rgb


@pytest.mark.parametrize("s", ["#1E90FF", "1e90ff", " #1E90FF "])
def test_parse_hex_6_digits(s):
    assert parse_hex(s) == (30, 144, 255)


def test_parse_hex_3_digits():
    assert parse_hex("#fff") == (255, 255, 255)
    assert parse_hex("0a5") == (0, 170, 85)


@pytest.mark.parametrize("s", ["", "#12", "#12345", "#GGGGGG", "#-1ffff", "+fffff", "##fff", "#1E90FF00"])
def test_parse_hex_rejects_bad_input(s):
    with pytest.raises(ValueError):
        parse_hex(s)


def test_load_rgb_converts_rgba_and_gray(tmp_path):
    for mode in ("RGBA", "L"):
        Image.new(mode, (4, 3)).save(tmp_path / f"{mode}.png")
        assert load_rgb(tmp_path / f"{mode}.png").mode == "RGB"


def test_load_rgb_srgb_profile_keeps_pixels(tmp_path):
    srgb = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
    Image.new("RGB", (4, 3), (200, 100, 50)).save(tmp_path / "srgb.png", icc_profile=srgb)
    px = np.asarray(load_rgb(tmp_path / "srgb.png"), np.int16)
    assert np.abs(px - (200, 100, 50)).max() <= 1  # sRGB -> sRGB is the identity, up to rounding


def test_load_rgb_ignores_a_broken_profile(tmp_path):
    Image.new("RGB", (4, 3), (200, 100, 50)).save(tmp_path / "bad.png", icc_profile=b"not a profile")
    assert (np.asarray(load_rgb(tmp_path / "bad.png")) == (200, 100, 50)).all()


@pytest.mark.parametrize("name", ["out.png", "out.jpg"])
def test_save_rgb_creates_the_folder(tmp_path, name):
    save_rgb(Image.new("RGB", (4, 3), (200, 100, 50)), tmp_path / "new" / name)
    assert load_rgb(tmp_path / "new" / name).size == (4, 3)


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
