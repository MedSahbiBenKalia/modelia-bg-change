"""Tests for change_bg.py: arguments, output files and error handling, with a fake model."""
import numpy as np
import pytest
from PIL import Image

from bgchange.io import load_alpha, load_rgb
from change_bg import main


class FakeSegmenter:
    """Stands in for segment.Segmenter: the left half of the image is the subject."""
    device = "cpu"

    def __init__(self, device=None):
        pass

    def __call__(self, img):
        alpha = np.zeros((img.height, img.width), np.float32)
        alpha[:, : img.width // 2] = 1
        return alpha


class NoModel:
    def __init__(self, device=None):
        raise AssertionError("the model should not be loaded")


@pytest.fixture
def photo(tmp_path):
    path = tmp_path / "photo.png"
    Image.new("RGB", (40, 30), (200, 100, 50)).save(path)
    return path


def test_writes_output_and_mask(photo, monkeypatch):
    monkeypatch.setattr("bgchange.segment.Segmenter", FakeSegmenter)
    main([str(photo), "--color", "00f", "--save-mask"])
    out = np.asarray(load_rgb(photo.with_name("photo_bg.png")))  # default output name
    assert out.shape == (30, 40, 3)
    assert (out[:, :20] == (200, 100, 50)).all()
    assert (out[:, 20:] == (0, 0, 255)).all()
    mask = load_alpha(photo.with_name("photo_bg_mask.png"))
    assert (mask[:, :20] == 1).all() and (mask[:, 20:] == 0).all()


def test_output_path(photo, tmp_path, monkeypatch):
    monkeypatch.setattr("bgchange.segment.Segmenter", FakeSegmenter)
    main([str(photo), "--color", "#1E90FF", "-o", str(tmp_path / "out" / "blue.jpg")])
    assert load_rgb(tmp_path / "out" / "blue.jpg").size == (40, 30)


def test_edges_option(photo, monkeypatch):
    monkeypatch.setattr("bgchange.segment.Segmenter", FakeSegmenter)
    main([str(photo), "--color", "00f", "--edges", "hard"])
    assert load_rgb(photo.with_name("photo_bg.png")).size == (40, 30)


@pytest.mark.parametrize("args", [["--color", "#12"], ["--color", "red"], [], ["--color", "#fff", "--edges", "blur"]])
def test_bad_arguments_stop_before_loading_the_model(photo, monkeypatch, args):
    monkeypatch.setattr("bgchange.segment.Segmenter", NoModel)
    with pytest.raises(SystemExit) as e:
        main([str(photo), *args])
    assert e.value.code == 2  # argparse usage error


def test_missing_input_stops_before_loading_the_model(tmp_path, monkeypatch):
    monkeypatch.setattr("bgchange.segment.Segmenter", NoModel)
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path / "nope.jpg"), "--color", "#fff"])
    assert e.value.code == 2
