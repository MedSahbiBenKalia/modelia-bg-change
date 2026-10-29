"""Tests for bgchange/metrics.py."""
import numpy as np

from bgchange.metrics import boundary, boundary_iou, iou, mae


def square(size=100, lo=20, hi=80):
    m = np.zeros((size, size), np.float32)
    m[lo:hi, lo:hi] = 1
    return m


def test_perfect_prediction():
    gt = square()
    assert iou(gt, gt) == 1.0
    assert mae(gt, gt) == 0.0
    assert boundary_iou(gt, gt) == 1.0


def test_empty_prediction():
    gt = square()
    empty = np.zeros_like(gt)
    assert iou(empty, gt) == 0.0
    assert mae(empty, gt) == gt.mean()
    assert boundary_iou(empty, gt) == 0.0


def test_iou_known_value():
    # 60x60 gt vs the same square shifted by 30 px: overlap 30x60, union 90x60
    gt, pred = square(), np.roll(square(), 30, axis=1)
    assert np.isclose(iou(pred, gt), 1800 / 5400)


def test_soft_alpha_threshold_and_mae():
    gt = square()
    pred = gt * 0.6  # every foreground pixel at 0.6: still >= 0.5
    assert iou(pred, gt) == 1.0
    assert np.isclose(mae(pred, gt), 0.4 * gt.mean())


def test_boundary_is_a_band_inside_the_mask():
    m = square().astype(bool)
    b = boundary(m, dilation_ratio=0.02)  # d = round(0.02 * 141.4) = 3 px
    assert not (b & ~m).any()  # band stays inside the mask
    assert b[20, 50] and b[22, 50] and not b[23, 50]  # 3 px wide
    assert not b[50, 50]  # interior excluded


def test_boundary_iou_more_sensitive_than_iou():
    gt = square()
    pred = square(lo=21, hi=79)  # 1 px smaller on every side
    assert boundary_iou(pred, gt) < iou(pred, gt)
