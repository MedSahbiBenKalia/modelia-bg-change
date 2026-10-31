"""Tests for bgchange/metrics.py."""
import numpy as np

from bgchange.metrics import boundary, boundary_iou, edge_band, gradient_error, iou, mae, masked_mae, soft_pixels


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


def test_edge_band_covers_both_sides_of_the_outline():
    b = edge_band(square(), dilation_ratio=0.02)  # d = 3 px; the square covers rows and columns 20..79
    assert b[17, 50] and not b[16, 50]  # 3 px outside
    assert b[22, 50] and not b[23, 50]  # 3 px inside
    assert not b[50, 50] and not b[5, 50]  # interior and far background excluded


def test_edge_band_ignores_the_image_border():
    m = np.zeros((100, 100), np.float32)
    m[50:] = 1  # a body cut by the bottom of the frame
    b = edge_band(m, dilation_ratio=0.02)
    assert b[50, 50] and b[47, 50] and b[50, 0]  # the real outline (row 50), up to the side of the frame
    assert not b[99, 50] and not b[80, 0]  # the bottom and side borders inside the body are not edges


def test_soft_pixels_ignore_near_0_and_near_1():
    a = np.array([0.0, 0.01, 0.03, 0.5, 0.97, 0.99, 1.0], np.float32)
    assert soft_pixels(a).tolist() == [False, False, True, True, True, False, False]


def test_masked_mae_on_rgb_and_empty_mask():
    a = np.zeros((4, 4, 3), np.uint8)
    b = a.copy()
    b[0, 0] = 30  # one pixel off by 30 on every channel
    mask = np.zeros((4, 4), bool)
    mask[0, :2] = True
    assert masked_mae(a, b, mask) == 15.0  # mean over the two masked pixels
    assert np.isnan(masked_mae(a, b, np.zeros((4, 4), bool)))


def test_gradient_error_zero_when_equal_and_higher_for_a_hard_edge():
    gt = np.tile(np.linspace(0, 1, 40, dtype=np.float32), (20, 1))  # a soft ramp
    mask = np.ones_like(gt, bool)
    assert gradient_error(gt, gt, mask) == 0.0
    assert gradient_error((gt >= 0.5).astype(np.float32), gt, mask) > 0
