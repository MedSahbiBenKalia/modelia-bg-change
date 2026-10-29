"""Evaluation metrics: IoU, MAE, boundary IoU, SAD, gradient, connectivity, shadow error.

All functions take a predicted alpha and a ground truth as float arrays in [0, 1]
with the same shape. Soft ground truth (PPM-100 mattes) is binarised at 0.5 for
the IoU-type metrics; MAE uses the soft values directly.
"""
import cv2
import numpy as np


def _binary(a: np.ndarray, thr: float = 0.5) -> np.ndarray:
    return a >= thr


def iou(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> float:
    """Intersection over union of the masks thresholded at `thr`."""
    p, g = _binary(pred, thr), _binary(gt, thr)
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, g).sum() / union)


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean absolute error on the soft alpha."""
    return float(np.abs(pred.astype(np.float32) - gt.astype(np.float32)).mean())


def boundary(mask: np.ndarray, dilation_ratio: float = 0.02) -> np.ndarray:
    """Inner boundary band of a binary mask: mask minus its erosion.

    Width d = dilation_ratio * image diagonal, as in Cheng et al., Boundary IoU (CVPR 2021).
    The reference code erodes d times with a 3x3 kernel; one erosion with a
    (2d+1)x(2d+1) square kernel gives the same result and is faster.
    """
    h, w = mask.shape
    d = max(1, int(round(dilation_ratio * np.hypot(h, w))))
    m = mask.astype(np.uint8)
    padded = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)  # image border counts as boundary
    eroded = cv2.erode(padded, np.ones((2 * d + 1, 2 * d + 1), np.uint8))[1:h + 1, 1:w + 1]
    return (m - eroded).astype(bool)


def boundary_iou(pred: np.ndarray, gt: np.ndarray, thr: float = 0.5, dilation_ratio: float = 0.02) -> float:
    """IoU computed only on the boundary bands of the two masks (sensitive to edge quality)."""
    p = boundary(_binary(pred, thr), dilation_ratio)
    g = boundary(_binary(gt, thr), dilation_ratio)
    union = np.logical_or(p, g).sum()
    return 1.0 if union == 0 else float(np.logical_and(p, g).sum() / union)
