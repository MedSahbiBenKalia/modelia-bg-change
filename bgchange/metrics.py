"""Evaluation metrics: IoU, MAE, Boundary IoU (E1); soft pixels, edge band, masked MAE, gradient error (E2).

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


def soft_pixels(alpha: np.ndarray, eps: float = 0.02) -> np.ndarray:
    """Pixels that are neither clearly subject nor clearly background: eps < alpha < 1 - eps.

    In portraits these are mostly hair (E2). eps = 0.02 (5/255) stays above the JPEG noise of the PPM-100 mattes.
    """
    return (alpha > eps) & (alpha < 1 - eps)


def edge_band(gt: np.ndarray, dilation_ratio: float = 0.02, thr: float = 0.5) -> np.ndarray:
    """Band on both sides of the outline of `gt` thresholded at `thr`: its dilation minus its erosion.

    Same width as `boundary` (d = dilation_ratio * image diagonal), but on both sides, so it also covers
    the faint outer hair (alpha < 0.5) that the inner band misses. No padding: the image border is not
    an edge here, so a body cut by the frame adds no band.
    """
    h, w = gt.shape
    d = max(1, int(round(dilation_ratio * np.hypot(h, w))))
    m = _binary(gt, thr).astype(np.uint8)
    kernel = np.ones((2 * d + 1, 2 * d + 1), np.uint8)
    return (cv2.dilate(m, kernel) - cv2.erode(m, kernel)).astype(bool)


def masked_mae(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """Mean absolute difference over the pixels of `mask`; a and b are alphas (H, W) or RGB images (H, W, 3)."""
    if not mask.any():
        return float("nan")
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a[mask].astype(np.float32) - b[mask].astype(np.float32)).mean())


def gradient_error(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray, sigma: float = 1.4) -> float:
    """Gradient error (Rhemann et al., CVPR 2009) over the pixels of `mask`.

    Squared difference of the gradient magnitudes of the two alphas (Gaussian derivatives, sigma 1.4),
    averaged over the mask instead of summed, so large and small images weigh the same. High when an
    edge is jagged or sharper than the ground truth.
    """
    from scipy.ndimage import gaussian_gradient_magnitude  # scipy comes with pymatting
    if not mask.any():
        return float("nan")
    gp = gaussian_gradient_magnitude(pred.astype(np.float32), sigma)
    gg = gaussian_gradient_magnitude(gt.astype(np.float32), sigma)
    return float(((gp - gg) ** 2)[mask].mean())
