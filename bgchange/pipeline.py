"""End-to-end pipeline shared by the CLI, the notebooks and the tests.

    image -> segment -> alpha -> composite over the new colour -> output

"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

from bgchange.composite import over_colour

# RGB image -> soft alpha, float32 in [0, 1], at the image's size. segment.Segmenter is one.
SegmentFn = Callable[[Image.Image], np.ndarray]


@dataclass
class Result:
    image: Image.Image  # the subject on the new colour: RGB, same size as the input
    alpha: np.ndarray  # the alpha used for it (for --save-mask and the experiments)


def change_background(img: Image.Image, rgb: tuple[int, int, int], segment: SegmentFn) -> Result:
    """Put the subject of `img` (RGB) on the solid colour `rgb`."""
    alpha = segment(img)
    return Result(image=over_colour(img, alpha, rgb), alpha=alpha)
