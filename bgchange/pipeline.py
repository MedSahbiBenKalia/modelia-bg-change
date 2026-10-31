"""End-to-end pipeline shared by the CLI, the notebooks and the tests.

    image -> segment -> alpha -> edge mode -> composite over the new colour -> output

"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

from bgchange.composite import over_colour
from bgchange.matting import prepare

# RGB image -> soft alpha, float32 in [0, 1], at the image's size. segment.Segmenter is one.
SegmentFn = Callable[[Image.Image], np.ndarray]

DEFAULT_EDGES = "fg"  # chosen in E2: soft alpha + estimated foreground


@dataclass
class Result:
    image: Image.Image  # the subject on the new colour: RGB, same size as the input
    alpha: np.ndarray  # the alpha used for it (for --save-mask and the experiments)


def change_background(img: Image.Image, rgb: tuple[int, int, int], segment: SegmentFn,
                      edges: str = DEFAULT_EDGES) -> Result:
    """Put the subject of `img` (RGB) on the solid colour `rgb`; `edges`: one of matting.EDGE_MODES."""
    fg, alpha = prepare(img, segment(img), edges)
    return Result(image=over_colour(fg, alpha, rgb), alpha=alpha)
