"""End-to-end pipeline shared by the CLI, the notebooks and the tests.

    image -> segment -> alpha -> edge mode -> (shadow map) -> composite over the new colour -> output

"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image

from bgchange.composite import over_colour
from bgchange.matting import prepare
from bgchange.shadow import shadow_map, with_strength

# RGB image -> soft alpha, float32 in [0, 1], at the image's size. segment.Segmenter is one.
SegmentFn = Callable[[Image.Image], np.ndarray]

DEFAULT_EDGES = "fg"  # chosen in E2: soft alpha + estimated foreground
DEFAULT_SHADOW_METHOD = "inpaint"  # chosen in E3 (B2): no false shadow where the backdrop's light falls off


@dataclass
class Result:
    image: Image.Image  # the subject on the new colour: RGB, same size as the input
    alpha: np.ndarray  # the alpha used for it (for --save-mask and the experiments)


def change_background(img: Image.Image, rgb: tuple[int, int, int], segment: SegmentFn,
                      edges: str = DEFAULT_EDGES, shadow_strength: float = 0.0,
                      shadow_method: str = DEFAULT_SHADOW_METHOD) -> Result:
    """Put the subject of `img` (RGB) on the solid colour `rgb`; `edges`: one of matting.EDGE_MODES.

    shadow_strength > 0 keeps the subject's shadows on the new colour (1 = as measured), with the lit-background
    estimate `shadow_method` (one of shadow.LIT_METHODS); 0 gives a plain colour.
    """
    soft = segment(img)
    fg, alpha = prepare(img, soft, edges)
    # the map is measured with the soft alpha, as in E3, whatever the edge mode
    shade = with_strength(shadow_map(img, soft, shadow_method), shadow_strength) if shadow_strength > 0 else None
    return Result(image=over_colour(fg, alpha, rgb, shade), alpha=alpha)
