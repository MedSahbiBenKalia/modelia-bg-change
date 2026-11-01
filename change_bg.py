"""CLI: put the subject of an image (person or product) on a solid background colour.

    python change_bg.py input.jpg --color "#1E90FF" -o out.png --save-mask
    python change_bg.py input.jpg --color "#1E90FF" --keep-shadows --shadow-strength 0.8
"""
import argparse
import time
from pathlib import Path

from bgchange.io import load_rgb, parse_hex, save_alpha, save_rgb
from bgchange.matting import EDGE_MODES
from bgchange.pipeline import DEFAULT_EDGES, DEFAULT_SHADOW_METHOD, change_background
from bgchange.shadow import LIT_METHODS


def hex_colour(s: str) -> tuple[int, int, int]:
    """argparse type: parse_hex, raising the error type whose message argparse prints as is."""
    try:
        return parse_hex(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def strength(s: str) -> float:
    """argparse type: a number from 0 to 1."""
    try:
        v = float(s)
    except ValueError:
        v = -1.0
    if not 0 <= v <= 1:
        raise argparse.ArgumentTypeError(f"bad shadow strength {s!r}: expected a number from 0 to 1")
    return v


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Put the subject of an image on a solid background colour.")
    p.add_argument("input", type=Path, help="image of a person or a product")
    p.add_argument("--color", type=hex_colour, required=True, help='background colour: "#1E90FF", "1e90ff" or "#fff"')
    p.add_argument("-o", "--output", type=Path, help="output image (default: <input name>_bg.png next to the input)")
    p.add_argument("--save-mask", action="store_true", help="also save the alpha as <output name>_mask.png")
    p.add_argument("--edges", choices=EDGE_MODES, default=DEFAULT_EDGES,
                   help="fg: soft alpha + foreground colour estimation (default, see E2); soft: soft alpha only; hard: mask at 0.5")
    p.add_argument("--keep-shadows", action="store_true",
                   help="keep the subject's shadows on the new colour (off by default: on a textured background, "
                        "the texture is kept too, see E3)")
    p.add_argument("--shadow-strength", type=strength,
                   help="with --keep-shadows: 0 = no shadow, 1 = as measured (default 1)")
    p.add_argument("--shadow-method", choices=LIT_METHODS,
                   help="with --keep-shadows: inpaint (default, see E3) ignores light falloff on the backdrop but cuts "
                        "shadows longer than 10 %% of the image diagonal; flat keeps long shadows on a plain backdrop "
                        "but takes light falloff for shadow")
    p.add_argument("--device", help="cpu or cuda (default: cuda if available)")
    args = p.parse_args(argv)
    if not args.input.is_file():
        p.error(f"input not found: {args.input}")  # now, not after loading the model
    if not args.keep_shadows and (args.shadow_strength is not None or args.shadow_method):
        p.error("--shadow-strength and --shadow-method need --keep-shadows")
    args.output = args.output or args.input.with_name(f"{args.input.stem}_bg.png")
    args.shadow_strength = (1.0 if args.shadow_strength is None else args.shadow_strength) if args.keep_shadows else 0.0
    args.shadow_method = args.shadow_method or DEFAULT_SHADOW_METHOD
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from bgchange.segment import Segmenter  # imports torch (seconds): after parsing, so --help and typos answer at once

    img = load_rgb(args.input)
    t0 = time.perf_counter()
    segmenter = Segmenter(args.device)
    t1 = time.perf_counter()
    result = change_background(img, args.color, segmenter, args.edges, args.shadow_strength, args.shadow_method)
    t2 = time.perf_counter()

    save_rgb(result.image, args.output)
    print(f"saved {args.output}")
    if args.save_mask:
        mask_path = args.output.with_name(f"{args.output.stem}_mask.png")
        save_alpha(result.alpha, mask_path)
        print(f"saved {mask_path}")
    shadows = f", shadows kept ({args.shadow_method}, strength {args.shadow_strength:g})" if args.keep_shadows else ""
    print(f"{img.width}x{img.height} on {segmenter.device}{shadows}: model load {t1 - t0:.1f} s, processing {t2 - t1:.1f} s")


if __name__ == "__main__":
    main()
