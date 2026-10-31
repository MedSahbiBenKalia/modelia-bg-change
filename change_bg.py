"""CLI: put the subject of an image (person or product) on a solid background colour.

    python change_bg.py input.jpg --color "#1E90FF" -o out.png --save-mask
"""
import argparse
import time
from pathlib import Path

from bgchange.io import load_rgb, parse_hex, save_alpha, save_rgb
from bgchange.matting import EDGE_MODES
from bgchange.pipeline import DEFAULT_EDGES, change_background


def hex_colour(s: str) -> tuple[int, int, int]:
    """argparse type: parse_hex, raising the error type whose message argparse prints as is."""
    try:
        return parse_hex(s)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Put the subject of an image on a solid background colour.")
    p.add_argument("input", type=Path, help="image of a person or a product")
    p.add_argument("--color", type=hex_colour, required=True, help='background colour: "#1E90FF", "1e90ff" or "#fff"')
    p.add_argument("-o", "--output", type=Path, help="output image (default: <input name>_bg.png next to the input)")
    p.add_argument("--save-mask", action="store_true", help="also save the alpha as <output name>_mask.png")
    p.add_argument("--edges", choices=EDGE_MODES, default=DEFAULT_EDGES,
                   help="fg: soft alpha + foreground colour estimation (default, see E2); soft: soft alpha only; hard: mask at 0.5")
    p.add_argument("--device", help="cpu or cuda (default: cuda if available)")
    args = p.parse_args(argv)
    if not args.input.is_file():
        p.error(f"input not found: {args.input}")  # now, not after loading the model
    args.output = args.output or args.input.with_name(f"{args.input.stem}_bg.png")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    from bgchange.segment import Segmenter  # imports torch (seconds): after parsing, so --help and typos answer at once

    img = load_rgb(args.input)
    t0 = time.perf_counter()
    segmenter = Segmenter(args.device)
    t1 = time.perf_counter()
    result = change_background(img, args.color, segmenter, args.edges)
    t2 = time.perf_counter()

    save_rgb(result.image, args.output)
    print(f"saved {args.output}")
    if args.save_mask:
        mask_path = args.output.with_name(f"{args.output.stem}_mask.png")
        save_alpha(result.alpha, mask_path)
        print(f"saved {mask_path}")
    print(f"{img.width}x{img.height} on {segmenter.device}: model load {t1 - t0:.1f} s, processing {t2 - t1:.1f} s")


if __name__ == "__main__":
    main()
