"""Build the E2 edge-test set from the E1 subset (see data/e2_subset.md).

Copies the chosen E1 files to data/e2/<group>/{im,gt}/ (run make_e1_subset.py
first), makes the synthetic inputs in data/e2/A_synthetic/im/, and writes
everything to data/e2_subset.csv, so the notebook only reads that CSV.

    python data/make_e2_subset.py
"""
import csv
import shutil
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
OUT = DATA / "e2"
sys.path.insert(0, str(DATA.parent))  # bgchange/ is at the repo root; this script runs from data/

from bgchange.composite import over_colour  # noqa: E402
from bgchange.io import load_alpha, load_rgb, parse_hex, save_rgb  # noqa: E402

FIELDS = ["group", "source", "name", "image", "gt", "gt_type", "fg", "background", "note"]

# A_synthetic: each portrait is the foreground F, with its ground-truth matte, over a plain
# colour B: input = alpha * F + (1 - alpha) * B. The right answer on any new colour C is
# then known exactly: alpha * F + (1 - alpha) * C.
BACKGROUNDS = {
    "#00FF00": "saturated green: worst case for colour spill",
    "#F0F0F0": "near-white studio: the common real case",
}

# A portraits shown as zoomed crops, picked from two measurements (not by eye): the share of
# soft pixels (0.02 < alpha < 0.98) in the ground-truth matte, and BiRefNet's MAE in E1.
ZOOM = {
    "14996438642_58c976f957_o": "zoom: most hair (5.2 % of pixels soft)",
    "34504878700_d626a8a3c6_o": "zoom: second most hair (2.8 % soft), arms raised",
    "24179781765_21532649fd_o": "zoom: highest BiRefNet MAE in E1 (0.0067), long hair",
    "49020513826_d8fffe604d_o": "zoom: same MAE (0.0067), seated",
    "35998070871_36c398b9c6_o": "zoom, counter-example: worst BiRefNet IoU in E1 (0.942), almost no hair (0.3 % soft)",
}

# D: real fashion photos with soft edges, no ground truth (judged by eye). D06 is the control.
REAL = {
    "D01_vintage_shoot": "backlit strands on a busy outdoor background; fuzzy knit",
    "D05_jump_white_studio": "flyaway hair on white; motion blur on the feet",
    "D08_runway_sheer_veil": "soft alpha on the garment itself: sheer fabric",
    "D09_white_on_white": "light curly hair and a white top on near-white",
    "D10_low_light": "blond curls on a dark brown background",
    "D06_logo_tshirt": "control: hard edges and a logo, all edge modes should give the same image",
}

GLASS = "1#Accessories#3#Eyeglasses#4710367344_110c332442_o"


def copy_row(r, group, note):
    """Copy an E1 row's image (and ground truth) to data/e2/<group>/ and return the E2 row."""
    row = {**r, "group": group, "fg": "", "background": "", "note": note}
    for kind in ("image", "gt"):
        if r[kind]:
            src = DATA / r[kind]
            dst = OUT / group / ("im" if kind == "image" else "gt") / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            row[kind] = dst.relative_to(DATA).as_posix()
    return row


def synthetic_row(person, colour, note):
    """Composite a portrait over a plain colour, save it as PNG and return its row.

    PNG, not JPEG: compression would change the edge pixels, and the right answer
    would no longer be exact. The same over_colour as the pipeline makes the input.
    """
    name = f"{person['name']}_bg{colour[1:]}"
    dst = OUT / "A_synthetic" / "im" / f"{name}.png"
    img = over_colour(load_rgb(DATA / person["image"]), load_alpha(DATA / person["gt"]), parse_hex(colour))
    save_rgb(img, dst)
    return {**person, "group": "A_synthetic", "name": name, "image": dst.relative_to(DATA).as_posix(),
            "fg": person["image"], "background": colour, "note": note}


def main():
    e1 = DATA / "e1_subset.csv"
    if not e1.exists():
        raise SystemExit("data/e1_subset.csv not found: run python data/make_e1_subset.py first")
    with open(e1, encoding="utf-8") as f:
        e1_rows = list(csv.DictReader(f))
    missing = [r["image"] for r in e1_rows if not (DATA / r["image"]).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images missing from data/e1/ (first: {missing[0]}): "
                         "run python data/make_e1_subset.py first")
    if OUT.exists():
        shutil.rmtree(OUT)  # rebuilt from scratch, so no file is left from an older list

    # A: the 15 E1 portraits (random, SEED = 0), soft mattes: alpha metrics.
    people = [copy_row(r, "A_people", ZOOM.get(r["name"], "")) for r in e1_rows if r["group"] == "A_people"]
    rows = list(people)
    # A_synthetic: the same portraits on known backgrounds: colour metrics against the exact answer.
    # Their gt and fg point at the A_people copies (the true matte and the true foreground).
    for i, person in enumerate(people, 1):
        for colour, note in BACKGROUNDS.items():
            rows.append(synthetic_row(person, colour, note))
        print(f"synthetic {i}/{len(people)}", end="\r")
    # D: chosen real photos, in the order of REAL.
    real = {r["name"]: r for r in e1_rows if r["group"] == "D_real"}
    rows += [copy_row(real[name], "D_real", note) for name, note in REAL.items()]
    # Glass: the DIS ground truth marks the lenses opaque, so this one is judged by eye only.
    glass = next(r for r in e1_rows if r["name"] == GLASS)
    rows.append(copy_row(glass, "B_glass", "glass: ground truth marks the lenses opaque, judged by eye"))

    with open(DATA / "e2_subset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} images -> {OUT} and e2_subset.csv")


if __name__ == "__main__":
    main()
