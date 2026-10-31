"""Build the E3 shadow test set (see data/e3_subset.md).

A_synthetic: the 15 E2 portraits on a known lit background, with a known shadow
B_desoba:    the 34 DESOBA test photos with one object and its shadow (downloads DESOBA once, 1.86 GB)
D_real:      fashion photos with shadows, no ground truth (6 from E1, 3 listed in data/e3_sources.csv)

A and B rows also give the lit background L and the true shadow factor T (1 = lit), so the right
output on a new colour N is known: alpha * F + (1 - alpha) * T * N. Writes data/e3/<group>/...
and data/e3_subset.csv, so the notebook only reads that CSV.

    python data/make_e3_subset.py
"""
import csv
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DATA = Path(__file__).resolve().parent
OUT = DATA / "e3"
sys.path.insert(0, str(DATA.parent))  # bgchange/ is at the repo root
sys.path.insert(0, str(DATA))  # make_e1_subset, also when run with python -I

from bgchange.io import load_rgb  # noqa: E402
from make_e1_subset import download  # noqa: E402

FIELDS = ["group", "source", "name", "image", "gt", "gt_type", "fg", "lit", "shadow", "other",
          "background", "shadow_kind", "note"]
SEED = 0

# A_synthetic: input = alpha * F + (1 - alpha) * T * L + noise, with F the portrait photo, alpha its matte,
# L a lit background and T the subject's own silhouette, moved and blurred.
SIZE = 1024  # long side: BiRefNet predicts at 1024 px anyway, and shadows are smooth
LITS = {  # name: (grey at the top, grey at the bottom, why)
    "white": (240, 240, "flat #F0F0F0: white studio"),
    "dark": (64, 64, "flat #404040: dark studio, the shadow is weak against the noise"),
    "gradient": (240, 150, "vertical 240 -> 150 grey: light falloff, which is not a shadow (like D07, D10)"),
}
SHADOWS = {  # name: (offset, blur sigma) in % of the image diagonal; None: no shadow, the control
    "none": None,
    "short_hard": (2, 0.3),
    "short_soft": (2, 2),
    "long_hard": (12, 0.3),  # longer than the 10 % margin of the inpainting method (B2)
    "long_soft": (12, 2),
}
DARKNESS = 0.5  # the shadow's core gets half the light; its blurred edge covers every T from 0.5 to 1
NOISE = 2.0  # grey levels (std) added to the whole input, like sensor noise

# B_desoba: DESOBA (Hong et al., AAAI 2022) = SOBA photos with every shadow removed by photo editors.
DESOBA_URL = ("https://www.dropbox.com/scl/fi/h9uoif9upyxzf5nf5tz8r/DESOBA_DATASET.zip"
              "?rlkey=6qpmtbdjwnexj28vlzdaagtvm&st=lbntxz9n&dl=1")
DESOBA_ZIP = DATA / "DESOBA" / "DESOBA_DATASET.zip"  # kept, so a rebuild does not download it again
# The shadow-free photo is used up to this many px around the object's shadow mask: on 32 of the 34
# images, 98-100 % of the editors' change lies there (the soft edge of a shadow is outside its mask).
PENUMBRA = 20
LUMA = np.array([0.299, 0.587, 0.114], np.float32)  # Rec. 601 luminance weights

# D_real: E1 photos with a shadow or a lighting trap; the new ones are in data/e3_sources.csv.
REAL = {
    "D02_white_sneaker": "soft contact shadow on white",
    "D06_logo_tshirt": "flat garment, faint shadow",
    "D07_hanger_shirt": "light falloff on a dark blue background, and a cast shadow",
    "D10_low_light": "dark vignette, not a shadow: false-shadow check",
    "D05_jump_white_studio": "control: white studio, almost no shadow, the output should not change",
    "D04_lace_dress": "textured ground (leaves, soil): expected failure",
}


def rel(path: Path) -> str:
    return path.relative_to(DATA).as_posix()


def save_png(a: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path)


def save_u16(a: np.ndarray, path: Path) -> None:
    """A [0, 1] map as a 16-bit PNG (8 bits would round T by up to 0.002); read it with cv2.IMREAD_UNCHANGED."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), (np.clip(a, 0, 1) * 65535).round().astype(np.uint16)):
        raise OSError(f"could not write {path}")


def resize_long(img: Image.Image, size: int) -> Image.Image:
    s = size / max(img.size)
    return img.resize((round(img.width * s), round(img.height * s)), Image.Resampling.LANCZOS)


def lit_background(kind: str, h: int, w: int) -> np.ndarray:
    top, bottom, _ = LITS[kind]
    column = np.linspace(top, bottom, h).round().astype(np.uint8)  # rounded once: the input and lit/ share it
    return np.broadcast_to(column[:, None, None], (h, w, 3)).copy()


def cast_shadow(alpha: np.ndarray, offset: float, sigma: float, angle: float) -> np.ndarray:
    """T = 1 - DARKNESS * blur(shift(alpha)): the subject's silhouette moved by `offset` px at `angle`."""
    h, w = alpha.shape
    move = np.float32([[1, 0, offset * np.cos(angle)], [0, 1, offset * np.sin(angle)]])
    # replicate the border: the body goes on past the frame (the portraits are cut), so does its shadow
    moved = cv2.warpAffine(alpha, move, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return 1 - DARKNESS * cv2.GaussianBlur(moved, (0, 0), sigma)


def synthetic_rows(rng: np.random.Generator) -> list[dict]:
    with open(DATA / "e2_subset.csv", encoding="utf-8") as f:
        people = [r for r in csv.DictReader(f) if r["group"] == "A_people"]
    root = OUT / "A_synthetic"
    rows = []
    for i, person in enumerate(people, 1):
        name = person["name"]
        img = resize_long(load_rgb(DATA / person["image"]), SIZE)
        matte = resize_long(Image.open(DATA / person["gt"]).convert("L"), SIZE)
        fg_path, gt_path = root / "fg" / f"{name}.png", root / "gt" / f"{name}.png"
        save_png(np.asarray(img), fg_path)
        save_png(np.asarray(matte), gt_path)
        fg = np.asarray(img, np.float32)
        alpha = np.asarray(matte, np.float32) / 255
        h, w = alpha.shape
        diag = np.hypot(h, w)

        # One direction per portrait, the same for all its shadows: to the left or right, 0 to 30 degrees down.
        tilt = rng.uniform(0, 30)
        right = rng.random() < 0.5
        angle = np.radians(tilt) if right else np.pi - np.radians(tilt)
        direction = f"shadow to the {'right' if right else 'left'}, {tilt:.0f} deg down"
        shadows = {}
        for kind, spec in SHADOWS.items():
            if spec is None:
                shadows[kind] = (np.ones_like(alpha), "")
                continue
            t = cast_shadow(alpha, spec[0] / 100 * diag, spec[1] / 100 * diag, angle)
            path = root / "shadow" / f"{name}_{kind}.png"
            save_u16(t, path)
            shadows[kind] = (t, rel(path))

        for bg, (_, _, why) in LITS.items():
            lit = lit_background(bg, h, w)
            lit_path = root / "lit" / f"{name}_{bg}.png"
            save_png(lit, lit_path)
            for kind, (t, t_path) in shadows.items():
                a = alpha[..., None]
                x = a * fg + (1 - a) * t[..., None] * lit + rng.normal(0, NOISE, fg.shape)
                im_path = root / "im" / f"{name}_{bg}_{kind}.png"
                save_png(np.clip(x, 0, 255).round().astype(np.uint8), im_path)  # PNG: JPEG would change the answer
                rows.append({"group": "A_synthetic", "source": "PPM-100 + synthetic shadow", "name": im_path.stem,
                             "image": rel(im_path), "gt": rel(gt_path), "gt_type": "alpha", "fg": rel(fg_path),
                             "lit": rel(lit_path), "shadow": t_path, "other": "", "background": bg,
                             "shadow_kind": kind, "note": f"{why}; {direction}" if t_path else why})
        print(f"synthetic {i}/{len(people)}", end="\r")
    print()
    return rows


def fetch_desoba() -> None:
    if DESOBA_ZIP.exists():
        return
    DESOBA_ZIP.parent.mkdir(parents=True, exist_ok=True)
    part = DESOBA_ZIP.with_suffix(".part")
    print("downloading DESOBA (1.86 GB) ...")
    req = urllib.request.Request(DESOBA_URL, headers={"User-Agent": "modelia-bg-change/0.1 (student project)"})
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    part.replace(DESOBA_ZIP)  # an interrupted download leaves a .part file, never a broken zip


def desoba_rows() -> list[dict]:
    fetch_desoba()
    z = zipfile.ZipFile(DESOBA_ZIP)  # read in place: only 34 images are needed

    def read(kind, name, mode):
        return np.asarray(Image.open(io.BytesIO(z.read(f"DESOBA_DATASET/{kind}/{name}"))).convert(mode))

    root = OUT / "B_desoba"
    ring = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * PENUMBRA + 1, 2 * PENUMBRA + 1))
    rows = []
    for name in z.read("DESOBA_DATASET/Testing_labels.txt").decode().split():
        objects = read("InstanceMask", name, "L")
        values = np.unique(objects[objects > 0])
        if len(values) != 1:
            continue  # DESOBA's "BOS-free" rule: one object and its shadow, no other object-shadow pair
        masks = read("ShadowMask", name, "L")
        own = masks == values[0]  # a shadow has the grey value of its object; 255 = shadow without an object
        other = (masks > 0) & ~own
        img, free = read("ShadowImage", name, "RGB"), read("DeshadowedImage", name, "RGB")

        # L: the photo without the object's shadow (DESOBA's own composite), other shadows kept.
        near = (cv2.dilate(own.astype(np.uint8), ring) > 0) & ~other & (objects == 0)
        lit = np.where(near[..., None], free, img)
        y_img, y_lit = img.astype(np.float32) @ LUMA, lit.astype(np.float32) @ LUMA
        t = np.where(near, np.clip(y_img / np.maximum(y_lit, 1), 0, 1), 1)  # exactly 1 where L is the photo

        stem = Path(name).stem
        paths = {k: root / k / f"{stem}.png" for k in ("im", "gt", "lit", "shadow", "other")}
        paths["im"].parent.mkdir(parents=True, exist_ok=True)
        paths["im"].write_bytes(z.read(f"DESOBA_DATASET/ShadowImage/{name}"))  # the original file, not re-encoded
        save_png(((objects > 0) * 255).astype(np.uint8), paths["gt"])
        save_png(lit, paths["lit"])
        save_u16(t, paths["shadow"])
        save_png((other * 255).astype(np.uint8), paths["other"])
        rows.append({"group": "B_desoba", "source": "DESOBA test (SOBA images)", "name": stem,
                     "image": rel(paths["im"]), "gt": rel(paths["gt"]), "gt_type": "binary", "fg": "",
                     "lit": rel(paths["lit"]), "shadow": rel(paths["shadow"]), "other": rel(paths["other"]),
                     "background": "real", "shadow_kind": "real",
                     "note": f"object {100 * (objects > 0).mean():.1f} %, its shadow {100 * own.mean():.1f} %, "
                             f"other shadows {100 * other.mean():.1f} % of the image"})
    if len(rows) != 34:
        raise SystemExit(f"{len(rows)} BOS-free test images, DESOBA's README says 34: check the zip")
    return rows


def real_rows() -> list[dict]:
    folder = OUT / "D_real" / "im"
    folder.mkdir(parents=True, exist_ok=True)
    with open(DATA / "e1_subset.csv", encoding="utf-8") as f:
        e1 = {r["name"]: r for r in csv.DictReader(f) if r["group"] == "D_real"}
    rows = []
    for name, why in REAL.items():
        src = DATA / e1[name]["image"]
        shutil.copy2(src, folder / src.name)
        rows.append({"group": "D_real", "source": e1[name]["source"], "name": name, "image": rel(folder / src.name),
                     "gt_type": "none", "note": why})
    with open(DATA / "e3_sources.csv", encoding="utf-8") as f:
        for src in csv.DictReader(f):
            dst = folder / src["file"]
            download(src["url"], dst)
            rows.append({"group": "D_real", "source": f"{src['source']} ({src['license']})", "name": dst.stem,
                         "image": rel(dst), "gt_type": "none", "note": src["hard_case"]})
    return [{k: r.get(k, "") for k in FIELDS} for r in rows]


def main():
    for need, how in (("e1_subset.csv", "make_e1_subset.py"), ("e2_subset.csv", "make_e2_subset.py")):
        if not (DATA / need).exists():
            raise SystemExit(f"data/{need} not found: run python data/{how} first")
    if OUT.exists():
        shutil.rmtree(OUT)  # rebuilt from scratch, so no file is left from an older list
    rows = synthetic_rows(np.random.default_rng(SEED)) + desoba_rows() + real_rows()
    with open(DATA / "e3_subset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} images -> {OUT} and e3_subset.csv")


if __name__ == "__main__":
    main()
