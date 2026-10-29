"""Build the E1 benchmark subset from DIS-VD, PPM-100 and a few fashion photos (see data/e1_subset.md).

Copies the chosen image / ground-truth pairs to data/e1/<group>/{im,gt}/,
downloads the group D photos listed in data/e1_fashion_sources.csv (no ground
truth) and writes everything to data/e1_subset.csv, so the notebook only reads that CSV.

    python data/make_e1_subset.py
"""
import csv
import random
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent
OUT = DATA / "e1"
SEED = 0
N_PEOPLE = 15

# Group C is hand-picked (stress cases that look like hard fashion inputs).
# Picked from a contact sheet of the Rack / CoatHanger / WineGlass / Basket /
# Umbrella / Handcraft images; the reason is kept next to each file.
STRUCTURE = {
    "11#Furniture#5#CoatHanger#2973928525_8d85ae066d_o": "coat rack with hanging clothes",
    "10#Frame#5#Rack#3834658574_a4af36cd8a_o": "thin wire racks, background visible through",
    "10#Frame#5#Rack#6982605453_20cfb8c092_o": "wire shelf with shoes, cluttered background",
    "10#Frame#5#Rack#IMG_20210512_231316": "thin metal hooks on a light background",
    "14#Kitchenware#10#WineGlass#14092554090_5af6e18034_o": "transparent glass, low light",
    "14#Kitchenware#10#WineGlass#4450148455_8f460f541a_o": "transparent glass, busy background",
    "5#Artifact#1#Basket#3342299538_59c014a904_o": "woven pattern (like knitwear) + strong cast shadow",
    "5#Artifact#1#Basket#IMG_20210518_194803": "mesh holes, saturated colour",
    "21#Tool#18#Umbrella#8197552642_6f092320e2_o": "person holding an object, thin umbrella ribs",
    "5#Artifact#3#Handcraft#10873642323_1bafce3aa5_o": "lace-like cut-out pattern",
}


def copy_pair(group, im, gt, source, gt_type, note=""):
    for kind, src in (("im", im), ("gt", gt)):
        dst = OUT / group / kind / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return {
        "group": group,
        "source": source,
        "name": im.stem,
        "image": f"e1/{group}/im/{im.name}",
        "gt": f"e1/{group}/gt/{gt.name}",
        "gt_type": gt_type,
        "note": note,
    }


def download(url, dst, tries=5):
    """Wikimedia answers 429 when called too fast: wait and retry."""
    req = urllib.request.Request(url, headers={"User-Agent": "modelia-bg-change/0.1 (student project)"})
    for i in range(tries):
        try:
            dst.write_bytes(urllib.request.urlopen(req, timeout=60).read())
            return
        except urllib.error.HTTPError as e:
            if e.code != 429 or i == tries - 1:
                raise
            time.sleep(10 * (i + 1))


def fashion_rows():
    """D: real fashion photos, no ground truth (visual comparison only)."""
    folder = OUT / "D_real" / "im"
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(DATA / "e1_fashion_sources.csv", encoding="utf-8") as f:
        for src in csv.DictReader(f):
            dst = folder / src["file"]
            if not dst.exists():
                download(src["url"], dst)
                time.sleep(2)
            rows.append({"group": "D_real", "source": f"{src['source']} ({src['license']})",
                         "name": dst.stem, "image": f"e1/D_real/im/{dst.name}", "gt": "",
                         "gt_type": "none", "note": src["hard_case"]})
    return rows


def main():
    rows = []

    # A: people and hair, random sample of PPM-100 (soft alpha mattes).
    ppm = DATA / "PPM-100"
    names = sorted(p.name for p in (ppm / "image").iterdir())
    for name in sorted(random.Random(SEED).sample(names, N_PEOPLE)):
        rows.append(copy_pair("A_people", ppm / "image" / name, ppm / "matte" / name,
                              "PPM-100", "alpha"))

    # B: fashion products, every DIS-VD Accessories image (binary masks).
    dis = DATA / "DIS-VD"
    for im in sorted((dis / "im").glob("1#Accessories#*.jpg")):
        rows.append(copy_pair("B_products", im, dis / "gt" / f"{im.stem}.png",
                              "DIS-VD", "binary", im.stem.split("#")[3]))

    # C: hard structures, hand-picked DIS-VD images (binary masks).
    for stem, reason in STRUCTURE.items():
        rows.append(copy_pair("C_structure", dis / "im" / f"{stem}.jpg", dis / "gt" / f"{stem}.png",
                              "DIS-VD", "binary", reason))

    rows += fashion_rows()

    with open(DATA / "e1_subset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} images -> {OUT} and e1_subset.csv")


if __name__ == "__main__":
    main()
