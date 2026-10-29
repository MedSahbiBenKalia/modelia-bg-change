# E1 benchmark subset

Built by `python data/make_e1_subset.py` (needs `data/DIS-VD/` and `data/PPM-100/`).
Output: `data/e1/<group>/{im,gt}/` (not committed) and `data/e1_subset.csv` (the list the notebook reads).

## Why this composition

Modelia's inputs are garment photos and people wearing them. A random DIS-VD sample would be
mostly buildings and furniture (Architecture + Furniture = 133 of 470, see `dis-vd.md`) and has no
person category, so the subset is built around four groups and **metrics are reported per group**.

| Group | Source | Images | How chosen | Ground truth | What it tests |
|---|---|---:|---|---|---|
| A_people | PPM-100 | 15 | random, `SEED = 0` (not cherry-picked) | soft alpha matte | people, hair, full and half body, busy outdoor backgrounds |
| B_products | DIS-VD Accessories | 20 | all of them | binary mask | bags, clothes, eyeglasses, jewelry, rings, shoes, watch |
| C_structure | DIS-VD | 10 | hand-picked, reason per image in the CSV `note` | binary mask | thin structures, transparency, lace-like and woven patterns, cast shadow |
| D_real | Wikimedia Commons (8) + GTB Cup site (2) | 10 | hand-picked hard cases, reason per image in `e1_fashion_sources.csv` | none (visual only) | white on white, hanger shot, logo t-shirt, lace, sheer veil, flyaway hair on white, low light |

Group C is a stress set, not a representative sample: hand-picking adds selection bias, so it is
reported separately and not mixed into the A/B averages.

## Training-data leakage (checked on the model cards, 2026-10-29)

| Model | Trained on (as stated by the authors) | DIS-VD (B, C) | PPM-100 (A) |
|---|---|---|---|
| BiRefNet (general) | DIS5K-TR, DIS-TEs, DUTS, HRSOD, UHRSD, HRS10K, P3M-10k, TR-humans | clean, but **DIS-VD is its reported test/validation set** (used for model selection) | clean |
| BiRefNet_lite | same list, swin_v1_tiny | same as above | clean |
| BiRefNet-matting | P3M-10k, TR-humans, AM-2k, AIM-500, Human-2k, Distinctions-646, HIM2K, **PPM-100** | clean | **leaked: trained on PPM-100** |
| BEN2 | "DIS5k and our 22K proprietary dataset" | unclear (which DIS5K splits is not stated) | unknown |
| IS-Net (general use) | DIS5K (exact splits of the general-use weights not stated) | unclear | unknown |
| U²-Net | DUTS-TR | clean | clean |
| InSPyReNet (transparent-background base) | not stated for this checkpoint (repo checkpoints use DUTS / HRSOD / UHRSD) | unknown | unknown |
| RMBG-2.0 | not public; the BEN2 card claims RMBG-2.0 "did not preserve the DIS 5k validation dataset" (a competitor's claim, not verified) | possibly leaked | unknown |

Consequence: BiRefNet-matting's score on group A is optimistic and is marked as such in the results.
Every public portrait-matting set we could use (P3M, AIM-500, AM-2k, PPM-100) is in its training list,
so we keep PPM-100 and flag it instead of switching dataset.

## Licences

| Data | Licence | Committed? |
|---|---|---|
| DIS5K / DIS-VD | DIS5K Terms of Use (PDF in the DIS repo); images from Flickr | no, download only |
| PPM-100 | images: Flickr CC (commercial use and mods allowed); **mattes: CC BY-NC-SA 4.0** | no, download only |
| Group D photos | per image, see `e1_fashion_sources.csv` (Commons images: CC BY / CC BY-SA, author and link listed) | no, downloaded by the script |
| D09, D10 (gtbcup.com) | **unknown**, no licence or photographer shown on the site: used for evaluation only, never redistributed, downloaded from the original URL | no |

PPM-100 mattes are non-commercial: fine for this evaluation, but they could not be used to train or
tune a model for Modelia's product.
