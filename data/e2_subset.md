# E2 edge-test set

Built by `python data/make_e2_subset.py` (needs `data/e1/`, built by `make_e1_subset.py`; about 40 s).
Output: `data/e2/<group>/{im,gt}/` (not committed, 206 MB) and `data/e2_subset.csv` (the list the notebook reads).
E2 downloads nothing new: the real images are copies of E1 files, and the synthetic ones are made from them.

```
data/e2/
  A_people/{im,gt}/   15 portraits and their mattes (copied from E1)
  A_synthetic/im/     30 PNG inputs: <portrait>_bg00FF00.png, <portrait>_bgF0F0F0.png
  D_real/im/          6 real photos (copied from E1)
  B_glass/{im,gt}/    1 eyeglasses image and its mask (copied from E1)
```

The `A_synthetic` rows have no files of their own for the ground truth: their `gt` column points at the
matte in `A_people/gt/` and their `fg` column at the photo in `A_people/im/`, the true matte and the
true foreground the input was made from.

## Why this composition

E2 compares what we do at the edge of the mask (hard threshold, raw soft alpha, soft alpha with
foreground colour estimation). Inside the subject and in the background, all three give the same pixels,
so every image here has a lot of soft edge: hair, sheer fabric, motion blur.

| Group | Rows | From | Ground truth | What it measures |
|---|---:|---|---|---|
| A_people | 15 | the E1 PPM-100 portraits (random, `SEED = 0`) | soft alpha matte | the alpha: hard vs soft, in the edge band |
| A_synthetic | 30 | the same 15 portraits × 2 plain backgrounds | the exact composite (see below) | the colour of the composite: how much old background leaks into the edge |
| D_real | 6 | E1 group D, chosen for soft edges | none | by eye, zoomed crops |
| B_glass | 1 | E1 group B (eyeglasses) | binary mask (lenses marked opaque) | by eye: an expected failure |

DIS-VD is not used for the alpha metrics: its masks are binary, so a hard threshold would win by construction.

## A_synthetic: a test with a known answer

Each PPM-100 photo is used as the foreground F, with its ground-truth matte α, over a plain colour B:
`input = α·F + (1 − α)·B` (the same `over_colour` as the pipeline). The right answer on a new
colour C is then known exactly: `α·F + (1 − α)·C`. The inputs are saved as PNG, not JPEG:
compression would change the edge pixels, and the answer would no longer be exact. Checked on one
image after the build: the saved input is within 0.5 grey level of the formula everywhere, and every
pixel with α = 0 is exactly the background colour.

| Background | Why |
|---|---|
| `#00FF00` | saturated green: worst case for colour spill |
| `#F0F0F0` | near-white studio: the common real case |

I wanted a busy photo as the second background, but none of our licensed photos has a large area
without a person in it (D04's ground is 640 px wide and has legs in it). Busy backgrounds are covered
by the real A photos (mostly outdoors) and by D01, judged by eye.

Caveats:
- In the hair, a PPM-100 photo already contains some of its original background, so F there is not the
  true colour of the hair. The test is still exact, because the right answer is built from the same F.
- PPM-100 mattes are JPEG files, so α has small compression noise (a few levels near 0 and 1).
  Same reason: the input and the right answer use the same α.
- Input and answer are rounded to 8 bits, so "exact" means within one grey level.

## Zoomed crops (A)

Picked from two measurements, not by eye: the share of soft pixels (0.02 < α < 0.98) in the
ground-truth matte, and BiRefNet's per-image MAE in E1 (`results/e1/e1_metrics.csv`). The soft
shares come from a scratch check and are recomputed in the first cell of `experiments/e2_edges.ipynb`.

| Image | Why |
|---|---|
| 14996438642 | most hair: 5.2 % of pixels soft |
| 34504878700 | second most hair: 2.8 % soft, arms raised |
| 24179781765 | highest BiRefNet MAE in E1 (0.0067), long hair |
| 49020513826 | same MAE (0.0067), seated |
| 35998070871 | counter-example: worst BiRefNet IoU on A in E1 (0.942) with almost no hair (0.3 % soft) |

## Real photos (D) and glass

| Image | Why | Shown in figures? |
|---|---|---|
| D01 vintage shoot | backlit strands on a busy outdoor background; fuzzy knit | yes (CC BY 3.0 de) |
| D05 jump, white studio | flyaway hair on white; motion blur on the feet | yes (CC BY 2.0) |
| D08 runway sheer veil | soft alpha on the garment itself (sheer fabric) | yes (CC BY-SA 4.0) |
| D09 white on white | light curly hair and a white top on near-white | **no** (licence unknown) |
| D10 low light | blond curls on a dark brown background | **no** (licence unknown) |
| D06 logo t-shirt | control: hard edges and a logo. All edge modes should give the same image, and the logo must not change | yes (CC BY-SA 4.0) |
| Eyeglasses 4710367344 (DIS-VD) | glass: the mask is opaque inside the lenses, so the old background stays there whatever we do at the edge | DIS5K terms of use |

Not used: D02 (white sneaker), D03 (shirt rack), D04 (lace dress) and D07 (hanger) have
segmentation or shadow problems, not soft edges (D07 goes to E3). `data/my_images/1.jpeg` and
`2.jpeg` are the same files as D09 and D10 (same md5). `data/my_images/1.jpg` (large curly hair on a dark
background) is left out until its source and licence are written down.

## Resolution

Native size, as in E1. A scratch timing of the foreground estimation (pymatting `estimate_foreground_ml`,
one run, this laptop's CPU) gave 0.8 s at 2.8 MP and 7.1 s at 20 MP, so there is no need to downscale.

## Licences

Same as E1 (`e1_subset.md`). The synthetic inputs are made from PPM-100 mattes (CC BY-NC-SA 4.0),
so they and any figure showing them stay local, like the PPM-100 figures in E1.
