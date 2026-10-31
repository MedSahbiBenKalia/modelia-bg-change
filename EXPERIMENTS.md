# Experiments

One entry per try: idea -> what I did -> result -> decision (failures included).

## E1.0 Demos: does each candidate run, and what does it look like?

**Idea.** Before benchmarking, check that every candidate segmentation model runs locally on CPU and look at its mask on one hard case (fine hair).

**What I did.** One notebook per model in `experiments/demos/`, all with the same pipeline: load the image with the EXIF rotation applied, predict a soft alpha at the original size, composite over a plain colour with Porter-Duff *over* (`out = alpha * image + (1 - alpha) * colour`), and show the mask next to the ground truth. Test image: one PPM-100 portrait with fine hair (`5475151559_0245f5ca69_o.jpg`, 1957×1310). Timing is a single `%time` on the first call, on CPU, so it includes warm-up.

RMBG-2.0 is a reference row only. Its weights are CC BY-NC (non-commercial), so I called BRIA's public Hugging Face Space instead of downloading them. That timing includes the upload and the queue, so it can't be compared with the others.

**Result (one image, qualitative, not a benchmark):**

| Model | How it runs | Wall time (CPU, 1 run) | What I see vs the ground truth |
|---|---|---|---|
| BiRefNet | transformers, `trust_remote_code` | 39.1 s | close to GT, fine strands kept |
| BiRefNet_lite | transformers, `trust_remote_code` | 26.8 s | very close to BiRefNet |
| BiRefNet-matting | transformers, `trust_remote_code` | 38.7 s | close to GT, slightly softer strands |
| BEN2 | `ben2` package (GitHub) | 23.6 s | clean, but cuts some fine strands (harder edge) |
| InSPyReNet | `transparent-background` | 35.0 s | close to GT |
| IS-Net | `rembg` (ONNX) | 3.0 s | over-segments: grey background halo around the hair |
| U²-Net | `rembg` (ONNX) | 1.1 s | blurry, over-segments |
| RMBG-2.0 (reference) | public HF Space | 66 s (network + queue) | close to GT |

**Decision.** The older baselines (U²-Net, IS-Net) are roughly 8-35× faster on CPU but visibly worse on hair. The newer models (BiRefNet family, BEN2, InSPyReNet) look similar on this one image, and their size doesn't predict speed (BEN2 was the fastest of them), so I can't choose one by eye. The choice goes to the measured benchmark on ~30 DIS-VD images (IoU, MAE, Boundary IoU, seconds per image, peak memory) in `experiments/e1_models.ipynb`.

## E1.1 Benchmark: which segmentation model goes into the pipeline?

**Idea.** E1.0 could not separate the recent models by eye on one image. So I scored all of them against ground truth on people and on products, kept the groups apart, and looked at real fashion photos for what the metrics miss. The hypotheses were written before the run (top of `experiments/e1_models.ipynb`).

**What I did.**
- **Data** (`data/e1_subset.md`): A = 15 people (PPM-100, soft alpha mattes); B = 20 products (all of DIS-VD Accessories: 8 jewellery, 3 bags, 3 eyeglasses, 3 rings, 1 clothes, 1 shoe, 1 watch); C = 10 hand-picked thin or transparent structures (DIS-VD), a stress set reported apart; D = 10 real fashion photos without ground truth, judged by eye.
- **Models**: the 7 from E1.0, all wrapped the same way (PIL image in, full-size soft alpha out, with resizing and upsampling inside). RMBG-2.0 through its public Space, as a visual reference only.
- **Metrics per image**: IoU (alpha and ground truth thresholded at 0.5), Boundary IoU (band of 2 % of the image diagonal), MAE on the soft alpha. Speed is the median time per image (image decoding excluded) after one untimed warm-up image; memory is GPU memory in use above the level before the model was loaded.
- **One run** on Google Colab (Tesla T4, torch 2.11 + CUDA 13). BiRefNet models in fp16, BEN2 with fp16 autocast, InSPyReNet in fp32 (its package has no fp16 path). No full CPU run.
- The two best models were close, so I compared them **image by image** (`experiments/e1_paired.py`): mean per-image difference, number of images each one wins, and a Wilcoxon signed-rank test (paired, and it does not assume normal differences; a few failures skew them).

**Result.** Tables in `results/e1/`: `e1_summary.md` (per group), `e1_metrics.csv` (per image), `e1_speed.md`, `e1_paired.md`.

People + products (A+B, 35 images), worst single image per group, speed and memory on the T4:

| Model | IoU | BIoU | MAE | worst IoU A / B | s per image | GPU memory |
|---|---:|---:|---:|---|---:|---:|
| InSPyReNet | 0.965 | 0.929 | 0.008 | 0.92 / 0.84 | 1.03 (fp32) | 3694 MB |
| **BiRefNet** | 0.959 | 0.920 | 0.008 | 0.94 / 0.85 | 0.40 (fp16) | 2496 MB |
| IS-Net | 0.939 | 0.880 | 0.016 | 0.49 / 0.85 | CPU fallback | - |
| BEN2 | 0.917 | 0.863 | 0.023 | 0.31 / 0.81 | 0.61 (fp16 autocast) | 3212 MB |
| BiRefNet_lite | 0.900 | 0.852 | 0.031 | 0.95 / 0.15 | 0.20 (fp16) | 1674 MB |
| BiRefNet-matting | 0.807 | 0.762 | 0.063 | 0.94 / 0.08 | 0.40 (fp16) | 2438 MB |
| U²-Net | 0.563 | 0.498 | 0.136 | 0.00 / 0.00 | CPU fallback | - |

BiRefNet vs InSPyReNet, image by image (difference = BiRefNet - InSPyReNet):

| Group | IoU diff | BiRefNet better on | p | Also |
|---|---:|---:|---:|---|
| A people (15) | +0.003 | 13/15 | 0.008 | MAE lower on 15/15 (p < 0.001) |
| B products (20) | -0.013 | 7/20 | 0.017 | BIoU -0.025 (p = 0.019) |
| A+B (35) | -0.006 | 20/35 | 0.61 | BIoU p = 0.63; MAE lower on 29/35 (p = 0.003), but the mean MAE difference is ~0 |
| C stress set (10) | -0.051 | 1/10 | 0.010 | |

Hypotheses:
1. *BiRefNet is best on B and C*: **rejected.** InSPyReNet and IS-Net score higher on both, although DIS-VD is BiRefNet's own validation set.
2. *Matting-style models have the lowest MAE on A*: **partly.** BiRefNet-matting ties BiRefNet (0.003), and its score is optimistic (PPM-100 is in its training set). BEN2 is second worst on A (0.025): in one portrait it takes the desk as part of the person (IoU 0.31).
3. *BiRefNet_lite is clearly faster for a small loss*: **partly.** It is 2× faster with the same accuracy on people, but on products it fails badly 3 times (IoU 0.15, 0.44, 0.57).
4. *U²-Net is clearly the worst on Boundary IoU*: **confirmed** (0.50 on A+B).
5. *Every model struggles on D09 and on the bottom of the dress in D10*: **rejected for D09, partly true for D10** (below).

Real photos (group D), by eye at figure size (about 250 px per panel), so hair detail is not judged here:
- D10 (low light): InSPyReNet lets the background show through the bottom of the black dress; IS-Net and U²-Net do it more, BiRefNet_lite slightly. BiRefNet, BiRefNet-matting and BEN2 keep the dress whole.
- D09 (white on white): the five recent models keep the white top; IS-Net makes it semi-transparent and U²-Net removes it.
- D07 (hanger): BiRefNet removes the hanger hook (the shirt is intact); BEN2 keeps the clothes rail.
- D03 (shirt rack): BiRefNet-matting deletes most of the garments. D08 (runway): it keeps the second model walking behind.
- D02, D05: IS-Net cuts a hole in the white sneaker; IS-Net and U²-Net lose the white stripes of the socks.

What did not work, and caveats:
- **ONNX Runtime on the GPU.** For IS-Net and U²-Net, ONNX Runtime's CUDA backend failed on the first image (cuDNN `HEURISTIC_QUERY_FAILED`, cuDNN 9.27 from torch's CUDA 13 build) and fell back to the CPU, so they have no GPU timing. The likely cause is a mismatch between the `onnxruntime-gpu` build and CUDA 13 / cuDNN 9.27 (not verified). Their accuracy is unaffected (same fp32 model). Their saved timing rows still say `CUDAExecutionProvider`, because the notebook read the backend before the first image; their GPU memory (+4 MB, +0 MB) shows the truth. I fixed this in the notebook after the run (one blank image runs before the backend is read) but did not re-run: both are baselines, not candidates.
- **Speed is not compared like for like**: BiRefNet ran in fp16 and InSPyReNet in fp32, so 0.40 s vs 1.03 s says little about the models themselves. In E1.0 on CPU (one run each, warm-up included) they were close: 39.1 s vs 35.0 s.
- **fp16 vs fp32**: on group A, BiRefNet's means in this fp16 run are the same to 3 decimals as in an interrupted fp32 CPU run (IoU 0.987, BIoU 0.946, MAE 0.003; 15 images, BiRefNet only).
- **Training-data overlap** (`data/e1_subset.md`): the B and C scores of BiRefNet and BiRefNet_lite may be optimistic (DIS-VD is their validation set), and so may BEN2's and IS-Net's (trained on DIS5K, splits not stated); BiRefNet-matting's A score is optimistic (PPM-100 is in its training set). The IS-Net mark was added after the run, so the notebook's saved tables lack it.
- **Ground truth for glass**: DIS ground truth marks glass (eyeglass lenses, wine glasses) as fully opaque. BiRefNet-matting makes glass see-through and scores 0.02 to 0.19 IoU on those images. Neither is right for a colour change: opaque glass keeps the old background inside it. To look at in E2.
- **B is mostly accessories** (8 of 20 are jewellery, 1 is clothing), so "products" here does not mean garments; D is the only garment evidence, and it has no ground truth.
- **RMBG-2.0**: the free ZeroGPU quota ran out after D01 to D03, so it is only a visual reference on those three.
- **Setup on Colab**: pip upgraded numpy after it had been imported, and the next import failed (`cannot import name '_slice'`); restarting the session fixed it. The setup cell now detects this and asks for the restart.

**Decision.** BiRefNet (general) goes into the pipeline; InSPyReNet is the runner-up.
1. On people + products together the two are tied: the IoU and BIoU differences are within noise (p = 0.61 and 0.63). Choosing InSPyReNet for a 0.006 higher mean would over-read 35 images.
2. Where they differ, BiRefNet wins on people: its MAE is lower on all 15 portraits, and MAE on soft mattes measures the edge that shows when a person is put on a new colour. Modelia's output is people wearing garments. InSPyReNet wins on accessories and thin structures, and on the one shoe (IoU 0.98 vs 0.90).
3. On the real photos, InSPyReNet's failure is the worse one for garments: it makes part of a black dress see-through (D10), while BiRefNet drops a hanger hook and keeps the garment whole (D07). This is one image each: a hint, not proof.
4. The brief suggests BiRefNet ("or anything you think works better"), and this benchmark does not show that InSPyReNet works better overall. BiRefNet is MIT licensed, and the whole family loads the same way.

What would change the decision: E2 or E4 showing better full-resolution edges with InSPyReNet, or a garment set with ground truth (flat-lay, hanger, on-model) that reverses the product result. Ruled out: BiRefNet_lite (3 failures out of 20 products), BiRefNet-matting (fails on objects that are not people), BEN2 (takes other objects as foreground: the desk, the clothes rail), IS-Net and U²-Net (old baselines with large failures).

Figures: the hard-case grids and the D figures are in the notebook outputs and in `results/e1/figures/` (not committed). They show images from DIS-VD (DIS5K terms of use), PPM-100 (Flickr CC images; mattes CC BY-NC-SA 4.0) and Wikimedia Commons (authors and licences in `data/e1_fashion_sources.csv`). The D09 and D10 figures are never shown in the notebook.

## E2 Edges: hard mask, soft alpha, or soft alpha + foreground estimation?

**Idea.** At an edge pixel the photo is already a mix, I = αF + (1 − α)B (F: the subject's colour, B: the old background). Compositing the photo itself on a new colour C gives the right answer αF + (1 − α)C plus α(1 − α)(B − F): the old background leaks into the hair, even with a perfect alpha. So I compared three edge modes (`bgchange/matting.py`):
- **H**: BiRefNet's alpha thresholded at 0.5;
- **S**: the soft alpha, composited with the photo (the pipeline after E1);
- **S+F**: the soft alpha, composited with the foreground colour estimated by pymatting's `estimate_foreground_ml` (Germer et al., ICPR 2020, default settings).

ViTMatte refinement was left out for time. The hypotheses were written before the run (top of `experiments/e2_edges.ipynb`).

**What I did.**
- **Data** (`data/e2_subset.md`): the 15 E1 portraits (PPM-100, soft mattes) for the alpha. 30 synthetic images: each portrait used as F with its true alpha, on `#00FF00` and on `#F0F0F0`, so that the right answer on any new colour is known exactly. 6 real photos (D01, D05, D08, D09, D10, and D06 as a control) and the DIS eyeglasses, judged by eye.
- **Metrics**, in the edge region of the ground truth (a band of 2 % of the image diagonal on both sides of the outline, plus every soft pixel), split into soft pixels (0.02 < α < 0.98, mostly hair) and the sharp rest:
  - alpha: MAE and gradient error (Rhemann et al., CVPR 2009);
  - colour: mean absolute RGB difference to the right answer, in 8-bit units (0 to 255), on two new colours (`#1E90FF` and white), with the true alpha (only the compositing error is left) and with BiRefNet's alpha (the real pipeline);
  - S+F vs S per portrait: number of wins and a Wilcoxon signed-rank test.
- **One run on Colab**: BiRefNet on the Tesla T4 (fp16), foreground estimation on its 2-thread CPU. Two checks passed: BiRefNet's full-image MAE equals E1's (largest difference 0.0000), and with the true alpha, S's measured error equals the predicted leak α(1 − α)|B − F| on every pixel of the 30 images, to 1.42 levels (rounding allows 1.5).

**Result.** Tables in `results/e2/`: `e2_alpha_summary.md`, `e2_colour_summary.md`, `e2_paired.md`, `e2_speed.md`, `e2_hypotheses_numbers.md`.

Alpha on the 15 portraits (mean):

| | MAE soft (hair) | MAE sharp (body, clothes) | MAE edge | gradient error edge (×1000) |
|---|---:|---:|---:|---:|
| S (BiRefNet alpha) | 0.181 | 0.0057 | 0.018 | 0.30 |
| H (thresholded) | 0.265 | 0.0039 | 0.024 | 0.63 |

Soft pixels are 9 % of the edge area but hold 70 % of BiRefNet's alpha error there (more than their area share on 15/15). Across the portraits, BiRefNet's E1 MAE grows with the share of soft pixels (Spearman 0.84, p = 0.0001).

Colour error in the soft pixels, new colour `#1E90FF` (mean over the 15 portraits, 0 to 255):

| Alpha | Old background | H | S | S+F |
|---|---|---:|---:|---:|
| true | green | 22.0 | 16.6 | 5.6 |
| true | near-white | 19.7 | 17.5 | 5.4 |
| BiRefNet | green | 29.5 | 26.1 | 20.0 |
| BiRefNet | near-white | 23.8 | 23.0 | 14.8 |

On white, S+F is also the best everywhere; H beats S only when near-white is moved to white (old and new background almost the same).

S+F vs S, image by image: with the true alpha S+F removes 66 to 69 % of the error (15/15 per condition, p = 6.1e-05, the smallest possible with 15 pairs). With BiRefNet's alpha it removes 14 to 41 % (better on 59 of 60 comparisons; the one loss is -3 %, the small jumping figure 35998070871 moved from near-white to white).

Speed (`e2_speed.md`): BiRefNet takes 0.42 s per image on the T4 (median image 10 MP). Foreground estimation takes 0.36 s per megapixel on Colab's CPU (0.92 s for 2.6 MP) and 0.24 s per megapixel on my laptop (smoke test), where BiRefNet takes about 31 s per image. The first call of a session compiles pymatting's numba code: 39 s on Colab, once per install (numba caches it on disk).

Real photos, by eye (figures in `results/e2/figures/`, not committed):
- White backgrounds moved to near-black (D05, D09): S leaves a grey-white halo in the hair; S+F gives the hair back its own colour. Dark backgrounds moved to white (D10, D01): S leaves a brown or green haze; S+F is cleaner. H is jagged and loses the strands everywhere.
- D06 control: inside the garment S+F differs from S by at most 1 level; the differences are on the outline only (up to 81 levels: the fringe that S+F removes). The logo is untouched.
- Glass: the lenses keep the old background in every mode.

Hypotheses:
1. *Most of BiRefNet's remaining alpha error is in the soft pixels*: **confirmed** (70 % of the error on 9 % of the area, 15/15).
2. *H is worse than S on hair*: **confirmed** (MAE and gradient error, 15/15). But H is better on the sharp part of the outline (15/15): BiRefNet's alpha is slightly too soft on body and clothes edges.
3. *With the true alpha, S's error is the predicted leak, larger on green than on near-white, and S+F at least halves it*: **partly.** The formula and "at least halves" are confirmed (-66 to -69 %). "Larger on green" is rejected (16.6 vs 17.5): the leak grows with |B − F|, and dark hair is as far from near-white as from green.
4. *With BiRefNet's alpha, S+F still beats S, by less*: **confirmed** (59/60, -14 to -41 %).
5. *Foreground estimation takes under 4 s for 2.6 MP*: **confirmed** (0.92 s on Colab, 0.62 s on the laptop), plus the first-call compile.
6. *D06: S+F changes the garment by at most 3 levels*: **confirmed** (1 level).
7. *Glass fails*: **confirmed**, as expected.

What did not work, and caveats:
- **Most of the remaining error is BiRefNet's alpha, not the compositing.** With BiRefNet's alpha, S+F still has 9.5 to 27 levels of error in the hair, against about 5.5 with the true alpha. In the zoomed crops, BiRefNet's hair alpha is a smooth blur without strands. A guess, not measured: part of it comes from predicting at 1024 px and upsampling to 10 to 20 MP (the E4 question, dropped for time).
- **Transparency is not handled.** BiRefNet makes the glass opaque, and also the sheer skirt of D08 (alpha > 0.95 on 98.9 % of a crop of the lower skirt, measured after the run), so what is seen through them stays. In D08 the zoom rule picked the bouquet (the softest alpha), so the sheer fabric was not tested at the edge.
- **Soft pixels are not always hair**: in 49020513826 they are mostly a glass held in the hand (soft in the ground truth, opaque for BiRefNet).
- **The colour score is not perceptual** (plain RGB differences, not a colour-difference formula such as CIEDE2000). It ranks the methods on the same pixels but does not say how visible an error is.
- **The synthetic backgrounds are plain** (green, near-white), and in the hair the PPM-100 photos already contain some of their original background. Busy real backgrounds were only judged by eye. 15 portraits from one dataset, no confidence intervals.
- **On a GPU server, foreground estimation (CPU) is slower than the segmentation**: 3.8 s vs 0.42 s for a ~10 MP image. On a laptop CPU it is about 2 % of the time.
- The notebook's Wilcoxon column shows 0.0 (rounded display); the real values are in `e2_paired.md`.

**Decision.** S+F becomes the default edge mode of the pipeline and the CLI (`--edges fg`; `soft` and `hard` stay available for comparison). It beats S on 59 of 60 comparisons, leaves the inside of the garment unchanged, and costs under 1 s for a 2.6 MP photo on CPU. H is ruled out for people: worse in the hair on every portrait.

What would change the decision: a perceptual colour metric, or busy real backgrounds where foreground estimation makes visible artefacts. Next steps, not done: a better alpha in the hair (matting refinement such as ViTMatte with a trimap built from BiRefNet's mask, or higher-resolution inference), and transparency-aware matting for glass and sheer garments.
