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
