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
