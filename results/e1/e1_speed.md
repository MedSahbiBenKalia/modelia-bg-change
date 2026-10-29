| model | device | input_px | hardware | backend | median_s | mean_s | max_s | load_s | mem_mb |
|---|---|---|---|---|---|---|---|---|---|
| BiRefNet | cuda | 1024 | Tesla T4 | torch cuda float16 | 0.40 | 0.42 | 0.53 | 11.98 | 2496.00 |
| BiRefNet_lite | cuda | 1024 | Tesla T4 | torch cuda float16 | 0.20 | 0.21 | 0.35 | 0.57 | 1674.00 |
| BiRefNet-matting | cuda | 1024 | Tesla T4 | torch cuda float16 | 0.40 | 0.42 | 0.56 | 0.97 | 2438.00 |
| BEN2 | cuda | 1024 | Tesla T4 | torch cuda (fp16 autocast) | 0.61 | 0.66 | 1.12 | 1.15 | 3212.00 |
| InSPyReNet | cuda | 1024 | Tesla T4 | torch cuda | 1.03 | 1.08 | 1.56 | 3.32 | 3694.00 |
| IS-Net | cuda | 1024 | Tesla T4 | onnxruntime CUDAExecutionProvider | 3.54 | 3.60 | 4.40 | 1.65 | 4.00 |
| U2-Net | cuda | 320 | Tesla T4 | onnxruntime CUDAExecutionProvider | 1.40 | 1.44 | 1.86 | 0.21 | 0.00 |