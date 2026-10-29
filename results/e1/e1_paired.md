# E1: paired comparison

Same images for both models: mean per-image difference, number of images where the first model is better, two-sided Wilcoxon signed-rank p-value. Made by `experiments/e1_paired.py`.

## BiRefNet vs InSPyReNet

| group | metric | mean diff (BiRefNet - InSPyReNet) | BiRefNet better on | p |
|---|---|---:|---:|---:|
| A_people | iou | +0.0033 | 13/15 | 0.008 |
| A_people | biou | +0.0141 | 12/15 | 0.095 |
| A_people | mae | -0.0014 | 15/15 | <0.001 |
| B_products | iou | -0.0130 | 7/20 | 0.017 |
| B_products | biou | -0.0254 | 7/20 | 0.019 |
| B_products | mae | +0.0014 | 14/20 | 0.245 |
| A+B | iou | -0.0060 | 20/35 | 0.610 |
| A+B | biou | -0.0085 | 19/35 | 0.633 |
| A+B | mae | +0.0002 | 29/35 | 0.003 |
| C_structure | iou | -0.0510 | 1/10 | 0.010 |
| C_structure | biou | -0.0748 | 1/10 | 0.004 |
| C_structure | mae | +0.0086 | 3/10 | 0.049 |

## BiRefNet vs BiRefNet_lite

| group | metric | mean diff (BiRefNet - BiRefNet_lite) | BiRefNet better on | p |
|---|---|---:|---:|---:|
| A_people | iou | +0.0013 | 11/15 | 0.055 |
| A_people | biou | +0.0100 | 10/15 | 0.151 |
| A_people | mae | -0.0004 | 10/15 | 0.107 |
| B_products | iou | +0.1024 | 15/20 | 0.004 |
| B_products | biou | +0.1110 | 17/20 | 0.001 |
| B_products | mae | -0.0382 | 13/20 | 0.019 |
| A+B | iou | +0.0591 | 26/35 | <0.001 |
| A+B | biou | +0.0677 | 27/35 | <0.001 |
| A+B | mae | -0.0220 | 23/35 | 0.005 |
| C_structure | iou | +0.0292 | 9/10 | 0.064 |
| C_structure | biou | +0.0430 | 9/10 | 0.049 |
| C_structure | mae | -0.0069 | 8/10 | 0.084 |
