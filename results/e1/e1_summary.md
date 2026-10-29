### IOU (higher is better)

| model | A_people | B_products | C_structure | A+B |
|---|---|---|---|---|
| BiRefNet | 0.987 | 0.939 * | 0.895 * | 0.959 |
| BiRefNet_lite | 0.986 | 0.836 * | 0.866 * | 0.900 |
| BiRefNet-matting | 0.987 * | 0.671 | 0.580 | 0.807 |
| BEN2 | 0.923 | 0.913 * | 0.847 * | 0.917 |
| InSPyReNet | 0.984 | 0.952 | 0.946 | 0.965 |
| IS-Net | 0.928 | 0.947 | 0.941 | 0.939 |
| U2-Net | 0.902 | 0.310 | 0.401 | 0.563 |

### BIOU (higher is better)

| model | A_people | B_products | C_structure | A+B |
|---|---|---|---|---|
| BiRefNet | 0.946 | 0.901 * | 0.854 * | 0.920 |
| BiRefNet_lite | 0.936 | 0.790 * | 0.811 * | 0.852 |
| BiRefNet-matting | 0.952 * | 0.620 | 0.541 | 0.762 |
| BEN2 | 0.863 | 0.863 * | 0.792 * | 0.863 |
| InSPyReNet | 0.932 | 0.926 | 0.929 | 0.929 |
| IS-Net | 0.830 | 0.917 | 0.919 | 0.880 |
| U2-Net | 0.805 | 0.267 | 0.245 | 0.498 |

### MAE (lower is better)

| model | A_people | B_products | C_structure | A+B |
|---|---|---|---|---|
| BiRefNet | 0.003 | 0.012 * | 0.026 * | 0.008 |
| BiRefNet_lite | 0.004 | 0.051 * | 0.033 * | 0.031 |
| BiRefNet-matting | 0.003 * | 0.107 | 0.146 | 0.063 |
| BEN2 | 0.025 | 0.021 * | 0.055 * | 0.023 |
| InSPyReNet | 0.005 | 0.011 | 0.017 | 0.008 |
| IS-Net | 0.020 | 0.013 | 0.019 | 0.016 |
| U2-Net | 0.019 | 0.224 | 0.242 | 0.136 |

`*` training-data overlap:

- BiRefNet-matting on A_people: PPM-100 is in its training set
- BiRefNet on B_products: DIS-VD is its validation set
- BiRefNet on C_structure: DIS-VD is its validation set
- BiRefNet_lite on B_products: DIS-VD is its validation set
- BiRefNet_lite on C_structure: DIS-VD is its validation set
- BEN2 on B_products: trained on DIS5K, splits not stated
- BEN2 on C_structure: trained on DIS5K, splits not stated
- RMBG-2.0_API on B_products: may be trained on DIS-VD (claim by the BEN2 authors)
- RMBG-2.0_API on C_structure: may be trained on DIS-VD (claim by the BEN2 authors)