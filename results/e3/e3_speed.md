| step | hardware | images | median megapixels | median s | max s | first call s |
|---|---|---|---|---|---|---|
| BiRefNet on cuda | Tesla T4 | 268 | 0.70 | 0.45 | 0.50 | 31.41 |
| foreground estimation | Intel(R) Xeon(R) CPU @ 2.00GHz (2 threads) | 259 | 0.70 | 0.26 | 1.01 | 47.54 |
| shadow map B1 (flat) | Intel(R) Xeon(R) CPU @ 2.00GHz (2 threads) | 268 | 0.70 | 0.05 | 0.20 | nan |
| shadow map B2 (inpaint) | Intel(R) Xeon(R) CPU @ 2.00GHz (2 threads) | 268 | 0.70 | 0.14 | 0.51 | nan |

`first call s`: loading BiRefNet, or the first foreground estimation of the session (numba compile or cache load).