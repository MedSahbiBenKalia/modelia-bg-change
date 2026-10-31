| step | hardware | images | median megapixels | median s per image | s per megapixel | load s | s for 2.6 MP |
|---|---|---|---|---|---|---|---|
| BiRefNet on cuda | Tesla T4 | 52 | 10.04 | 0.42 | nan | 29.14 | nan |
| foreground estimation (S+F) | Intel(R) Xeon(R) CPU @ 2.00GHz (2 threads) | 60 | 10.76 | 3.83 | 0.36 | 39.07 | 0.92 |

`load s`: loading BiRefNet, or the first foreground estimation of the session (numba compile or cache load).