S+F vs S in the soft pixels, per portrait (lower error is better).

| colour | alpha | old background | n | S | S+F | S+F / S | S+F better on | wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| #1E90FF | birefnet | #00FF00 | 15 | 26.062 | 19.967 | 0.766 | 15 | 6.1e-05 |
| #1E90FF | birefnet | #F0F0F0 | 15 | 23.046 | 14.824 | 0.643 | 15 | 6.1e-05 |
| #1E90FF | true | #00FF00 | 15 | 16.571 | 5.629 | 0.340 | 15 | 6.1e-05 |
| #1E90FF | true | #F0F0F0 | 15 | 17.543 | 5.369 | 0.306 | 15 | 6.1e-05 |
| #FFFFFF | birefnet | #00FF00 | 15 | 31.256 | 26.986 | 0.863 | 15 | 6.1e-05 |
| #FFFFFF | birefnet | #F0F0F0 | 15 | 16.114 | 9.542 | 0.592 | 14 | 0.00012 |
| #FFFFFF | true | #00FF00 | 15 | 16.577 | 5.624 | 0.339 | 15 | 6.1e-05 |
| #FFFFFF | true | #F0F0F0 | 15 | 17.541 | 5.370 | 0.306 | 15 | 6.1e-05 |