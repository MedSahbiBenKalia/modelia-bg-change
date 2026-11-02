# Hours

| Date | Hours | What |
|---|---|---|
| 26-28 Oct | 7 | Environment (venv, Python 3.13, CPU torch), repo skeleton, reading about the 8 candidate models, one demo notebook per model |
| 29 Oct | 2 | E1 data: DIS-VD category counts, benchmark subset (people / products / structure / real fashion photos), sources, licences, training-data leakage check |
| 29 Oct | 2.5 | E1 run on Colab (T4): notebook fixes for Colab (numpy restart, ONNX CPU fallback), results, paired test, model choice (BiRefNet), EXPERIMENTS.md E1.1 |
| 31 Oct | 1 | CLI (`change_bg.py`) and pipeline |
| 31 Oct | 1 | E2 data: edge-test set (15 portraits, 30 synthetic with known answer, 6 real, 1 glass), zoom picks from soft-pixel share |
| 31 Oct | 2 | E2 edges: notebook, matting and metric code with tests, Colab T4 run, analysis, EXPERIMENTS.md entry, `--edges` option in `bgchange/pipeline.py` and `change_bg.py` (default fg), tests |
| 31 Oct | 1.25 | E3 plan (model, methods, test design) and data: 225 synthetic inputs with a known shadow, 34 DESOBA photos with a hand-removed shadow, 9 real photos; `data/make_e3_subset.py`, `data/e3_subset.md`|
| 31 Oct | 1.5 | E3 code: `bgchange/shadow.py` (B1 flat and B2 inpainted shadow maps, strength) with tests, `over_colour` shade, `experiments/e3_shadows.ipynb` (methods A/B1/B2/C, true and BiRefNet alpha, sensitivity, figures, Drive results)|
| 1 Nov | 1.5 | E3 run on Colab (T4), results, analysis, EXPERIMENTS.md entry, decision (B2, off by default, strength 1), `--keep-shadows`, `--shadow-strength`, `--shadow-method` in `bgchange/pipeline.py` and `change_bg.py` with tests, README; E4 dropped for time (empty skeleton removed) |
| 2 Nov | 1.5 | README: results table, examples on two test photos (with and without shadow), what still fails, related work (3 references checked online), AI-usage table, what I would do next|
| **Total** | **21.25** | |
