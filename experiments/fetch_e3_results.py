"""Copy the E3 results of the Colab run (Google Drive: modelia-e1/results_e3/) into results/e3/.

Same shared Drive folder as E1 and E2 (it holds e1/, e2/, e3/ and their results). Alphas and
per-image rows stay on Drive. Figures and composites are copied for viewing but not committed
(they show dataset images, see .gitignore). The folder is shared by link, so gdown needs no login.
Needs gdown 6: older versions stop at folders of more than 50 files, and e3/A_synthetic/im/ has 225.
If Google throttles gdown ("too many accesses"), download the folder from the Drive web page instead.

    python experiments/fetch_e3_results.py
"""
from pathlib import Path

import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/1kBxHVZqNXHGMIEhsNyoxQw_4XF3tB1xk?usp=sharing"
OUT = Path(__file__).resolve().parents[1] / "results" / "e3"
SKIP = ("alphas", "per_image", "smoke")  # per-image outputs and the smoke test, kept on Drive


def main():
    files = gdown.download_folder(FOLDER_URL, skip_download=True, quiet=True)
    n = 0
    for f in files:
        parts = Path(f.path.replace("\\", "/")).parts  # gdown uses the OS separator
        if parts[0] != "results_e3" or (len(parts) > 2 and parts[1] in SKIP):
            continue  # E1/E2 files, E3 input images, or per-image outputs
        dst = OUT.joinpath(*parts[1:])
        dst.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=f.id, output=str(dst), quiet=True)
        n += 1
    print(f"{n} files -> {OUT}" if n else "no results_e3/ on Drive: has the Colab run finished?")


if __name__ == "__main__":
    main()
