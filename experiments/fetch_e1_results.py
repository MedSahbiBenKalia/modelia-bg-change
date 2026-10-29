"""Copy the E1 results of the Colab run (Google Drive: modelia-e1/results/) into results/e1/.

The E1 run was made on Colab (Tesla T4), so its tables, timings and figures are the E1 results.
Per-image alphas and composites stay on Drive. The figures are copied for viewing but not
committed (they show dataset images, see .gitignore). The Drive folder is shared by link, so
gdown needs no login. If Google throttles gdown ("too many accesses"), download the folder
from the Drive web page instead.

    python experiments/fetch_e1_results.py
"""
from pathlib import Path

import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/1kBxHVZqNXHGMIEhsNyoxQw_4XF3tB1xk?usp=sharing"
OUT = Path(__file__).resolve().parents[1] / "results" / "e1"
SKIP = ("alphas", "composites")  # per-image outputs, kept on Drive


def main():
    files = gdown.download_folder(FOLDER_URL, skip_download=True, quiet=True)
    n = 0
    for f in files:
        parts = Path(f.path.replace("\\", "/")).parts  # gdown uses the OS separator
        if parts[0] != "results" or (len(parts) > 2 and parts[1] in SKIP):
            continue  # input images, or per-image outputs
        dst = OUT.joinpath(*parts[1:])
        dst.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=f.id, output=str(dst), quiet=True)
        n += 1
    print(f"{n} files -> {OUT}" if n else "no results/ on Drive: has the Colab run finished?")


if __name__ == "__main__":
    main()
