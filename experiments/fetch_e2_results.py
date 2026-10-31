"""Copy the E2 results of the Colab run (Google Drive: modelia-e2/results/) into results/e2/.

Share the Drive folder modelia-e2 by link (anyone with the link can view) and paste the link into
FOLDER_URL. Alphas and per-image rows stay on Drive. Figures and composites are copied for viewing
but not committed (they show dataset images, see .gitignore). If Google throttles gdown ("too many
accesses"), download the folder from the Drive web page instead.

    python experiments/fetch_e2_results.py
"""
from pathlib import Path

import gdown

FOLDER_URL = ""  # share link of MyDrive/modelia-e2
OUT = Path(__file__).resolve().parents[1] / "results" / "e2"
SKIP = ("alphas", "per_image", "smoke")  # per-image outputs and the smoke test, kept on Drive


def main():
    if not FOLDER_URL:
        raise SystemExit("paste the share link of MyDrive/modelia-e2 into FOLDER_URL first")
    files = gdown.download_folder(FOLDER_URL, skip_download=True, quiet=True)
    n = 0
    for f in files:
        parts = Path(f.path.replace("\\", "/")).parts  # gdown uses the OS separator
        if "results" not in parts:
            continue  # input images
        rel = parts[parts.index("results") + 1:]
        if not rel or rel[0] in SKIP:
            continue
        dst = OUT.joinpath(*rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=f.id, output=str(dst), quiet=True)
        n += 1
    print(f"{n} files -> {OUT}" if n else "no results/ on Drive: has the Colab run finished?")


if __name__ == "__main__":
    main()
