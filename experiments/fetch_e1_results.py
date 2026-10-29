"""Copy the E1 results written on Colab (Google Drive: modelia-e1/results/) into results/e1/.

The Drive folder is shared by link, so gdown can list and download it without a login.
Per-image folders (alphas/, composites/) come as the zips made in section 8 of
e1_models.ipynb and are unpacked here.

    python experiments/fetch_e1_results.py
"""
import shutil
from pathlib import Path

import gdown

FOLDER_URL = "https://drive.google.com/drive/folders/1XorV6sxMkklR9busqG-He3I4Ay0qsZAO"
OUT = Path(__file__).resolve().parents[1] / "results" / "e1"


def main():
    files = gdown.download_folder(FOLDER_URL, skip_download=True, quiet=True)
    n = 0
    for f in files:
        parts = Path(f.path.replace("\\", "/")).parts
        if parts[0] != "results" or parts[1] in ("alphas", "composites"):
            continue  # input images, or per-image files that are also in the zips
        dst = OUT.joinpath(*parts[1:])
        dst.parent.mkdir(parents=True, exist_ok=True)
        gdown.download(id=f.id, output=str(dst), quiet=True)
        n += 1
    for name in ("alphas", "composites"):
        if (OUT / f"{name}.zip").exists():
            shutil.unpack_archive(OUT / f"{name}.zip", OUT / name)
    print(f"{n} files -> {OUT}")


if __name__ == "__main__":
    main()
