from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TARGET = DATA_DIR / "flight-dataset_saccade-evasion_augmented.hdf5"

ARCHIVE_URL = "https://janelia.figshare.com/ndownloader/articles/25309105/versions/4"
TARGET_NAME = "flight-dataset_saccade-evasion_augmented.hdf5"


def main() -> None:
    if TARGET.exists():
        print(f"Already present: {TARGET}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        archive = Path(temp_dir) / "flybody-datasets-v4.zip"
        print("Downloading the FlyBody supporting archive (~4 GB)...")
        print(ARCHIVE_URL)
        urllib.request.urlretrieve(ARCHIVE_URL, archive)

        print("Extracting the flight dataset...")
        with zipfile.ZipFile(archive) as zf:
            matches = [
                name for name in zf.namelist()
                if name.endswith("/" + TARGET_NAME) or name == TARGET_NAME
            ]
            if not matches:
                raise FileNotFoundError(
                    f"{TARGET_NAME} was not found in the Figshare archive"
                )

            source = matches[0]
            with zf.open(source) as src, TARGET.open("wb") as dst:
                shutil.copyfileobj(src, dst)

    print(f"Installed: {TARGET}")


if __name__ == "__main__":
    main()
