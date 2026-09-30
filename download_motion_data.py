from __future__ import annotations

import json
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TARGET = DATA_DIR / "flight-dataset_saccade-evasion_augmented.hdf5"
WING_TARGET = DATA_DIR / WING_NAME

ARTICLE_ID = 25309105
VERSION = 4
ARCHIVE_NAME = "datasets_flight-imitation.zip"
TARGET_NAME = "flight-dataset_saccade-evasion_augmented.hdf5"
WING_NAME = "wing_pattern_fmech.npy"
API_URL = (
    f"https://api.figshare.com/v2/articles/{ARTICLE_ID}"
    f"/versions/{VERSION}"
)


def download(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "FlyMeToTheFlies/1.0"},
    )
    with urllib.request.urlopen(request) as response, destination.open("wb") as dst:
        shutil.copyfileobj(response, dst)


def main() -> None:
    if TARGET.exists() and WING_TARGET.exists():
        print(f"Already present: {TARGET}")
        print(f"Already present: {WING_TARGET}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Looking up the FlyBody flight dataset archive...")
    request = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "FlyMeToTheFlies/1.0"},
    )

    with urllib.request.urlopen(request) as response:
        metadata = json.load(response)

    files = metadata.get("files", [])
    matches = [file for file in files if file.get("name") == ARCHIVE_NAME]

    if not matches:
        available = ", ".join(file.get("name", "<unnamed>") for file in files)
        raise FileNotFoundError(
            f"{ARCHIVE_NAME} was not found in Figshare version "
            f"{VERSION}. Available files: {available}"
        )

    download_url = matches[0].get("download_url")
    if not download_url:
        raise RuntimeError(f"No download URL was provided for {ARCHIVE_NAME}")

    with tempfile.TemporaryDirectory() as temp_dir:
        archive = Path(temp_dir) / ARCHIVE_NAME
        print(f"Downloading {ARCHIVE_NAME}...")
        print(download_url)
        download(download_url, archive)

        print(f"Extracting {TARGET_NAME}...")
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            targets = {
                TARGET_NAME: TARGET,
                WING_NAME: WING_TARGET,
            }
            for target_name, target_path in targets.items():
                matches = [
                    name for name in names
                    if name.endswith("/" + target_name) or name == target_name
                ]
                if not matches:
                    raise FileNotFoundError(
                        f"{target_name} was not found inside {ARCHIVE_NAME}"
                    )
                source = matches[0]
                temporary_target = target_path.with_suffix(target_path.suffix + ".part")
                try:
                    with zf.open(source) as src, temporary_target.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                    temporary_target.replace(target_path)
                except Exception:
                    temporary_target.unlink(missing_ok=True)
                    raise

    print(f"Installed: {TARGET}")
    print(f"Installed: {WING_TARGET}")


if __name__ == "__main__":
    main()
