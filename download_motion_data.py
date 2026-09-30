from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
TARGET = DATA_DIR / "flight-dataset_saccade-evasion_augmented.hdf5"

ARTICLE_ID = 25309105
VERSION = 4
TARGET_NAME = "flight-dataset_saccade-evasion_augmented.hdf5"
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
    if TARGET.exists():
        print(f"Already present: {TARGET}")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Looking up the FlyBody dataset file...")
    print(API_URL)
    request = urllib.request.Request(
        API_URL,
        headers={"User-Agent": "FlyMeToTheFlies/1.0"},
    )

    with urllib.request.urlopen(request) as response:
        metadata = json.load(response)

    files = metadata.get("files", [])
    matches = [file for file in files if file.get("name") == TARGET_NAME]

    if not matches:
        available = ", ".join(file.get("name", "<unnamed>") for file in files)
        raise FileNotFoundError(
            f"{TARGET_NAME} was not found in Figshare version "
            f"{VERSION}. Available files: {available}"
        )

    download_url = matches[0].get("download_url")
    if not download_url:
        raise RuntimeError(f"No download URL was provided for {TARGET_NAME}")

    print("Downloading the flight dataset...")
    print(download_url)
    temporary_target = TARGET.with_suffix(TARGET.suffix + ".part")

    try:
        download(download_url, temporary_target)
        temporary_target.replace(TARGET)
    except Exception:
        temporary_target.unlink(missing_ok=True)
        raise

    print(f"Installed: {TARGET}")


if __name__ == "__main__":
    main()
