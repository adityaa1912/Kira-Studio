import os
import zipfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = Path(os.getenv("ANIME_DATASET_DIR", PROJECT_ROOT / "backend" / "data" / "images"))
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "archive (4).zip"
ARCHIVE_PATH = Path(os.getenv("ANIME_DATASET_ARCHIVE", DEFAULT_ARCHIVE))


def dataset_status():
    if DATASET_DIR.is_dir():
        image_count = sum(1 for item in DATASET_DIR.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        return {
            "available": image_count > 0,
            "image_count": image_count,
            "source": "extracted",
            "path": str(DATASET_DIR),
        }

    if ARCHIVE_PATH.is_file():
        try:
            with zipfile.ZipFile(ARCHIVE_PATH) as archive:
                image_count = sum(
                    1
                    for item in archive.infolist()
                    if Path(item.filename).suffix.lower() in IMAGE_SUFFIXES
                )
            return {
                "available": image_count > 0,
                "image_count": image_count,
                "source": "archive",
                "path": str(ARCHIVE_PATH),
            }
        except (OSError, zipfile.BadZipFile):
            pass

    return {"available": False, "image_count": 0, "source": "missing", "path": str(ARCHIVE_PATH)}