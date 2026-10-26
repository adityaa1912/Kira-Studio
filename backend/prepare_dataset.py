import argparse
import zipfile
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def extract_dataset(archive_path, destination):
    archive_path = Path(archive_path).expanduser().resolve()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    written = 0

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or Path(member.filename).suffix.lower() not in IMAGE_SUFFIXES:
                continue
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"Archive member escapes destination: {member.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                output.write(source.read())
            written += 1

    return written


def main():
    parser = argparse.ArgumentParser(description="Extract anime face images from the dataset ZIP.")
    parser.add_argument("archive", type=Path, help="Path to archive (4).zip")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        help="Extraction directory (default: backend/data)",
    )
    arguments = parser.parse_args()
    count = extract_dataset(arguments.archive, arguments.destination)
    print(f"Extracted {count:,} images into {arguments.destination.resolve()}")


if __name__ == "__main__":
    main()