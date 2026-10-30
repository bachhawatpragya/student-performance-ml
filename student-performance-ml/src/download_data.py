
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile, BadZipFile
from io import BytesIO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_FILE = DATA_DIR / "student-mat.csv"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/320/"
    "student+performance.zip"
)


def extract_csv(archive_bytes):
    """Find student-mat.csv in an outer or nested ZIP archive."""

    with ZipFile(BytesIO(archive_bytes)) as outer_zip:
        # First, check whether the CSV is directly in the outer ZIP.
        for name in outer_zip.namelist():
            if Path(name).name.lower() == "student-mat.csv":
                return outer_zip.read(name)

        # Some versions contain another ZIP inside the outer ZIP.
        for name in outer_zip.namelist():
            if name.lower().endswith(".zip"):
                try:
                    nested_bytes = outer_zip.read(name)

                    with ZipFile(BytesIO(nested_bytes)) as nested_zip:
                        for nested_name in nested_zip.namelist():
                            if (
                                Path(nested_name).name.lower()
                                == "student-mat.csv"
                            ):
                                return nested_zip.read(nested_name)

                except BadZipFile:
                    continue

    raise FileNotFoundError(
        "student-mat.csv was not found in the downloaded ZIP archives."
    )


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        print(f"Dataset already exists: {DATA_FILE}")
        return DATA_FILE

    print("Downloading the UCI Student Performance dataset...")

    request = Request(
        DATASET_URL,
        headers={"User-Agent": "StudentPerformanceML/1.0"},
    )

    try:
        with urlopen(request, timeout=60) as response:
            archive_bytes = response.read()

        csv_bytes = extract_csv(archive_bytes)
        DATA_FILE.write_bytes(csv_bytes)

        print(f"Dataset downloaded successfully: {DATA_FILE}")
        print(f"File size: {DATA_FILE.stat().st_size:,} bytes")

        return DATA_FILE

    except Exception as exc:
        if DATA_FILE.exists():
            DATA_FILE.unlink()

        raise RuntimeError(
            f"Dataset download failed: {exc}"
        ) from exc


if __name__ == "__main__":
    download_dataset()
