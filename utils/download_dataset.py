
from pathlib import Path
import urllib.request
import zipfile

# Dataset configuration
DATASET_URL = "https://cs231n.stanford.edu/tiny-imagenet-200.zip"

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "tiny-imagenet"
DATASET_DIR = DATA_DIR / "tiny-imagenet-200"
ZIP_PATH = DATA_DIR / "tiny-imagenet-200.zip"


def download_dataset():

    # Check if the dataset already exists
    if (DATASET_DIR / "train").is_dir() and (DATASET_DIR / "val").is_dir():
        print("TinyImageNet is already available.")
        return

    # Create the dataset directory
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Download dataset
    print("Downloading TinyImageNet...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    # Extract dataset
    print("Extracting TinyImageNet...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)

    # Remove the ZIP file
    ZIP_PATH.unlink()

    print("TinyImageNet is ready!")
    print("Dataset location:", DATASET_DIR)


if __name__ == "__main__":
    download_dataset()
