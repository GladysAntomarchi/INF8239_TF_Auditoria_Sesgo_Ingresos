
from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = (
    "https://archive.ics.uci.edu/ml/"
    "machine-learning-databases/adult/"
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

FILES = [
    "adult.data",
    "adult.test",
    "adult.names",
]


def download_dataset():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        destination = DATA_DIR / filename

        if destination.exists():
            print(f"Ya existe: {filename}")
            continue

        print(f"Descargando: {filename}")
        urlretrieve(BASE_URL + filename, destination)

        print(f"Guardado: {destination}")

    print("Proceso de descarga finalizado.")


if __name__ == "__main__":
    download_dataset()
