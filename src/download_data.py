
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
    """Descarga los archivos originales con controles básicos de integridad."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in FILES:
        destination = DATA_DIR / filename
        temporary = DATA_DIR / f"{filename}.part"

        # Reutilizar únicamente archivos existentes con contenido.
        if destination.is_file() and destination.stat().st_size > 0:
            print(f"Ya existe y no está vacío: {filename}")
            continue

        print(f"Descargando: {filename}")

        try:
            # Descargar primero a una ubicación temporal.
            urlretrieve(BASE_URL + filename, temporary)

            # Rechazar descargas vacías.
            if not temporary.is_file() or temporary.stat().st_size == 0:
                raise ValueError(
                    f"Archivo descargado vacío: {filename}"
                )

            # Sustituir el destino solo tras verificar la descarga.
            temporary.replace(destination)

            print(
                f"Guardado: {destination} "
                f"({destination.stat().st_size} bytes)"
            )

        except Exception:
            # Eliminar descargas temporales incompletas.
            temporary.unlink(missing_ok=True)
            raise

    print("Proceso de descarga finalizado.")


if __name__ == "__main__":
    download_dataset()
