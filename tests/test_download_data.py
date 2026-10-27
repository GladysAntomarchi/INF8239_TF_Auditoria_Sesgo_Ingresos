
from pathlib import Path
import sys

sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[1] / "src"),
)

import download_data


def test_download_replaces_empty_file(tmp_path, monkeypatch):
    """Un archivo vacío debe ser sustituido por una descarga válida."""

    monkeypatch.setattr(download_data, "DATA_DIR", tmp_path)
    monkeypatch.setattr(download_data, "FILES", ["adult.data"])

    destination = tmp_path / "adult.data"
    destination.write_bytes(b"")

    def fake_urlretrieve(url, filename):
        Path(filename).write_bytes(b"contenido de prueba")
        return str(filename), None

    monkeypatch.setattr(
        download_data,
        "urlretrieve",
        fake_urlretrieve,
    )

    download_data.download_dataset()

    assert destination.read_bytes() == b"contenido de prueba"
    assert not (tmp_path / "adult.data.part").exists()
