"""Dataset access: the public teaching CSV is downloaded on first use and is not stored in git."""
from __future__ import annotations
import hashlib
from pathlib import Path
from urllib.request import urlopen

DATA_URL = ("https://raw.githubusercontent.com/stedy/"
            "Machine-Learning-with-R-datasets/master/insurance.csv")
# SHA-256 of the 1,338-row file this project was developed and tested with.
DATA_SHA256 = "505c1cbc2e63d0363bac59501563df2530aadf4cdb9cfee226f4ef32f5468281"


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ensure_dataset(path: str | Path, url: str = DATA_URL,
                   expected_sha256: str | None = DATA_SHA256) -> Path:
    """Return the local CSV path, downloading it first if it does not exist.

    The file is checked against a pinned SHA-256, so a silently changed upstream file
    cannot produce different results without anyone noticing.
    """
    path = Path(path)
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urlopen(url, timeout=30) as response:
                content = response.read()
        except Exception as exc:  # network down, URL moved, ...
            raise FileNotFoundError(
                f"Could not download the dataset from {url} ({exc}). "
                f"Download it manually and save it as {path} (see data/README.md)."
            ) from exc
        path.write_bytes(content)
    if expected_sha256 and sha256_of(path) != expected_sha256:
        raise ValueError(f"{path} does not match the expected SHA-256 checksum: "
                         "the dataset changed or the file is damaged. Review it before use.")
    return path
