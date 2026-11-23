import pytest

from src.data import ensure_dataset, sha256_of


def test_existing_file_is_returned_without_download(tmp_path):
    csv = tmp_path / "insurance.csv"
    csv.write_text("age,sex\n30,male\n")
    assert ensure_dataset(csv, url="file:///does/not/exist", expected_sha256=None) == csv


def test_download_writes_file_and_checks_checksum(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("age,sex\n30,male\n41,female\n")
    target = tmp_path / "data" / "out.csv"
    path = ensure_dataset(target, url=source.as_uri(), expected_sha256=sha256_of(source))
    assert path.is_file() and path.read_text() == source.read_text()


def test_wrong_checksum_is_rejected(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("age,sex\n30,male\n")
    with pytest.raises(ValueError, match="checksum"):
        ensure_dataset(tmp_path / "out.csv", url=source.as_uri(), expected_sha256="0" * 64)


def test_failed_download_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download it manually"):
        ensure_dataset(tmp_path / "x.csv", url=(tmp_path / "missing.csv").as_uri())


def test_exact_duplicate_removed_before_split():
    import pandas as pd
    from src.pipeline import clean_training_data
    row = dict(age=28,sex='female',bmi=24.2,children=0,smoker='no',region='northwest',charges=1200.0)
    rows = [row, row.copy(), {**row, 'charges':1350.0}]
    cleaned, duplicate_count = clean_training_data(pd.DataFrame(rows))
    assert duplicate_count == 1
    assert len(cleaned) == 2
    assert not cleaned.duplicated().any()
    assert cleaned.index.tolist() == [0,1]


def test_real_dataset_leakage_prevention_when_available():
    import pandas as pd
    """Optional local integration check: original CSV is not included in GitHub."""
    from pathlib import Path
    from src.pipeline import clean_training_data
    original = Path(__file__).resolve().parents[1] / 'data' / 'insurance.csv'
    if not original.exists():
        return
    cleaned, dup = clean_training_data(pd.read_csv(original))
    assert len(cleaned) == 1337
    assert dup == 1
    assert not cleaned.duplicated().any()
