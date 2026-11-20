import pytest

from utils.osHandler import loadyaml


def test_loadyaml_reads_a_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("lr: 0.01\n")

    result = loadyaml(path)

    assert result == {"lr": 0.01}


def test_loadyaml_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.yaml"

    with pytest.raises(FileNotFoundError):
        loadyaml(missing)
