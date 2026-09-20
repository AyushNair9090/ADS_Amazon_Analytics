from pathlib import Path

POINTERS = [
    "amazon_transactions_clean.csv.dvc",
    "merged_amazon_dataset_1.csv.dvc",
    "models/best_model.pkl.dvc",
]


def test_dvc_pointers_exist_and_valid():
    for p in POINTERS:
        f = Path(p)
        assert f.exists(), f"{p} missing"
        text = f.read_text()
        assert "outs:" in text and "md5:" in text, f"{p} malformed"


def test_docker_files_exist():
    assert Path("Dockerfile").exists()
    assert Path("requirements.txt").exists()
