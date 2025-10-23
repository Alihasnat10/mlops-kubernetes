import shutil
from pathlib import Path

from imdb_sentiment.train import TrainConfig, train


def test_train_smoke(tmp_path):
    out = tmp_path / "runs"
    cfg = TrainConfig(
        train_size=200,
        val_size=100,
        epochs=1,
        batch_size=8,
        output_dir=str(out),
    )
    metrics = train(cfg)
    assert (out / "best").exists()
    assert "model_dir" in metrics