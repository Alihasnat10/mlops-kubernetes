import os
import pytest
from imdb_sentiment.inference import SentimentModel

# Use a tiny model to keep tests fast
TINY_MODEL = "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english"


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory):
    # HF will cache in ~/.cache; pointing to remote string is fine
    return TINY_MODEL


def test_predict_single(tiny_model_dir):
    m = SentimentModel(tiny_model_dir, device="cpu")
    out = m.predict(["I loved this movie!", "This was terrible."])
    assert len(out) == 2
    assert all("label" in r and "score" in r for r in out)