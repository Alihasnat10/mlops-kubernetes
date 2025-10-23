from fastapi.testclient import TestClient
from src.main import app


def test_health():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200


def test_predict_endpoint(monkeypatch):
    from imdb_sentiment.inference import SentimentModel

    class Dummy(SentimentModel):
        def __init__(self):
            pass
        def predict(self, texts):
            return [{"label": "POSITIVE", "score": 0.9, "all_scores": [0.1, 0.9]} for _ in texts]

    # monkeypatch the global model loaded on startup
    import src.main as main
    main.model = Dummy()

    client = TestClient(main.app)
    r = client.post("/predict", json={"texts": ["a", "b"]})
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 2
    assert data["results"][0]["label"] == "POSITIVE"