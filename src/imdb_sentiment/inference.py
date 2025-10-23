# SPDX-License-Identifier: MIT
"""Inference helper for loading a saved HF classifier and predicting labels."""
from __future__ import annotations
import json
import os
from typing import Iterable, List, Dict, Any

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentModel:
    def __init__(self, model_dir: str, device: str | None = None):
        self.model_dir = model_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

        # Optional: label maps
        label_map_path = os.path.join(model_dir, "label_map.json")
        if os.path.exists(label_map_path):
            with open(label_map_path, "r", encoding="utf-8") as f:
                maps = json.load(f)
            self.id2label = {int(k): v for k, v in maps.get("id2label", {}).items()}
        else:
            self.id2label = getattr(self.model.config, "id2label", {0: "NEGATIVE", 1: "POSITIVE"})

    @torch.inference_mode()
    def predict(self, texts: Iterable[str]) -> List[Dict[str, Any]]:
        enc = self.tokenizer(list(texts), return_tensors="pt", padding=True, truncation=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        preds = probs.argmax(axis=1)
        results = []
        for i, row in enumerate(probs):
            label_id = int(preds[i])
            label = self.id2label.get(label_id, str(label_id))
            results.append({
                "label": label,
                "score": float(row[label_id]),
                "all_scores": row.tolist(),
            })
        return results

# sm = SentimentModel("./runs/second_iteration/best")
# text = ["This was a bullshit."]
# print(sm.predict(text))