# SPDX-License-Identifier: MIT
# ruff: noqa: E402
"""Training script for IMDB sentiment using HuggingFace Trainer.

Usage (local CPU):
    python -m imdb_sentiment.train --output_dir runs/first_iteration

GPU (Docker/NVIDIA runtime):
    python -m imdb_sentiment.train --output_dir /outputs --use_gpu
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from typing import Dict

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
import evaluate

LOGGER = logging.getLogger(__name__)
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)


@dataclass
class TrainConfig:
    model_name: str = "distilbert/distilbert-base-uncased"
    dataset_name: str = "imdb"
    train_size: int = 1000  # use full by default; override for smoke
    val_size: int = 100
    output_dir: str = "runs/first_iteration"
    lr: float = 2e-5
    batch_size: int = 16
    epochs: int = 2
    weight_decay: float = 0.01
    seed: int = 42
    use_gpu: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_datasets(cfg: TrainConfig) -> DatasetDict:
    LOGGER.info("Loading dataset: %s", cfg.dataset_name)
    imdb = load_dataset(cfg.dataset_name)

    # Subselect for speed if requested
    train_subset = imdb["train"].select(range(min(cfg.train_size, len(imdb["train"]))))
    val_subset = imdb["test"].select(range(min(cfg.val_size, len(imdb["test"]))))

    small = DatasetDict({"train": train_subset, "validation": val_subset, "test": imdb["test"]})
    return small


def compute_metrics_builder():
    accuracy = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return accuracy.compute(predictions=preds, references=labels)

    return compute_metrics


def train(cfg: TrainConfig) -> Dict:
    set_seed(cfg.seed)

    device = "cuda" if cfg.use_gpu and torch.cuda.is_available() else "cpu"
    LOGGER.info("Using device: %s", device)

    datasets = build_datasets(cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized = datasets.map(preprocess, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {v: k for k, v in id2label.items()}

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2, id2label=id2label, label2id=label2id
    )
    model.to(device)

    os.makedirs(cfg.output_dir, exist_ok=True)

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        report_to=[],
        learning_rate=cfg.lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=3,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_builder(),
    )

    train_result = trainer.train()
    LOGGER.info("Training complete. Metrics: %s", train_result.metrics)

    # Save final artifacts in a clean folder
    best_dir = os.path.join(cfg.output_dir, "best")
    trainer.save_model(best_dir)

    # Persist label maps for downstream inference
    with open(os.path.join(best_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump({"id2label": id2label, "label2id": label2id}, f)

    # Also push a simple eval on test for record
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])  # full test set
    with open(os.path.join(cfg.output_dir, "train_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"train": train_result.metrics, "test": test_metrics}, f, indent=2)

    return {"train": train_result.metrics, "test": test_metrics, "model_dir": best_dir}


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default=TrainConfig.model_name)
    p.add_argument("--dataset_name", default=TrainConfig.dataset_name)
    p.add_argument("--train_size", type=int, default=TrainConfig.train_size)
    p.add_argument("--val_size", type=int, default=TrainConfig.val_size)
    p.add_argument("--output_dir", default=TrainConfig.output_dir)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--batch_size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--weight_decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--seed", type=int, default=TrainConfig.seed)
    p.add_argument("--use_gpu", action="store_true")
    args = p.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    cfg = parse_args()
    LOGGER.info("Train config: %s", asdict(cfg))
    _ = train(cfg)