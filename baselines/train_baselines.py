#!/usr/bin/env python3
"""Run a principal baseline on the canonical S2C v19 split.

This adapter does not select known intents randomly and never reads
It feeds the v19 gate train/val/test JSON directly into the existing managers
while retaining their original BERT-based method implementations.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


METHODS = ("MSP", "OpenMax", "DOC", "DeepUnk", "KNNCL", "ADB", "DA-ADB")
BACKBONES = {
    "MSP": "bert",
    "OpenMax": "bert",
    "DOC": "bert_doc",
    "DeepUnk": "bert",
    "KNNCL": "bert_knncl",
    "ADB": "bert",
    "DA-ADB": "bert",
}
CONFIGS = {method: f"{method}.py" for method in METHODS}
CONFIGS["DA-ADB"] = "DA-ADB.py"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"empty or invalid v19 split: {path}")
    return [dict(row) for row in rows]


class V19Data:
    """Minimal DataManager-compatible object backed by v19 JSON rows."""

    def __init__(self, args: SimpleNamespace, root: Path, dataset: str, kir: float, seed: int) -> None:
        from dataloaders.bert_loader import get_loader
        from dataloaders.utils import InputExample

        split_root = root / dataset / f"kir{int(round(kir * 100)):02d}_seed{seed}" / "gate"
        train = _read_rows(split_root / "train.json")
        val = _read_rows(split_root / "val.json")
        test = _read_rows(split_root / "test.json")
        known = sorted({str(row["intent"]) for row in train if int(row["label"]) == 0})
        if not known:
            raise ValueError(f"v19 train split has no Known intents: {split_root}")
        self.data_dir = split_root
        self.known_label_list = known
        self.unseen_label = "<UNK>"
        self.unseen_label_id = len(known)
        self.label_list = known + [self.unseen_label]
        self.num_labels = len(known)

        def examples(rows: list[dict[str, Any]], mode: str) -> list[InputExample]:
            output = []
            for index, row in enumerate(rows):
                label = str(row["intent"]) if int(row["label"]) == 0 else self.unseen_label
                output.append(InputExample(f"{mode}-{index}", str(row["text"]), label=label))
            return output

        train_examples = examples(train, "train")
        val_examples = examples(val, "val")
        test_examples = examples(test, "test")
        self.dataloader = SimpleNamespace(
            train_labeled_loader=get_loader(train_examples, args, self.label_list, "train_labeled", sampler_mode="random"),
            train_unlabeled_loader=get_loader([], args, self.label_list, "train_unlabeled", sampler_mode="sequential"),
            eval_loader=get_loader(val_examples, args, self.label_list, "eval", sampler_mode="sequential"),
            test_loader=get_loader(test_examples, args, self.label_list, "test", sampler_mode="sequential"),
            num_train_examples=len(train_examples),
        )


def _args(method: str, model_root: Path, output_root: Path, seed: int, epochs: int, batch_size: int, gpu_id: str) -> SimpleNamespace:
    return SimpleNamespace(
        type="open_intent_detection", logger_name="V19Baseline", log_dir=str(output_root / "logs"),
        dataset="stackoverflow", known_cls_ratio=0.50, labeled_ratio=1.0, method=method,
        train=True, pretrain=False, save_model=False, backbone=BACKBONES[method],
        config_file_name=CONFIGS[method], seed=seed, gpu_id=gpu_id,
        pipe_results_path=str(output_root / "pipe_results"), data_dir="unused",
        output_dir=str(output_root), model_dir="models", load_pretrained_method=None,
        result_dir=str(output_root / "results"), results_file_name=f"{method}.json",
        save_results=True, loss_fct="CrossEntropyLoss", bert_model=str(model_root),
        num_train_epochs=int(epochs), train_batch_size=int(batch_size),
        eval_batch_size=int(batch_size), test_batch_size=int(batch_size),
        max_seq_length=64,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=("clinc150", "banking77", "stackoverflow"), default="stackoverflow")
    parser.add_argument("--kir", choices=(0.25, 0.50, 0.75), type=float, default=0.50)
    parser.add_argument("--seed", choices=(13, 42, 87), type=int, default=42)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-id", default="0")
    args = parser.parse_args()
    if not (args.model_root / "config.json").is_file():
        raise SystemExit(f"BERT model directory is missing config.json: {args.model_root}")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    output_root = args.output_root.resolve(); output_root.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parent / "open_intent_detection"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from backbones.base import ModelManager
    from methods import method_map

    run_args = _args(args.method, args.model_root.resolve(), output_root, args.seed, args.epochs, args.batch_size, args.gpu_id)
    run_args.dataset = args.dataset
    run_args.known_cls_ratio = 0.50
    data = V19Data(run_args, args.data_root.resolve(), args.dataset, args.kir, args.seed)
    run_args.known_label_list = data.known_label_list
    run_args.label_list = data.label_list
    run_args.unseen_label = data.unseen_label
    run_args.unseen_label_id = data.unseen_label_id
    run_args.num_labels = data.num_labels
    run_args.tot_num_labels = data.num_labels + 1
    run_args.max_seq_length = 64
    run_args.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)
    model_manager = ModelManager(run_args, data, logger_name="V19Baseline")
    model = model_manager.set_model(run_args, "bert")
    manager = method_map[args.method](run_args, data, model, logger_name="V19Baseline")
    manager.train(run_args, data)
    result = manager.test(run_args, data, show=False)
    serializable = {key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in result.items()}
    serializable.update({"protocol": "historical_v19_paper_main", "dataset": args.dataset, "kir": args.kir, "seed": args.seed, "method": args.method, "backbone": "bert", "data_root": str(args.data_root.resolve())})
    (output_root / f"{args.method}.json").write_text(json.dumps(serializable, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"method": args.method, "dataset": args.dataset, "kir": args.kir, "seed": args.seed, "output": str(output_root / f"{args.method}.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
