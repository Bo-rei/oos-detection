"""Representation training exported from the current experiment runners."""
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from src.gate.representation import build_racal_model, choose_device, encode_rows, set_seed
from src.gate.boundary import fit_k1_detector, evaluate_open

def _class_centers(values: np.ndarray, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    names = sorted({str(row["intent"]) for row in rows})
    label_map = {name: index for index, name in enumerate(names)}
    labels = np.asarray([label_map[str(row["intent"])] for row in rows], dtype=np.int64)
    centers = np.asarray([values[labels == index].mean(axis=0) for index in range(len(names))], dtype=np.float32)
    centers /= np.clip(np.linalg.norm(centers, axis=1, keepdims=True), 1e-12, None)
    return centers, label_map, labels


def _center_losses(
    features: torch.Tensor,
    targets: torch.Tensor,
    centers: torch.Tensor,
    temperature: float,
    intra_weight: float,
    inter_weight: float,
    classification_weight: float,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    distances = torch.cdist(features, centers, p=2).pow(2)
    own = distances.gather(1, targets[:, None]).squeeze(1)
    masked = distances.clone()
    masked.scatter_(1, targets[:, None], float("inf"))
    other = masked.min(dim=1).values
    logits = -distances / max(float(temperature), 1e-6)
    ce = torch.nn.functional.cross_entropy(logits, targets)
    intra = own.mean()
    inter = torch.relu(own - other + float(margin)).mean()
    total = classification_weight * ce + intra_weight * intra + inter_weight * inter
    return total, {"classification_ce": float(ce.detach().cpu()), "intra_compactness": float(intra.detach().cpu()), "inter_margin": float(inter.detach().cpu()), "total": float(total.detach().cpu())}

def configure(model, recipe, warmup):
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    for p in model.projection.parameters():
        p.requires_grad_(recipe['projection_enabled'])
    # A disabled zero-initialized residual projection remains exactly identity.
    if not warmup or not recipe['projection_enabled']:
        # ``layers=0`` is the explicit projection-only control.  Python's
        # ``layer[-0:]`` would otherwise select every Transformer block.
        if recipe['layers'] == 0:
            return torch.optim.AdamW(
                [
                    dict(
                        params=[
                            p
                            for n, p in model.named_parameters()
                            if n.startswith('projection.') and p.requires_grad
                        ],
                        lr=recipe['projection_lr'],
                    )
                ]
            )
        for layer in model.encoder.encoder.layer[-recipe['layers']:]:
            for p in layer.parameters():
                p.requires_grad_(True)
    groups = []
    for prefix, lr in [('encoder.', recipe['lr']), ('projection.', recipe['projection_lr'])]:
        params = [p for n, p in model.named_parameters() if n.startswith(prefix) and p.requires_grad]
        if params:
            groups.append(dict(params=params, lr=lr))
    return torch.optim.AdamW(groups)

@dataclass(frozen=True)
class TrainConfig:
    model_path: str
    dataset: str
    data_root: str
    seed: int
    projection_hidden_dim: int = 256
    warmup_epochs: int = 1
    finetune_epochs: int = 3
    patience: int = 1
    batch_size: int = 64
    max_length: int = 256
    projection_lr: float = 2e-4
    backbone_lr: float = 2e-5
    temperature: float = 0.07
    intra_weight: float = 0.1
    inter_weight: float = 0.1
    classification_weight: float = 1.0
    inter_margin: float = 0.20
    threshold: float = 1.0
    radius_lambda: float = 1.0
    representation_mode: str = "last2_minilm_plus_projection"
    selection_metric: str = "known_only"
    device: str = "auto"


def _make_optimizer(model: torch.nn.Module, config: TrainConfig, phase: str) -> torch.optim.Optimizer:
    projection = [p for p in model.projection.parameters() if p.requires_grad]
    if phase == "warmup":
        return torch.optim.AdamW(projection, lr=config.projection_lr)
    adapted = [p for name, p in model.named_parameters() if not name.startswith("projection.") and p.requires_grad]
    return torch.optim.AdamW([{"params": projection, "lr": config.projection_lr}, {"params": adapted, "lr": config.backbone_lr}])


def _set_phase(model: torch.nn.Module, phase: str) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("projection."))
    if phase != "finetune":
        return
    if getattr(model, "mode", "") == "lora_minilm_plus_projection":
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.requires_grad_(True)
        return
    for block in model.encoder.encoder.layer[-2:]:
        for parameter in block.parameters():
            parameter.requires_grad_(True)

def run_trainable(
    config: TrainConfig,
    snapshot: Mapping[str, Any],
    views: Mapping[str, list[dict[str, Any]]],
    run_dir: Path,
    method: str = "trainable_k1",
) -> dict[str, Any]:
    started = time.time()
    set_seed(config.seed)
    device = choose_device(config.device)
    model_path = Path(config.model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = build_racal_model(model_path, config.representation_mode, config.projection_hidden_dim).to(device)
    train_rows = views["train"]
    if config.selection_metric not in {"known_only", "oos_f1"}:
        raise ValueError(f"Unsupported selection_metric: {config.selection_metric}")
    calibration_rows = (
        [row for row in views["val"] if int(row["label"]) == 0]
        if config.selection_metric == "known_only"
        else list(views["val"])
    )
    label_names = sorted({str(row["intent"]) for row in train_rows})
    label_map = {name: index for index, name in enumerate(label_names)}
    targets = torch.as_tensor([label_map[str(row["intent"])] for row in train_rows], dtype=torch.long, device=device)
    indices = np.arange(len(train_rows), dtype=np.int64)
    best_state: dict[str, Any] | None = None
    best_score = -float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    epoch_number = 0
    for phase, epochs in (("warmup", config.warmup_epochs), ("finetune", config.finetune_epochs)):
        _set_phase(model, phase)
        optimizer = _make_optimizer(model, config, phase)
        for _ in range(epochs):
            epoch_number += 1
            model.train()
            with torch.no_grad():
                refreshed = encode_rows(model, tokenizer, train_rows, device, config.batch_size, config.max_length)
                centers_np, _, _ = _class_centers(refreshed, train_rows)
            centers = torch.as_tensor(centers_np, dtype=torch.float32, device=device)
            order = np.random.default_rng(config.seed + epoch_number * 7919).permutation(indices)
            losses: list[dict[str, float]] = []
            for start in range(0, len(order), config.batch_size):
                batch_indices = order[start : start + config.batch_size]
                tokens = tokenizer(
                    [str(train_rows[int(index)]["text"]) for index in batch_indices],
                    padding=True,
                    truncation=True,
                    max_length=config.max_length,
                    return_tensors="pt",
                ).to(device)
                features = model(tokens)
                total, parts = _center_losses(
                    features,
                    targets[torch.as_tensor(batch_indices, dtype=torch.long, device=device)],
                    centers,
                    config.temperature,
                    config.intra_weight,
                    config.inter_weight,
                    config.classification_weight,
                    config.inter_margin,
                )
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(parts)
            model.eval()
            train_values = encode_rows(model, tokenizer, train_rows, device, config.batch_size, config.max_length)
            calibration_values = encode_rows(model, tokenizer, calibration_rows, device, config.batch_size, config.max_length)
            detector = fit_k1_detector(
                train_values, train_rows, "mahalanobis_diag", radius_lambda=config.radius_lambda
            )
            calibration_metrics, _ = evaluate_open(detector, calibration_values, calibration_rows, config.threshold)
            if config.selection_metric == "oos_f1":
                score = float(calibration_metrics["f1_u"] + 0.05 * calibration_metrics["known_recall"])
            else:
                score = float(calibration_metrics["f1_k"] + 0.05 * calibration_metrics["known_recall"])
            history.append({
                "epoch": epoch_number,
                "phase": phase,
                "loss": float(np.mean([item["total"] for item in losses])),
                "selection_score": score,
                "calibration_known_f1": float(calibration_metrics["f1_k"]),
                "calibration_known_recall": float(calibration_metrics["known_recall"]),
                "calibration_oos_f1": float(calibration_metrics["f1_u"]),
            })
            if score > best_score + 1e-12:
                best_score, best_epoch, stale = score, epoch_number, 0
                best_state = {
                    "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
                    "mode": model.mode,
                    "label_map": label_map,
                    "epoch": epoch_number,
                    "freeze_report": model.freeze_report(),
                }
            else:
                stale += 1
            if phase == "finetune" and stale > config.patience:
                break
        del optimizer
        if phase == "finetune" and stale > config.patience:
            break
    if best_state is None:
        raise RuntimeError(f"No trainable checkpoint produced for {config.dataset}")
    model.load_state_dict(best_state["model"])
    model.eval()
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, run_dir / "checkpoint.pt")
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2) + "\n")
    return {"checkpoint": str(run_dir / "checkpoint.pt"), "epoch": best_epoch,
            "selection_score": best_score, "test_read": False}
