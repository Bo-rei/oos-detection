"""Standard Gate metrics: OOS is positive and larger score means more OOS-like."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score, average_precision_score, precision_recall_fscore_support, roc_auc_score, roc_curve


def compute_binary_oos_metrics(
    y_true_binary: Sequence[int] | np.ndarray,
    oos_scores: Sequence[float] | np.ndarray,
    threshold: float = 1.0,
) -> dict[str, float]:
    labels = np.asarray(y_true_binary, dtype=np.int64).reshape(-1)
    scores = np.asarray(oos_scores, dtype=np.float64).reshape(-1)
    if labels.size != scores.size or not np.isin(labels, (0, 1)).all():
        raise ValueError("OOS labels must be aligned binary values: Known=0, OOS=1")
    if not np.isfinite(scores).all():
        raise ValueError("OOS scores must be finite")
    predicted = (scores > threshold).astype(np.int64)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, labels=[1], average=None, zero_division=0)
    known, oos = labels == 0, labels == 1
    if np.unique(labels).size == 2:
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        valid_fpr = fpr[tpr >= 0.95]
        auroc = float(roc_auc_score(labels, scores))
        aupr = float(average_precision_score(labels, scores))
        fpr95 = float(valid_fpr.min()) if valid_fpr.size else math.nan
    else:
        auroc = aupr = fpr95 = math.nan
    false_accept = float(np.mean(predicted[oos] == 0)) if oos.any() else math.nan
    false_reject = float(np.mean(predicted[known] == 1)) if known.any() else math.nan
    return {
        "oos_precision": float(precision[0]),
        "oos_recall": float(recall[0]),
        "oos_f1": float(f1[0]),
        "id_recall": 1.0 - false_reject if known.any() else math.nan,
        "oos_rejection": float(recall[0]),
        "auroc": auroc,
        "aupr_oos": aupr,
        "fpr95": fpr95,
        "false_accept_rate": false_accept,
        "false_reject_rate": false_reject,
    }


STAGES = ('correct_oos_rejection', 'oos_accepted_by_gate', 'known_rejected_by_gate', 'known_wrong_domain', 'known_wrong_expert', 'correct_known_prediction')


def _safe_float(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def _error_stage(row: Mapping[str, Any], prediction: Mapping[str, Any]) -> str:
    true_oos = int(row["label"]) == 1
    predicted_oos = bool(prediction.get("is_oos", prediction.get("gate_pred", 1)))
    if true_oos:
        return "correct_oos_rejection" if predicted_oos else "oos_accepted_by_gate"
    if predicted_oos:
        return "known_rejected_by_gate"
    if str(prediction.get("domain", "")) != str(row.get("domain", "")):
        return "known_wrong_domain"
    if str(prediction.get("intent", "__missing__")) != str(row.get("intent", "")):
        return "known_wrong_expert"
    return "correct_known_prediction"


def compute_metrics(rows: Sequence[Mapping[str, Any]], predictions: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, int]]:
    if len(rows) != len(predictions):
        raise ValueError("row/prediction length mismatch")
    true_oos = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    pred_oos = np.asarray([int(bool(prediction.get("is_oos", prediction.get("gate_pred", 1)))) for prediction in predictions], dtype=np.int64)
    known_mask = true_oos == 0
    oos_mask = true_oos == 1
    known_intents = sorted({str(row["intent"]) for row in rows if int(row["label"]) == 0})
    gold_labels = ["__oos__" if label else str(row["intent"]) for row, label in zip(rows, true_oos, strict=True)]
    predicted_labels = ["__oos__" if pred else str(prediction.get("intent", "__missing__")) for pred, prediction in zip(pred_oos, predictions, strict=True)]
    all_labels = [*known_intents, "__oos__"]
    gate_scores = np.asarray([float(prediction.get("gate_score", 0.0)) for prediction in predictions], dtype=np.float64)
    stage_counts = {stage: 0 for stage in STAGES}
    for row, prediction in zip(rows, predictions, strict=True):
        stage_counts[_error_stage(row, prediction)] += 1

    known_pass = known_mask & (pred_oos == 0)
    router_errors = sum(
        int(known_pass[index]) and str(predictions[index].get("domain", "")) != str(rows[index].get("domain", ""))
        for index in range(len(rows))
    )
    expert_errors = stage_counts["known_wrong_expert"]
    metrics = {
        "oos_f1": float(f1_score(true_oos, pred_oos, pos_label=1, zero_division=0)),
        "f1_all": float(f1_score(gold_labels, predicted_labels, labels=all_labels, average="macro", zero_division=0)),
        "known_macro_f1": float(f1_score([label for label, mask in zip(gold_labels, known_mask, strict=True) if mask], [label for label, mask in zip(predicted_labels, known_mask, strict=True) if mask], labels=known_intents, average="macro", zero_division=0)),
        "overall_accuracy": float(np.mean(np.asarray(gold_labels, dtype=object) == np.asarray(predicted_labels, dtype=object))),
        "known_recall": float(np.mean(pred_oos[known_mask] == 0)),
        "false_accept_rate": float(np.mean(pred_oos[oos_mask] == 0)),
        "false_reject_rate": float(np.mean(pred_oos[known_mask] == 1)),
        "auroc": float(roc_auc_score(true_oos, gate_scores)),
        "aupr_oos": float(average_precision_score(true_oos, gate_scores)),
        "router_error_rate": float(router_errors / max(int(np.sum(known_pass)), 1)),
        "expert_error_rate": float(expert_errors / max(int(np.sum(known_pass)), 1)),
        "gate_id_recall": float(np.mean(pred_oos[known_mask] == 0)),
        "gate_oos_rejection": float(np.mean(pred_oos[oos_mask] == 1)),
        "known_count": int(np.sum(known_mask)),
        "oos_count": int(np.sum(oos_mask)),
        "gate_accept_known_count": int(np.sum(known_pass)),
    }
    return metrics, stage_counts
