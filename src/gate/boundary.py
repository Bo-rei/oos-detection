"""Current boundary scoring and Known-only calibration helpers."""
from __future__ import annotations
from typing import Any, Mapping, Sequence
import numpy as np
from sklearn.metrics import f1_score
from src.gate.multi_sphere_oos_detector import MultiSphereOOSDetector
from src.evaluation.metrics import compute_binary_oos_metrics

def fit_k1_detector(
    train: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    distance: str,
    radius_lambda: float = 1.0,
) -> MultiSphereOOSDetector:
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=1,
        radius_method="mean_std",
        radius_lambda=float(radius_lambda),
        distance_metric=distance,
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=42,
        acceptance_mode="nearest_sphere",
    )
    detector.fit(np.asarray(train), np.asarray([str(row["intent"]) for row in rows], dtype=object))
    return detector


def _open_label(detector: MultiSphereOOSDetector, output: Mapping[str, Any], index: int) -> str:
    if int(output["pred"][index]) == 1:
        return "__oos__"
    cluster = int(output["nearest_cluster"][index])
    return str(detector.cluster_to_intent.get(cluster, "__unknown__"))


def evaluate_open(
    detector: MultiSphereOOSDetector,
    embeddings: np.ndarray,
    rows: Sequence[Mapping[str, Any]],
    threshold: float = 1.0,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    output = detector.predict_with_scores(np.asarray(embeddings))
    labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
    metrics = compute_binary_oos_metrics(labels, output["score"], threshold)
    # predict_with_scores uses the detector's serialized threshold. The
    # evaluator also accepts an explicit threshold, so use that same decision
    # for the multiclass/open-set fields below; otherwise f1_u/f1_k silently
    # remain tied to the detector default when threshold != 1.
    thresholded_output = dict(output)
    thresholded_output["pred"] = (np.asarray(output["score"]) > float(threshold)).astype(np.int64)
    known_intents = sorted({str(row["intent"]) for row in rows if int(row["label"]) == 0})
    truth = [str(row["intent"]) if int(row["label"]) == 0 else "__oos__" for row in rows]
    predicted = [_open_label(detector, thresholded_output, i) for i in range(len(rows))]
    all_labels = [*known_intents, "__oos__"]
    metrics.update({
        "f1_all": float(f1_score(truth, predicted, labels=all_labels, average="macro", zero_division=0)),
        "f1_u": float(f1_score(truth, predicted, labels=["__oos__"], average="macro", zero_division=0)),
        "f1_k": float(f1_score(truth, predicted, labels=known_intents, average="macro", zero_division=0)),
        "accuracy": float(np.mean(np.asarray(truth, dtype=object) == np.asarray(predicted, dtype=object))),
        "known_recall": float(metrics["id_recall"]),
    })
    predictions = []
    for index, row in enumerate(rows):
        predictions.append({
            "sample_id": row["sample_id"],
            "gold_intent": row["intent"],
            "gold_is_oos": int(row["label"]),
            "predicted_is_oos": int(thresholded_output["pred"][index]),
            "predicted_intent": predicted[index],
            "nearest_cluster": int(output["nearest_cluster"][index]),
            "distance": float(output["distance"][index]),
            "radius": float(output["radius"][index]),
            "oos_score": float(output["score"][index]),
            "oos_source": row.get("oos_source"),
        })
    return metrics, predictions

def _distance_matrix(
    detector: Any,
    embeddings: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if detector.l2_normalize:
        values = values / np.clip(np.linalg.norm(values, axis=1, keepdims=True), 1e-12, None)
    centers = np.asarray([sphere.center for sphere in detector.spheres], dtype=np.float64)
    if detector.distance_metric == "mahalanobis_diag":
        inverse = np.asarray([sphere.inv_diag_cov for sphere in detector.spheres], dtype=np.float64)
    else:
        inverse = None
    chunks: list[np.ndarray] = []
    for start in range(0, len(values), chunk_size):
        batch = values[start : start + chunk_size]
        diff = batch[:, None, :] - centers[None, :, :]
        if inverse is None:
            distances = np.linalg.norm(diff, axis=2)
        else:
            distances = np.sqrt(np.sum(np.square(diff) * inverse[None, :, :], axis=2))
        chunks.append(distances)
    return np.concatenate(chunks, axis=0)

def _score_from_distances(distances, detector, radius, owners, rule, fusion, weight):
    nearest = np.argmin(distances, axis=1)
    first = distances[np.arange(len(distances)), nearest]
    second = np.min(
        np.where(owners[None, :] != owners[nearest, None], distances, np.inf),
        axis=1,
    )
    normalized = distances / np.maximum(radius[None, :], 1e-12)
    ids = nearest if rule == "nearest_sphere" else np.argmin(normalized, axis=1)
    base = normalized[np.arange(len(distances)), ids]
    ratio = first / np.maximum(second, 1e-12)
    inverse_margin = first / np.maximum(second - first, 1e-12)
    if fusion == "none":
        score = base
    elif fusion == "ratio":
        score = base + weight * ratio
    elif fusion == "inverse_margin":
        score = base + weight * inverse_margin
    elif fusion == "max_ratio":
        score = np.maximum(base, weight * ratio)
    else:
        raise ValueError(fusion)
    return score, ids

def _replay_locked(
    train_values: np.ndarray,
    train_rows: Sequence[Mapping[str, Any]],
    values: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[MultiSphereOOSDetector, dict[str, np.ndarray]]:
    """Reconstruct the exact Known-only search score family, including fusion."""
    detector = MultiSphereOOSDetector(
        center_mode="class_centroid_mixture",
        subcenters_per_intent=int(config["k"]),
        radius_method="mean_std",
        radius_lambda=1.0,
        distance_metric=str(config["distance"]),
        covariance_eps=1e-6,
        l2_normalize=True,
        random_state=42,
        acceptance_mode="nearest_sphere",
    )
    train_intents = np.asarray([str(row["intent"]) for row in train_rows], dtype=object)
    detector.fit(train_values, train_intents)
    train_distances = _distance_matrix(detector, train_values)
    distances = _distance_matrix(detector, values)
    owners = np.asarray([detector.cluster_to_intent[i] for i in range(len(detector.spheres))])
    boundary = str(config["boundary"])
    parameter = float(boundary.rsplit("_", 1)[1])
    if boundary.startswith("mean_std_"):
        detector.radius_lambda = parameter
        detector._compute_radii()
        radius = np.asarray([sphere.radius for sphere in detector.spheres], dtype=np.float64)
    elif boundary.startswith("center_quantile_"):
        assignment = np.argmin(
            np.where(train_intents[:, None] == owners[None, :], train_distances, np.inf),
            axis=1,
        )
        detector.radius_lambda = 3.0
        detector._compute_radii()
        radius = np.asarray([
            np.quantile(train_distances[assignment == index, index], parameter)
            if np.any(assignment == index) else detector.spheres[index].radius
            for index in range(len(owners))
        ])
    elif boundary.startswith("intent_quantile_"):
        radius = np.asarray([
            np.quantile(train_distances[train_intents == name].min(axis=1), parameter)
            for name in owners
        ])
    else:
        raise ValueError(f"unsupported boundary: {boundary}")
    score, ids = _score_from_distances(
        distances,
        detector,
        radius,
        owners,
        str(config["rule"]),
        str(config.get("fusion", "none")),
        float(config.get("fusion_weight", 0.0)),
    )
    return detector, {"score": np.asarray(score), "nearest_cluster": np.asarray(ids)}
