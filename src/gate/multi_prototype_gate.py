"""Multi-prototype gate for known intent boundary modeling.

Supports per-intent variable number of prototypes via K-means clustering.
Provides uncertainty estimation via two-threshold scoring interval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class MultiPrototypeGateConfig:
    """Configuration for MultiPrototypeGate."""

    k_default: int = 1
    k_overrides: Dict[str, int] = field(default_factory=dict)
    random_state: int = 42
    n_init: int = 10
    l2_normalize: bool = True
    score_top_k: int = 2
    score_mode: str = "top2_margin_conf"


class MultiPrototypeGate:
    """Multi-prototype gate using per-intent K-means cluster centers.

    Each known intent is represented by multiple prototypes.
    Gate score is the maximum cosine similarity to any prototype across
    all known intents.

    Args:
        config: Gate configuration controlling k and normalization.
    """

    def __init__(self, config: Optional[MultiPrototypeGateConfig] = None) -> None:
        self.config = config or MultiPrototypeGateConfig()
        self.prototypes: Dict[str, np.ndarray] = {}
        self.intent_labels: List[str] = []
        self._is_fitted: bool = False

    def _aggregate_intent_similarity(self, sims: np.ndarray) -> float:
        mode = str(self.config.score_mode)
        if mode == "max":
            return float(np.max(sims))

        top_k = max(1, int(self.config.score_top_k))
        local_k = min(top_k, int(sims.shape[0]))
        if local_k == sims.shape[0]:
            return float(np.mean(sims))
        top_vals = np.partition(sims, -local_k)[-local_k:]
        return float(np.mean(top_vals))

    def fit(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        k: Optional[int] = None,
        k_overrides: Optional[Dict[str, int]] = None,
    ) -> Dict[str, Any]:
        """Fit multi-prototype centers from labeled embeddings.

        Args:
            embeddings: L2-normalized embedding matrix of shape (N, D).
            labels: String-valued intent labels of shape (N,).
            k: Default number of prototypes per intent. Overrides config if set.
            k_overrides: Per-intent k override map. Merged with config overrides.

        Returns:
            Dictionary with intent_count and per-intent prototype counts.
        """
        embeddings = np.asarray(embeddings, dtype=np.float32)
        labels = np.asarray(labels)

        if self.config.l2_normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-12, None)

        k_default = int(k) if k is not None else int(self.config.k_default)

        merged_overrides: Dict[str, int] = dict(self.config.k_overrides)
        if k_overrides is not None:
            merged_overrides.update(k_overrides)

        self.prototypes.clear()
        self.intent_labels = []

        unique_intents = sorted(set(str(label) for label in labels.tolist()))

        for intent in unique_intents:
            mask = np.asarray([str(label) == intent for label in labels.tolist()])
            intent_embeddings = embeddings[mask]

            if intent_embeddings.shape[0] == 0:
                continue

            k_intent = int(merged_overrides.get(intent, k_default))
            k_intent = int(min(max(1, k_intent), intent_embeddings.shape[0]))

            if k_intent == 1:
                center = np.mean(intent_embeddings, axis=0, keepdims=True).astype(np.float32)
            else:
                kmeans = KMeans(
                    n_clusters=k_intent,
                    random_state=self.config.random_state,
                    n_init=self.config.n_init,
                )
                kmeans.fit(intent_embeddings)
                center = kmeans.cluster_centers_.astype(np.float32)

            if self.config.l2_normalize:
                norm = np.linalg.norm(center, axis=1, keepdims=True)
                center = center / np.clip(norm, 1e-12, None)

            self.prototypes[intent] = center
            self.intent_labels.append(intent)

        self._is_fitted = True

        return {
            "intent_count": len(self.prototypes),
            "intent_prototype_counts": {intent: int(v.shape[0]) for intent, v in self.prototypes.items()},
        }

    def score(self, query_embedding: np.ndarray) -> Tuple[float, str]:
        """Compute the maximum prototype similarity score for a query.

        Args:
            query_embedding: L2-normalized query vector of shape (D,).

        Returns:
            Tuple of (max_similarity_score, best_intent_name).

        Raises:
            RuntimeError: If gate has not been fitted.
        """
        if not self._is_fitted or len(self.prototypes) == 0:
            raise RuntimeError("MultiPrototypeGate has not been fitted.")

        query = np.asarray(query_embedding, dtype=np.float32).flatten()
        if self.config.l2_normalize:
            norm = float(np.linalg.norm(query))
            if norm > 1e-12:
                query = query / norm

        intent_scores: List[Tuple[str, float]] = []

        for intent in self.intent_labels:
            centers = self.prototypes[intent]
            sims = np.dot(centers, query)
            agg_sim = self._aggregate_intent_similarity(sims)
            intent_scores.append((intent, agg_sim))

        intent_scores.sort(key=lambda item: item[1], reverse=True)
        best_intent, top1_score = intent_scores[0]

        if self.config.score_mode in {"top2_margin", "top2_margin_conf"} and len(intent_scores) > 1:
            top2_score = intent_scores[1][1]
            margin = float(top1_score - top2_score)
            if self.config.score_mode == "top2_margin_conf":
                return float(margin * max(top1_score, 0.0)), best_intent
            return float(margin), best_intent

        return float(top1_score), best_intent

    def predict_with_uncertainty(
        self,
        query_embedding: np.ndarray,
        tau_low: float,
        tau_high: float,
    ) -> Dict[str, Any]:
        """Score a query and classify into accept / uncertain / reject regions.

        Args:
            query_embedding: L2-normalized query vector of shape (D,).
            tau_low: Lower similarity threshold; below this is hard reject.
            tau_high: Upper similarity threshold; above this is hard accept.

        Returns:
            Dictionary with keys:
                score: Maximum prototype similarity.
                best_intent: Best matching intent.
                decision: 'accept', 'uncertain', or 'reject'.
                is_id: Boolean, True if decision is 'accept'.
                in_uncertain_region: Boolean, True if decision is 'uncertain'.
        """
        score, best_intent = self.score(query_embedding)

        if score >= tau_high:
            decision = "accept"
        elif score < tau_low:
            decision = "reject"
        else:
            decision = "uncertain"

        return {
            "score": float(score),
            "best_intent": best_intent,
            "decision": decision,
            "is_id": decision == "accept",
            "in_uncertain_region": decision == "uncertain",
        }

    def score_batch(
        self,
        embeddings: np.ndarray,
    ) -> List[Tuple[float, str]]:
        """Score a batch of query embeddings.

        Args:
            embeddings: L2-normalized query matrix of shape (N, D).

        Returns:
            List of (score, best_intent) tuples.
        """
        if not self._is_fitted or len(self.prototypes) == 0:
            raise RuntimeError("MultiPrototypeGate has not been fitted.")

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if self.config.l2_normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-12, None)

        intent_list = self.intent_labels
        all_centers = np.vstack([self.prototypes[intent] for intent in intent_list])
        center_counts = [int(self.prototypes[intent].shape[0]) for intent in intent_list]

        sim_matrix = np.dot(embeddings, all_centers.T)
        results: List[Tuple[float, str]] = []
        for i in range(embeddings.shape[0]):
            sims = sim_matrix[i]
            start = 0
            intent_scores: List[Tuple[str, float]] = []
            for intent, count in zip(intent_list, center_counts):
                intent_sims = sims[start : start + count]
                agg_sim = self._aggregate_intent_similarity(intent_sims)
                intent_scores.append((intent, agg_sim))
                start += count

            intent_scores.sort(key=lambda item: item[1], reverse=True)
            best_intent, top1_score = intent_scores[0]

            if self.config.score_mode in {"top2_margin", "top2_margin_conf"} and len(intent_scores) > 1:
                top2_score = intent_scores[1][1]
                margin = float(top1_score - top2_score)
                if self.config.score_mode == "top2_margin_conf":
                    final_score = float(margin * max(top1_score, 0.0))
                else:
                    final_score = float(margin)
            else:
                final_score = float(top1_score)

            results.append((final_score, best_intent))

        return results

    def to_dict(self) -> Dict[str, Any]:
        """Serialize gate state to a JSON-compatible dictionary."""
        return {
            "config": {
                "k_default": self.config.k_default,
                "k_overrides": self.config.k_overrides,
                "random_state": self.config.random_state,
                "n_init": self.config.n_init,
                "l2_normalize": self.config.l2_normalize,
                "score_top_k": self.config.score_top_k,
                "score_mode": self.config.score_mode,
            },
            "intent_labels": self.intent_labels,
            "prototypes": {
                intent: centers.tolist()
                for intent, centers in self.prototypes.items()
            },
            "is_fitted": self._is_fitted,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiPrototypeGate":
        """Deserialize gate state from a dictionary.

        Args:
            data: Dictionary produced by to_dict().

        Returns:
            Restored MultiPrototypeGate instance.
        """
        cfg_data = data.get("config", {})
        config = MultiPrototypeGateConfig(
            k_default=int(cfg_data.get("k_default", 1)),
            k_overrides={str(k): int(v) for k, v in cfg_data.get("k_overrides", {}).items()},
            random_state=int(cfg_data.get("random_state", 42)),
            n_init=int(cfg_data.get("n_init", 10)),
            l2_normalize=bool(cfg_data.get("l2_normalize", True)),
            score_top_k=int(cfg_data.get("score_top_k", 2)),
            score_mode=str(cfg_data.get("score_mode", "top2_margin_conf")),
        )
        gate = cls(config=config)
        gate.intent_labels = [str(label) for label in data.get("intent_labels", [])]
        prototypes = {
            str(intent): np.asarray(centers, dtype=np.float32)
            for intent, centers in data.get("prototypes", {}).items()
        }
        if gate.config.l2_normalize:
            normalized: Dict[str, np.ndarray] = {}
            for intent, centers in prototypes.items():
                center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
                normalized[intent] = centers / np.clip(center_norms, 1e-12, None)
            prototypes = normalized

        gate.prototypes = prototypes
        gate._is_fitted = bool(data.get("is_fitted", len(gate.prototypes) > 0))
        return gate
