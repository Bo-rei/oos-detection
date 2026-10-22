"""
Multi-Sphere OOS Detector (Union-of-Hyperspheres)
==================================================
严格工程实现：K-means + 分位数半径 + Hard OOS 验证

基于审计结果的关键改进：
1. 半径使用分位数而非max distance（防止离群点）
2. 支持per-cluster半径调优
3. 包含Hard OOS测试接口
4. 完整的Val→Test验证流程

References:
- K-means baseline验证（F1=0.8458 on Test）
- 风险点：均值层面分离 ≠ 支持集线性可分
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import logging

logger = logging.getLogger(__name__)


@dataclass
class SphereConfig:
    """单个超球配置"""
    center: np.ndarray
    radius: float
    cluster_id: int
    intent_name: Optional[str] = None
    inv_diag_cov: Optional[np.ndarray] = None


@dataclass
class DetectorMetrics:
    """检测器指标"""
    id_recall: float
    oos_rejection: float
    f1_like: float
    threshold_or_quantile: float


class MultiSphereOOSDetector:
    """
    多中心超球OOS检测器

    核心思想：
    - 用K个超球覆盖ID支持集（而非单球）
    - 每个球的半径用分位数而非max避免离群点
    - 判决：到最近球心的距离 ≤ 该球半径 → ID

    参数：
        n_clusters: 簇数（默认等于意图数，可增加以降低簇内半径）
        radius_quantile: 半径分位数（0.90-0.98），Val上调优
        radius_method: 'quantile'（推荐） or 'mean_std'
        random_state: 随机种子
    """

    def __init__(
        self,
        n_clusters: Optional[int] = None,
        radius_quantile: float = 0.95,
        radius_method: str = 'quantile',
        radius_lambda: float = 2.0,
        center_mode: str = 'kmeans',
        distance_metric: str = 'euclidean',
        margin_gamma: Optional[float] = None,
        covariance_eps: float = 1e-6,
        l2_normalize: bool = False,
        subcenters_per_intent: int = 1,
        subcenters_overrides: Optional[Dict[str, int]] = None,
        random_state: int = 42,
        acceptance_mode: str = "nearest_sphere",
    ):
        self.n_clusters = n_clusters
        self.radius_quantile = radius_quantile
        self.radius_method = radius_method
        self.radius_lambda = radius_lambda
        self.center_mode = center_mode
        self.distance_metric = distance_metric
        self.margin_gamma = margin_gamma
        self.covariance_eps = covariance_eps
        self.l2_normalize = bool(l2_normalize)
        self.subcenters_per_intent = int(max(1, subcenters_per_intent))
        self.subcenters_overrides = {
            str(k): int(max(1, v))
            for k, v in (subcenters_overrides or {}).items()
        }
        self.random_state = random_state
        if acceptance_mode not in {"nearest_sphere", "normalized_union"}:
            raise ValueError(
                "acceptance_mode must be 'nearest_sphere' or 'normalized_union': "
                f"{acceptance_mode}"
            )
        # ``nearest_sphere`` preserves the historical protocol_v2 contract.
        # ``normalized_union`` is opt-in because changing it would invalidate
        # the frozen E2/E3 artifacts.  The latter accepts a sample when any
        # sphere contains it and reports the minimum distance/radius ratio.
        self.acceptance_mode = acceptance_mode

        self.kmeans = None
        self.spheres: List[SphereConfig] = []
        self.intent_to_cluster: Dict[str, int] = {}
        self.intent_to_clusters: Dict[str, List[int]] = {}
        self.cluster_to_intent: Dict[int, str] = {}

        self._train_embeddings: Optional[np.ndarray] = None
        self._train_cluster_labels: Optional[np.ndarray] = None

        self.fitted = False

    def _normalize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply L2 normalization on last axis when enabled."""
        x = np.asarray(embeddings, dtype=np.float64)
        if not self.l2_normalize:
            return x
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return x / norms

    def fit(
        self,
        train_embeddings: np.ndarray,
        train_intents: np.ndarray
    ):
        """
        训练K-means并计算超球半径

        Args:
            train_embeddings: (N, D) 训练集编码
            train_intents: (N,) 意图标签
        """
        train_embeddings = self._normalize_embeddings(np.asarray(train_embeddings))
        train_intents = np.asarray(train_intents)

        if self.center_mode == 'class_centroid':
            unique_intents = sorted(np.unique(train_intents).tolist())
            self.n_clusters = len(unique_intents)

            logger.info(f"Training class-centroid detector with {self.n_clusters} intents...")

            self.intent_to_cluster = {intent: idx for idx, intent in enumerate(unique_intents)}
            self.cluster_to_intent = {idx: intent for intent, idx in self.intent_to_cluster.items()}
            cluster_labels = np.array([self.intent_to_cluster[x] for x in train_intents], dtype=np.int64)

            class _StaticCenters:
                def __init__(self, centers: np.ndarray, labels: np.ndarray):
                    self.cluster_centers_ = centers
                    self.labels_ = labels
                    self.n_clusters = centers.shape[0]

            centers = []
            for idx in range(self.n_clusters):
                points = train_embeddings[cluster_labels == idx]
                centers.append(points.mean(axis=0))
            centers_np = np.asarray(centers, dtype=np.float32)
            self.kmeans = _StaticCenters(centers_np, cluster_labels)
            self.intent_to_clusters = {
                intent: [idx] for intent, idx in self.intent_to_cluster.items()
            }
        elif self.center_mode == 'class_centroid_mixture':
            unique_intents = sorted(np.unique(train_intents).tolist())

            logger.info(
                "Training class-centroid-mixture detector with %d intents, %d subcenters/intent...",
                len(unique_intents),
                self.subcenters_per_intent,
            )

            global_centers: List[np.ndarray] = []
            cluster_labels = np.zeros((train_embeddings.shape[0],), dtype=np.int64)
            self.intent_to_clusters = {}
            self.cluster_to_intent = {}

            next_cluster_id = 0
            for intent in unique_intents:
                idxs = np.where(train_intents == intent)[0]
                pts = train_embeddings[idxs]

                override_k = int(self.subcenters_overrides.get(intent, self.subcenters_per_intent))
                n_sub = int(min(override_k, max(1, pts.shape[0])))
                if n_sub == 1:
                    local_labels = np.zeros((pts.shape[0],), dtype=np.int64)
                    local_centers = np.asarray([pts.mean(axis=0)], dtype=np.float64)
                else:
                    km = KMeans(
                        n_clusters=n_sub,
                        random_state=self.random_state,
                        n_init=10,
                    )
                    local_labels = km.fit_predict(pts)
                    local_centers = km.cluster_centers_

                assigned_clusters: List[int] = []
                for local_k in range(n_sub):
                    gid = next_cluster_id
                    next_cluster_id += 1
                    assigned_clusters.append(gid)
                    self.cluster_to_intent[gid] = intent
                    global_centers.append(local_centers[local_k])

                self.intent_to_clusters[intent] = assigned_clusters
                self.intent_to_cluster[intent] = assigned_clusters[0]

                mapped_global_labels = np.array(
                    [assigned_clusters[int(local)] for local in local_labels],
                    dtype=np.int64,
                )
                cluster_labels[idxs] = mapped_global_labels

            self.n_clusters = int(next_cluster_id)

            class _StaticCenters:
                def __init__(self, centers: np.ndarray, labels: np.ndarray):
                    self.cluster_centers_ = centers
                    self.labels_ = labels
                    self.n_clusters = centers.shape[0]

            centers_np = np.asarray(global_centers, dtype=np.float64)
            self.kmeans = _StaticCenters(centers_np, cluster_labels)
        elif self.center_mode == 'kmeans':
            if self.n_clusters is None:
                self.n_clusters = len(np.unique(train_intents))

            logger.info(f"Training K-means with {self.n_clusters} clusters...")

            self.kmeans = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10
            )
            cluster_labels = self.kmeans.fit_predict(train_embeddings)

            silhouette_avg = silhouette_score(train_embeddings, cluster_labels)
            logger.info(f"K-means Silhouette Score: {silhouette_avg:.4f}")

            self._build_intent_mapping(train_intents, cluster_labels)
        else:
            raise ValueError(f"Unknown center_mode: {self.center_mode}")

        self._train_embeddings = train_embeddings
        self._train_cluster_labels = cluster_labels

        self._compute_radii(train_embeddings, cluster_labels)

        self.fitted = True
        logger.info(f"Fitted {len(self.spheres)} hyperspheres")

        # 输出半径统计
        radii = [s.radius for s in self.spheres]
        logger.info(f"Radius stats: mean={np.mean(radii):.4f}, "
                   f"max={np.max(radii):.4f}, min={np.min(radii):.4f}")

    def _build_intent_mapping(self, intents: np.ndarray, cluster_labels: np.ndarray):
        """建立意图到簇的映射（多数投票）"""
        unique_intents = np.unique(intents)
        self.intent_to_clusters = {}
        for intent in unique_intents:
            mask = intents == intent
            clusters_for_intent = cluster_labels[mask]
            most_common_cluster = np.bincount(clusters_for_intent).argmax()
            self.intent_to_cluster[intent] = int(most_common_cluster)
            self.intent_to_clusters[intent] = sorted(np.unique(clusters_for_intent).tolist())
        self.cluster_to_intent = {v: k for k, v in self.intent_to_cluster.items()}

    def _compute_sphere_radii(
        self,
        embeddings: np.ndarray,
        cluster_labels: np.ndarray
    ):
        """计算每个簇的半径（分位数或均值+std）"""
        self._compute_radii(embeddings, cluster_labels)

    def _compute_radii(
        self,
        embeddings: Optional[np.ndarray] = None,
        cluster_labels: Optional[np.ndarray] = None
    ):
        """重新计算半径（不改变中心），支持统计半径与对角马氏距离。"""
        if embeddings is None:
            if self._train_embeddings is None:
                raise RuntimeError("No training embeddings cached for radius recomputation")
            embeddings = self._train_embeddings
        if cluster_labels is None:
            if self._train_cluster_labels is None:
                raise RuntimeError("No cluster labels cached for radius recomputation")
            cluster_labels = self._train_cluster_labels

        embeddings = np.asarray(embeddings)
        cluster_labels = np.asarray(cluster_labels)

        self.spheres = []

        for cluster_id in range(self.n_clusters):
            mask = cluster_labels == cluster_id
            cluster_points = embeddings[mask]

            if len(cluster_points) == 0:
                logger.warning(f"Cluster {cluster_id} is empty, skipping")
                continue

            center = self.kmeans.cluster_centers_[cluster_id]
            diff = cluster_points - center

            inv_diag_cov = None
            if self.distance_metric == 'mahalanobis_diag':
                var = np.var(diff, axis=0) + self.covariance_eps
                inv_diag_cov = 1.0 / var
                distances = np.sqrt(np.sum((diff ** 2) * inv_diag_cov, axis=1))
            elif self.distance_metric == 'euclidean':
                distances = np.linalg.norm(diff, axis=1)
            else:
                raise ValueError(f"Unknown distance_metric: {self.distance_metric}")

            # 根据方法计算半径
            if self.radius_method == 'quantile':
                radius = float(np.quantile(distances, self.radius_quantile))
            elif self.radius_method == 'mean_std':
                radius = float(distances.mean() + self.radius_lambda * distances.std())
            else:
                raise ValueError(f"Unknown radius_method: {self.radius_method}")

            self.spheres.append(SphereConfig(
                center=center,
                radius=radius,
                cluster_id=cluster_id,
                intent_name=self.cluster_to_intent.get(cluster_id),
                inv_diag_cov=inv_diag_cov
            ))

    def _distance(self, emb: np.ndarray, sphere: SphereConfig) -> float:
        diff = emb - sphere.center
        if self.distance_metric == 'mahalanobis_diag':
            if sphere.inv_diag_cov is None:
                raise RuntimeError("inv_diag_cov missing for mahalanobis_diag mode")
            return float(np.sqrt(np.sum((diff ** 2) * sphere.inv_diag_cov)))
        return float(np.linalg.norm(diff))

    def _nearest_sphere_stats(self, emb: np.ndarray) -> Dict[str, float]:
        """Return score statistics under the explicitly selected acceptance contract."""
        distances_to_centers = [
            self._distance(emb, sphere)
            for sphere in self.spheres
        ]
        nearest_idx = int(np.argmin(distances_to_centers))
        nearest_distance = float(distances_to_centers[nearest_idx])
        nearest_radius = float(self.spheres[nearest_idx].radius)
        ratios = np.asarray(
            [distance / max(float(sphere.radius), 1e-12) for distance, sphere in zip(distances_to_centers, self.spheres)],
            dtype=np.float64,
        )
        if self.acceptance_mode == "normalized_union":
            selected_idx = int(np.argmin(ratios))
            score = float(ratios[selected_idx])
            selected_distance = float(distances_to_centers[selected_idx])
            selected_radius = float(self.spheres[selected_idx].radius)
        else:
            selected_idx = nearest_idx
            score = float(nearest_distance / max(nearest_radius, 1e-12))
            selected_distance = nearest_distance
            selected_radius = nearest_radius

        margin_ok = True
        if self.margin_gamma is not None and len(distances_to_centers) > 1:
            sorted_distances = np.partition(np.asarray(distances_to_centers), 1)
            d1 = float(sorted_distances[0])
            d2 = float(sorted_distances[1])
            margin_ok = d1 < float(self.margin_gamma) * d2

        if self.acceptance_mode == "normalized_union":
            # The acceptance region is the union of all fitted spheres.  A
            # raw-distance nearest sphere is not equivalent when radii differ.
            is_id = bool(np.any(ratios <= 1.0)) and margin_ok
        else:
            is_id = (nearest_distance <= nearest_radius) and margin_ok
        return {
            "nearest_idx": selected_idx,
            "distance": selected_distance,
            "radius": selected_radius,
            "score": float(score),
            "margin_ok": bool(margin_ok),
            "is_id": bool(is_id),
            "accepted_sphere_count": int(np.sum(ratios <= 1.0)),
        }

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        """
        预测：0=ID, 1=OOS

        判决规则：到最近球心的距离 ≤ 该球半径 → ID
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        embeddings = self._normalize_embeddings(np.asarray(embeddings))
        predictions = []

        for emb in embeddings:
            stats = self._nearest_sphere_stats(emb)
            predictions.append(0 if stats["is_id"] else 1)

        return np.array(predictions)

    def predict_with_scores(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict OOS labels and expose continuous gate scores.

        Returns:
            dict with keys:
            - pred: int array (0=ID, 1=OOS)
            - score: float array, higher means more OOS-like
            - nearest_cluster: int array (the raw-nearest or ratio-selected sphere)
            - distance: float array
            - radius: float array
            - margin_ok: bool array
        """
        if not self.fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        embeddings = self._normalize_embeddings(np.asarray(embeddings))
        preds = []
        scores = []
        nearest_clusters = []
        distances = []
        radii = []
        margin_flags = []
        accepted_sphere_counts = []

        for emb in embeddings:
            stats = self._nearest_sphere_stats(emb)
            preds.append(0 if stats["is_id"] else 1)
            scores.append(float(stats["score"]))
            nearest_clusters.append(int(stats["nearest_idx"]))
            distances.append(float(stats["distance"]))
            radii.append(float(stats["radius"]))
            margin_flags.append(bool(stats["margin_ok"]))
            accepted_sphere_counts.append(int(stats["accepted_sphere_count"]))

        return {
            "pred": np.asarray(preds, dtype=np.int64),
            "score": np.asarray(scores, dtype=np.float64),
            "nearest_cluster": np.asarray(nearest_clusters, dtype=np.int64),
            "distance": np.asarray(distances, dtype=np.float64),
            "radius": np.asarray(radii, dtype=np.float64),
            "margin_ok": np.asarray(margin_flags, dtype=np.bool_),
            "accepted_sphere_count": np.asarray(accepted_sphere_counts, dtype=np.int64),
        }

    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> DetectorMetrics:
        """评估性能"""
        predictions = self.predict(embeddings)

        id_mask = labels == 0
        oos_mask = labels == 1

        id_recall = np.sum((predictions == 0) & id_mask) / np.sum(id_mask)
        oos_rejection = np.sum((predictions == 1) & oos_mask) / np.sum(oos_mask)
        f1_like = 2 * (id_recall * oos_rejection) / (id_recall + oos_rejection + 1e-10)

        return DetectorMetrics(
            id_recall=float(id_recall),
            oos_rejection=float(oos_rejection),
            f1_like=float(f1_like),
            threshold_or_quantile=self.radius_quantile
        )

    def tune_radius_quantile(
        self,
        val_embeddings: np.ndarray,
        val_labels: np.ndarray,
        quantile_range: Tuple[float, float] = (0.85, 0.99),
        n_trials: int = 15
    ) -> float:
        """
        在Val集上调优radius_quantile

        Args:
            val_embeddings: 验证集编码
            val_labels: 验证集标签
            quantile_range: 搜索范围
            n_trials: 搜索次数

        Returns:
            最优quantile值
        """
        if not self.fitted:
            raise RuntimeError("Must fit() before tuning")

        quantiles = np.linspace(quantile_range[0], quantile_range[1], n_trials)
        best_f1 = 0
        best_quantile = self.radius_quantile

        # 保存原始半径用于恢复
        original_radii = [s.radius for s in self.spheres]
        original_quantile = self.radius_quantile

        logger.info(f"Tuning radius_quantile on Val set ({n_trials} trials)...")

        for q in quantiles:
            # 临时更新半径
            self.radius_quantile = q
            self._compute_radii()

            # 评估
            metrics = self.evaluate(val_embeddings, val_labels)

            if metrics.f1_like > best_f1:
                best_f1 = metrics.f1_like
                best_quantile = q

            if abs(q - quantile_range[0]) < 1e-6 or abs(q - quantile_range[1]) < 1e-6:
                logger.info(f"  q={q:.3f}: F1={metrics.f1_like:.4f}")

        logger.info(f"✓ Best quantile: {best_quantile:.3f} (F1={best_f1:.4f})")

        # 恢复原始设置（如果需要）
        self.radius_quantile = original_quantile
        self._compute_radii()
        for idx, radius in enumerate(original_radii):
            if idx < len(self.spheres):
                self.spheres[idx].radius = radius

        return best_quantile

    def diagnose_failures(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        texts: Optional[List[str]] = None
    ) -> Dict:
        """
        诊断失败样本

        返回：
        - false_negatives: ID误判为OOS的样本索引
        - false_positives: OOS误判为ID的样本索引
        - 每类的距离统计
        """
        predictions = self.predict(embeddings)

        false_negatives = np.where((labels == 0) & (predictions == 1))[0]
        false_positives = np.where((labels == 1) & (predictions == 0))[0]

        diagnosis = {
            'false_negatives': {
                'count': len(false_negatives),
                'indices': false_negatives.tolist(),
                'texts': [texts[i] for i in false_negatives] if texts else None
            },
            'false_positives': {
                'count': len(false_positives),
                'indices': false_positives.tolist(),
                'texts': [texts[i] for i in false_positives] if texts else None
            }
        }

        return diagnosis

    def save(self, path: Path):
        """保存检测器"""
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted detector")

        state = {
            'n_clusters': self.n_clusters,
            'radius_quantile': self.radius_quantile,
            'radius_method': self.radius_method,
            'radius_lambda': self.radius_lambda,
            'center_mode': self.center_mode,
            'distance_metric': self.distance_metric,
            'margin_gamma': self.margin_gamma,
            'covariance_eps': self.covariance_eps,
            'l2_normalize': self.l2_normalize,
            'subcenters_per_intent': self.subcenters_per_intent,
            'subcenters_overrides': self.subcenters_overrides,
            'random_state': self.random_state,
            'acceptance_mode': self.acceptance_mode,
            'spheres': [
                {
                    'center': s.center.tolist(),
                    'radius': s.radius,
                    'cluster_id': s.cluster_id,
                    'intent_name': s.intent_name,
                    'inv_diag_cov': None if s.inv_diag_cov is None else s.inv_diag_cov.tolist()
                }
                for s in self.spheres
            ],
            'intent_to_cluster': self.intent_to_cluster,
            'intent_to_clusters': self.intent_to_clusters,
            'cluster_to_intent': self.cluster_to_intent
        }

        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

        logger.info(f"Detector saved to {path}")

    def load(self, path: Path):
        """加载检测器"""
        with open(path, 'r') as f:
            state = json.load(f)

        self.n_clusters = state['n_clusters']
        self.radius_quantile = state['radius_quantile']
        self.radius_method = state['radius_method']
        self.radius_lambda = state.get('radius_lambda', self.radius_lambda)
        self.center_mode = state.get('center_mode', self.center_mode)
        self.distance_metric = state.get('distance_metric', self.distance_metric)
        self.margin_gamma = state.get('margin_gamma', self.margin_gamma)
        self.covariance_eps = state.get('covariance_eps', self.covariance_eps)
        self.l2_normalize = bool(state.get('l2_normalize', self.l2_normalize))
        self.subcenters_per_intent = int(
            state.get('subcenters_per_intent', self.subcenters_per_intent)
        )
        self.subcenters_overrides = {
            str(k): int(max(1, v))
            for k, v in state.get('subcenters_overrides', self.subcenters_overrides).items()
        }
        self.random_state = state['random_state']
        self.acceptance_mode = state.get('acceptance_mode', 'nearest_sphere')
        if self.acceptance_mode not in {'nearest_sphere', 'normalized_union'}:
            raise ValueError(f"Unsupported serialized acceptance_mode: {self.acceptance_mode}")
        self.intent_to_cluster = state['intent_to_cluster']
        self.intent_to_clusters = state.get(
            'intent_to_clusters',
            {k: [int(v)] for k, v in self.intent_to_cluster.items()}
        )
        self.cluster_to_intent = state.get(
            'cluster_to_intent',
            {int(v): k for k, v in self.intent_to_cluster.items()}
        )

        self.spheres = [
            SphereConfig(
                center=np.array(s['center']),
                radius=s['radius'],
                cluster_id=s['cluster_id'],
                intent_name=s.get('intent_name'),
                inv_diag_cov=None if s.get('inv_diag_cov') is None else np.array(s['inv_diag_cov'])
            )
            for s in state['spheres']
        ]

        self.fitted = True
        logger.info(f"Detector loaded from {path}")
