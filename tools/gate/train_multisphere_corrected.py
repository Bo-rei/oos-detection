"""
Multi-Sphere OOS Detector - 修正版训练流程
========================================

**关键修正**：
1. K-means 只在 Train 上训练一次，不在 Val 上重新聚类
2. 半径分位数调优：只改变半径值，簇中心不变
3. 验证"最近簇判定"逻辑确实生效

Usage:
    python tools/gate/train_multisphere_corrected.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
import logging
import numpy as np
from typing import Dict
from sentence_transformers import SentenceTransformer

from src.gate.multi_sphere_oos_detector import MultiSphereOOSDetector, DetectorMetrics

# 日志配置
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CorrectedMultiSphereTrainer:
    """修正版多球心训练器"""

    def __init__(
        self,
        data_root: Path,
        model_path: Path,
        output_dir: Path,
        min_id_recall_constraint: float = 0.85,
        center_mode: str = "class_centroid",
        distance_metric: str = "mahalanobis_diag",
        l2_normalize: bool = False,
        subcenters_per_intent: int = 1,
    ):
        self.data_root = Path(data_root)
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.encoder = None
        self.base_detector = None  # Train上训练的检测器
        self.known_intents = None
        self.min_id_recall_constraint = float(min_id_recall_constraint)
        self.center_mode = center_mode
        self.distance_metric = distance_metric
        self.l2_normalize = bool(l2_normalize)
        self.subcenters_per_intent = int(max(1, subcenters_per_intent))

        # 保存train数据用于半径重计算
        self.train_id_embeddings = None
        self.train_id_intents = None

    def load_and_encode_data(self):
        """加载并编码所有数据"""
        logger.info("=" * 80)
        logger.info("Loading and encoding datasets...")
        logger.info("=" * 80)

        # 加载已知意图
        known_intents_file = self.data_root / "KNOWN_INTENTS.json"
        with open(known_intents_file) as f:
            self.known_intents = set(json.load(f)["known_intents"])

        # 加载编码器
        logger.info(f"Loading encoder: {self.model_path}")
        self.encoder = SentenceTransformer(str(self.model_path))

        # 加载并编码数据
        splits = {}
        for split in ["train", "val", "test"]:
            data_file = self.data_root / "gate" / f"{split}.json"
            with open(data_file) as f:
                data = json.load(f)

            texts = [x["text"] for x in data]
            intents = np.array([x["intent"] for x in data])
            labels = np.array(
                [0 if x["intent"] in self.known_intents else 1 for x in data]
            )

            logger.info(f"Encoding {split}...")
            embeddings = self.encoder.encode(
                texts, batch_size=64, show_progress_bar=True
            )

            splits[split] = {
                "texts": texts,
                "intents": intents,
                "labels": labels,
                "embeddings": embeddings,
            }

            logger.info(
                f"  {split}: {len(texts)} samples "
                f"({np.sum(labels == 0)} ID, {np.sum(labels == 1)} OOS)"
            )

        return splits

    def train_base_detector(self, train_data: Dict):
        """在Train上训练基础检测器（Class centroid + 统计半径 + 对角马氏）"""
        logger.info("=" * 80)
        logger.info("【Phase 1】Training Base Detector on TRAIN")
        logger.info("=" * 80)

        # 只用ID样本
        train_id_mask = train_data["labels"] == 0
        self.train_id_embeddings = train_data["embeddings"][train_id_mask]
        self.train_id_intents = train_data["intents"][train_id_mask]

        logger.info(f"Training on {len(self.train_id_embeddings)} ID samples...")

        # 训练检测器（不改embedding，仅升级决策函数）
        self.base_detector = MultiSphereOOSDetector(
            n_clusters=None,  # 自动设为意图数
            radius_quantile=0.90,
            radius_method="mean_std",
            radius_lambda=1.0,
            center_mode=self.center_mode,
            distance_metric=self.distance_metric,
            margin_gamma=None,
            l2_normalize=self.l2_normalize,
            subcenters_per_intent=self.subcenters_per_intent,
            random_state=42,
        )

        self.base_detector.fit(self.train_id_embeddings, self.train_id_intents)

        logger.info(
            f"✓ Base detector trained with {len(self.base_detector.spheres)} spheres"
        )
        logger.info("  Centers fixed as per-intent centroids (no KMeans)")

        return self.base_detector

    def tune_radius_on_val(self, val_data: Dict):
        """在Val上调优统计半径lambda + 最近次近margin（中心不变）"""
        logger.info("=" * 80)
        logger.info("【Phase 2】Tuning lambda and margin on VAL")
        logger.info("=" * 80)
        logger.info("⚠️  CRITICAL: Only boundary parameters change, centers stay FIXED")

        lambda_candidates = [0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 2.00, 2.25, 2.50]
        gamma_candidates = [None, 0.98, 0.95, 0.92]
        results = []

        base_metrics = self.base_detector.evaluate(
            val_data["embeddings"], val_data["labels"]
        )
        min_id_recall = max(base_metrics.id_recall, self.min_id_recall_constraint)
        logger.info(
            "Baseline on Val: ID=%.3f OOS_Rej=%.3f F1=%.4f",
            base_metrics.id_recall,
            base_metrics.oos_rejection,
            base_metrics.f1_like,
        )
        logger.info("ID recall constraint on Val: >= %.3f", min_id_recall)

        for lam in lambda_candidates:
            for gamma in gamma_candidates:
                self.base_detector.radius_lambda = lam
                self.base_detector.margin_gamma = gamma
                self.base_detector._compute_radii(self.train_id_embeddings)

                metrics = self.base_detector.evaluate(
                    val_data["embeddings"], val_data["labels"]
                )

                results.append(
                    {
                        "lambda": lam,
                        "margin_gamma": gamma,
                        "id_recall": metrics.id_recall,
                        "oos_rejection": metrics.oos_rejection,
                        "f1_like": metrics.f1_like,
                        "meets_id_constraint": bool(metrics.id_recall >= min_id_recall),
                    }
                )

                gamma_str = "None" if gamma is None else f"{gamma:.2f}"
                logger.info(
                    "  λ=%s γ=%s: ID=%.3f OOS_Rej=%.3f F1=%.4f",
                    f"{lam:.2f}",
                    gamma_str,
                    metrics.id_recall,
                    metrics.oos_rejection,
                    metrics.f1_like,
                )

        constrained = [x for x in results if x["meets_id_constraint"]]
        if len(constrained) > 0:
            best_result = max(
                constrained, key=lambda x: (x["oos_rejection"], x["f1_like"])
            )
            logger.info(
                "Using constrained selection: maximize OOS_Rej with ID recall guard"
            )
        else:
            best_result = max(results, key=lambda x: x["f1_like"])
            logger.warning("No candidate meets ID recall guard, fallback to max F1")

        best_lambda = best_result["lambda"]
        best_gamma = best_result["margin_gamma"]

        logger.info(
            "\n✓ Best boundary: λ=%.2f γ=%s (ID=%.3f OOS_Rej=%.3f F1=%.4f)",
            best_lambda,
            "None" if best_gamma is None else f"{best_gamma:.2f}",
            best_result["id_recall"],
            best_result["oos_rejection"],
            best_result["f1_like"],
        )

        self.base_detector.radius_lambda = best_lambda
        self.base_detector.margin_gamma = best_gamma
        self.base_detector._compute_radii(self.train_id_embeddings)

        return best_result, results

    def evaluate_on_test(self, test_data: Dict):
        """在Test上最终评估"""
        logger.info("=" * 80)
        logger.info("【Phase 3】Final Evaluation on TEST")
        logger.info("=" * 80)
        logger.info(
            "Boundary config: λ=%.2f, γ=%s, metric=%s",
            self.base_detector.radius_lambda,
            "None"
            if self.base_detector.margin_gamma is None
            else f"{self.base_detector.margin_gamma:.2f}",
            self.base_detector.distance_metric,
        )

        metrics = self.base_detector.evaluate(
            test_data["embeddings"], test_data["labels"]
        )

        logger.info(
            "Using radius setup: method=%s, λ=%.2f",
            self.base_detector.radius_method,
            self.base_detector.radius_lambda,
        )
        logger.info(f"\nTest Performance:")
        logger.info(f"  ID Recall: {metrics.id_recall * 100:.2f}%")
        logger.info(f"  OOS Rejection: {metrics.oos_rejection * 100:.2f}%")
        logger.info(f"  F1-like: {metrics.f1_like:.4f}")

        # 计算混淆矩阵
        predictions = self.base_detector.predict(test_data["embeddings"])
        labels = test_data["labels"]

        tp = np.sum((predictions == 0) & (labels == 0))  # ID正确识别
        fn = np.sum((predictions == 1) & (labels == 0))  # ID漏判
        fp = np.sum((predictions == 0) & (labels == 1))  # OOS误判为ID
        tn = np.sum((predictions == 1) & (labels == 1))  # OOS正确拒绝

        logger.info(f"\nConfusion Matrix:")
        logger.info(f"               Pred ID    Pred OOS")
        logger.info(f"  True ID      {tp:4d}       {fn:4d}")
        logger.info(f"  True OOS     {fp:4d}       {tn:4d}")

        # 失败样本分析
        diagnosis = self.base_detector.diagnose_failures(
            test_data["embeddings"], test_data["labels"], test_data["texts"]
        )

        logger.info(f"\nFailure Analysis:")
        logger.info(
            f"  False Negatives (ID→OOS): {diagnosis['false_negatives']['count']}"
        )
        logger.info(
            f"  False Positives (OOS→ID): {diagnosis['false_positives']['count']}"
        )

        return (
            metrics,
            diagnosis,
            {"tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn)},
        )

    def save_results(self, val_results, test_metrics, test_diagnosis, confusion_matrix):
        """保存结果"""
        output_file = self.output_dir / "corrected_multisphere_results.json"

        results = {
            "config": {
                "n_clusters": len(self.base_detector.spheres),
                "center_mode": self.base_detector.center_mode,
                "distance_metric": self.base_detector.distance_metric,
                "l2_normalize": self.base_detector.l2_normalize,
                "subcenters_per_intent": self.base_detector.subcenters_per_intent,
                "radius_method": self.base_detector.radius_method,
                "final_radius_lambda": self.base_detector.radius_lambda,
                "final_margin_gamma": self.base_detector.margin_gamma,
                "note": "Class centroid + adaptive radius + diagonal Mahalanobis + margin tuning on Val",
            },
            "validation": {
                "tuning_results": val_results,
                "best_lambda": self.base_detector.radius_lambda,
                "best_margin_gamma": self.base_detector.margin_gamma,
            },
            "test": {
                "id_recall": test_metrics.id_recall,
                "oos_rejection": test_metrics.oos_rejection,
                "f1_like": test_metrics.f1_like,
                "confusion_matrix": confusion_matrix,
                "false_negatives": test_diagnosis["false_negatives"]["count"],
                "false_positives": test_diagnosis["false_positives"]["count"],
            },
        }

        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"\n✓ Results saved to {output_file}")

        # 保存检测器
        detector_file = self.output_dir / "corrected_multisphere_detector.json"
        self.base_detector.save(detector_file)

        return results

    def run(self):
        """完整训练流程"""
        logger.info("=" * 80)
        logger.info("Corrected Multi-Sphere OOS Detector - Training Pipeline")
        logger.info("=" * 80)

        # 1. 加载并编码数据
        splits = self.load_and_encode_data()

        # 2. 在Train上训练基础检测器
        self.train_base_detector(splits["train"])

        # 3. 在Val上调优半径（簇中心不变）
        best_boundary, val_results = self.tune_radius_on_val(splits["val"])

        # 4. 在Test上评估
        test_metrics, test_diagnosis, confusion_matrix = self.evaluate_on_test(
            splits["test"]
        )

        # 5. 保存结果
        results = self.save_results(
            val_results, test_metrics, test_diagnosis, confusion_matrix
        )

        logger.info("=" * 80)
        logger.info("【COMPLETE】Corrected Multi-Sphere Detector")
        logger.info("=" * 80)
        logger.info(f"Test F1: {test_metrics.f1_like:.4f}")
        logger.info(f"ID Recall: {test_metrics.id_recall * 100:.1f}%")
        logger.info(f"OOS Rejection: {test_metrics.oos_rejection * 100:.1f}%")
        logger.info("=" * 80)

        return results


def main():
    parser = argparse.ArgumentParser(description="Train corrected multisphere gate")
    parser.add_argument("--data_root", default="data/clinc150/kir50_seed42")
    parser.add_argument(
        "--model_path", default="models/all-MiniLM-L6-v2"
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/gate",
    )
    parser.add_argument("--min_id_recall_constraint", type=float, default=0.85)
    parser.add_argument(
        "--center_mode",
        choices=["class_centroid", "class_centroid_mixture", "kmeans"],
        default="class_centroid",
    )
    parser.add_argument(
        "--distance_metric",
        choices=["euclidean", "mahalanobis_diag"],
        default="mahalanobis_diag",
    )
    parser.add_argument("--l2_normalize", action="store_true")
    parser.add_argument("--subcenters_per_intent", type=int, default=1)
    args = parser.parse_args()

    trainer = CorrectedMultiSphereTrainer(
        data_root=Path(args.data_root),
        model_path=Path(args.model_path),
        output_dir=Path(args.output_dir),
        min_id_recall_constraint=args.min_id_recall_constraint,
        center_mode=args.center_mode,
        distance_metric=args.distance_metric,
        l2_normalize=args.l2_normalize,
        subcenters_per_intent=args.subcenters_per_intent,
    )

    results = trainer.run()


if __name__ == "__main__":
    main()
