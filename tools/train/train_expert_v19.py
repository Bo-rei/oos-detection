"""
Stage-2 Expert training — HiLSA-MoE v19
========================================
SmolLM-135M + LoRA + CrossEntropy, in-domain intent classification.
One script handles any single domain; call train_all_experts_v19.py
to train all 10 domains sequentially.

Usage:
    python tools/train/train_expert_v19.py \\
        --domain    banking \\
        --model_path <SmolLM_PATH> \\
        --data_dir   data/clinc150/kir50_seed42/experts \\
        --output_dir outputs/experts_v19 \\
        --epochs 15 \\
        --batch_size 32 \\
        --lr 2e-4 \\
        --seed 42
"""

import argparse
import csv
import json
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.router.router_model import SmolLMRouter  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

MAX_LENGTH = 64


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ExpertDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = MAX_LENGTH):
        raw = json.load(open(path))
        self.texts = [r["text"] for r in raw]
        self.labels = [int(r["label"]) for r in raw]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Return (loss, accuracy) on a DataLoader."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += len(labels)
    return total_loss / total, correct / total


def detect_num_classes(data_path: str) -> int:
    """Infer number of classes from the label field in a JSON split file."""
    raw = json.load(open(data_path))
    return len(set(int(r["label"]) for r in raw))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Environment self-check (R3.1)
    # ------------------------------------------------------------------
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env != "bo":
        log.error(
            f"Expected conda env 'bo', got '{conda_env}'. "
            "Please activate 'conda activate bo' first."
        )
        sys.exit(1)

    pythonpath = os.environ.get("PYTHONPATH", "")
    if str(PROJECT_ROOT) not in pythonpath:
        log.warning(
            f"PYTHONPATH does not contain project root ({PROJECT_ROOT}). "
            "Consider running: export PYTHONPATH=<PROJECT_ROOT>"
        )

    if not torch.cuda.is_available():
        log.warning("CUDA not available — training will be slow on CPU.")

    # Experts-lock check (R3.2)
    lock_file = PROJECT_ROOT / ".no_experts_run"
    if lock_file.exists():
        log.error("Experts training locked by policy (.no_experts_run exists). Exit 2.")
        sys.exit(2)

    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train SmolLM-135M Expert (v19)")
    parser.add_argument("--domain", required=True, help="Domain name (e.g. banking)")
    parser.add_argument("--model_path", default="models/smollm135m")
    parser.add_argument(
        "--data_dir",
        default="data/clinc150/kir50_seed42/experts",
        help="Root dir containing per-domain sub-dirs",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/experts_v19",
        help="Root output dir; checkpoint saved under <output_dir>/<domain>/",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience", type=int, default=5, help="Early-stop patience (val acc)"
    )
    args = parser.parse_args()

    set_seed(args.seed)

    domain_data_dir = Path(args.data_dir) / args.domain
    if not domain_data_dir.exists():
        log.error(f"Domain data directory not found: {domain_data_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir) / args.domain
    if output_dir.exists():
        log.error(
            f"Output dir already exists: {output_dir}. "
            "Delete it or choose a different --output_dir."
        )
        sys.exit(1)
    output_dir.mkdir(parents=True)

    # File handler (Y1.1)
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"=== Expert training: domain={args.domain} ===")
    log.info(f"Args: {vars(args)}")

    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # Detect num_classes from training data
    num_classes = detect_num_classes(str(domain_data_dir / "train.json"))
    log.info(f"Detected num_classes={num_classes} for domain='{args.domain}'")

    # ------------------------------------------------------------------
    log.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = ExpertDataset(
        str(domain_data_dir / "train.json"), tokenizer, args.max_length
    )
    val_ds = ExpertDataset(
        str(domain_data_dir / "val.json"), tokenizer, args.max_length
    )
    test_ds = ExpertDataset(
        str(domain_data_dir / "test.json"), tokenizer, args.max_length
    )
    log.info(f"Train {len(train_ds)} | Val {len(val_ds)} | Test {len(test_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    log.info(
        f"Building SmolLMRouter({num_classes}-class) as Expert "
        f"with LoRA r={args.lora_r}, α={args.lora_alpha}"
    )
    # SmolLMRouter is a general sequence classifier; reused here for Expert.
    model = SmolLMRouter(
        model_path=args.model_path,
        num_classes=num_classes,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    ).to(device)
    model.base.print_trainable_parameters()

    # ------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = len(train_loader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # ------------------------------------------------------------------
    metrics_path = output_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_acc", "val_loss", "val_acc"]
        )

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item() * len(labels)
            preds = logits.detach().argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

            if step % 20 == 0:
                log.info(
                    f"[{args.domain}] Epoch {epoch} step {step}/{len(train_loader)}  "
                    f"loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
                )

        train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)

        log.info(
            f"[{args.domain}] Epoch {epoch:3d}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, train_acc, val_loss, val_acc])

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            ckpt_path = output_dir / "best_model.pt"
            torch.save(model.state_dict(), ckpt_path)
            log.info(f"  ↑ New best val_acc={best_val_acc:.4f}, saved {ckpt_path}")
        else:
            no_improve += 1
            log.info(f"  no improvement ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                log.info("Early stopping triggered.")
                break

    # ------------------------------------------------------------------
    # Final test evaluation on best checkpoint
    log.info("Loading best checkpoint for test evaluation ...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, device)
    log.info(f"[{args.domain}] TEST  loss={test_loss:.4f}  acc={test_acc:.4f}")

    # Per-class accuracy
    intent_names = sorted(
        set(r["intent"] for r in json.load(open(domain_data_dir / "train.json"))),
    )
    # Build label→intent mapping from train data
    label2intent: dict[int, str] = {}
    for r in json.load(open(domain_data_dir / "train.json")):
        label2intent[int(r["label"])] = r["intent"]

    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            for p, gt in zip(preds.tolist(), labels.tolist()):
                class_correct[gt] += int(p == gt)
                class_total[gt] += 1

    log.info(f"[{args.domain}] Per-class test accuracy:")
    per_class = {}
    for i in range(num_classes):
        intent = label2intent.get(i, f"class_{i}")
        acc = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        per_class[intent] = acc
        log.info(
            f"  [{i}] {intent:30s}: {class_correct[i]}/{class_total[i]} = {acc:.4f}"
        )

    results = {
        "domain": args.domain,
        "num_classes": num_classes,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "per_class_acc": per_class,
        "args": vars(args),
    }
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {results_path}")

    # Save final checkpoint
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    log.info(f"Final model saved → {output_dir / 'final_model.pt'}")

    # Save config snapshot (R4.1)
    with open(output_dir / "config.yaml", "w") as f:
        import yaml

        yaml.dump(vars(args), f, default_flow_style=False)
    log.info(f"Config saved → {output_dir / 'config.yaml'}")


if __name__ == "__main__":
    if "PYTHONPATH" not in os.environ:
        os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
    main()
