"""
Stage-1 Router training — SmolLM cascade
========================================
SmolLM-135M + LoRA + CrossEntropy, 10-class domain classification.

Usage:
    python tools/train/train_router.py \\
        --model_path ../../assets/models/smollm135m \\
        --data_dir   data/v19/router \\
        --output_dir outputs/router_v19 \\
        --epochs 10 \\
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

# Make sure project root is on sys.path when run from any working directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.router import SmolLMRouter  # noqa: E402

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
class RouterDataset(Dataset):
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


def detect_num_classes(data_path: str) -> int:
    """Infer the number of router classes from the training split."""
    raw = json.load(open(data_path))
    labels = sorted({int(row["label"]) for row in raw})
    if len(labels) == 0:
        raise RuntimeError(f"Router training data is empty: {data_path}")
    expected = list(range(len(labels)))
    if labels != expected:
        log.warning(
            "Router labels are not dense from 0..n-1 (%s); using unique-label count=%d",
            labels,
            len(labels),
        )
    return len(labels)


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train SmolLM-135M Router")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/router")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--patience", type=int, default=5, help="Early-stop patience (val acc)"
    )
    parser.add_argument('--defer_test', action='store_true')
    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    if output_dir.exists():
        log.error(
            f"Output dir already exists: {output_dir}. Use a different --output_dir."
        )
        sys.exit(1)
    output_dir.mkdir(parents=True)

    # File handler
    fh = logging.FileHandler(output_dir / "train.log")
    fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
    logging.getLogger().addHandler(fh)

    log.info(f"Args: {vars(args)}")

    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # ------------------------------------------------------------------
    log.info(f"Loading tokenizer from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_dir = Path(args.data_dir)
    train_ds = RouterDataset(str(data_dir / "train.json"), tokenizer, args.max_length)
    val_ds = RouterDataset(str(data_dir / "val.json"), tokenizer, args.max_length)
    test_ds = [] if args.defer_test else RouterDataset(str(data_dir / "test.json"), tokenizer, args.max_length)
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
    num_classes = detect_num_classes(str(data_dir / "train.json"))
    log.info(f"Detected num_classes={num_classes} from {data_dir / 'train.json'}")

    log.info(
        f"Building SmolLMRouter ({num_classes}-class) with LoRA r={args.lora_r}, α={args.lora_alpha}"
    )
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
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()#这行代码的意义是：将模型设置为训练模式，启用dropout和batch normalization等训练时特有的行为
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

            if step % 50 == 0:
                log.info(
                    f"Epoch {epoch} step {step}/{len(train_loader)}  "
                    f"loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
                )

        train_loss = total_loss / total
        train_acc = correct / total
        val_loss, val_acc = evaluate(model, val_loader, device)

        log.info(
            f"Epoch {epoch:3d}  "
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
    if args.defer_test:
        log.info('Known-only checkpoint saved; test evaluation deferred')
        return
    log.info("Loading best checkpoint for test evaluation...")
    model.load_state_dict(torch.load(output_dir / "best_model.pt", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, device)
    log.info(f"TEST  loss={test_loss:.4f}  acc={test_acc:.4f}")

    # Also save per-domain accuracy
    model.eval()
    domain_correct = [0] * num_classes
    domain_total = [0] * num_classes
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask)
            preds = logits.argmax(dim=-1)
            for p, gt in zip(preds.tolist(), labels.tolist()):
                domain_correct[gt] += int(p == gt)
                domain_total[gt] += 1

    domain_names = sorted(
        {r["domain"] for r in json.load(open(data_dir / "train.json"))}
    )
    if len(domain_names) != num_classes:
        log.warning(
            "Domain name count (%d) does not match num_classes (%d); using label order for report only.",
            len(domain_names),
            num_classes,
        )
        domain_names = [f"class_{i}" for i in range(num_classes)]
    log.info("Per-domain test accuracy:")
    for i, (name, c, t) in enumerate(zip(domain_names, domain_correct, domain_total)):
        log.info(f"  [{i}] {name:22s}: {c}/{t} = {c / t:.4f}")

    results = {
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "per_domain_acc": {
            name: domain_correct[i] / domain_total[i]
            for i, name in enumerate(domain_names)
        },
        "args": vars(args),
    }
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Results saved → {output_dir / 'results.json'}")

    # Save final checkpoint
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    log.info(f"Final model saved → {output_dir / 'final_model.pt'}")


if __name__ == "__main__":
    if os.environ.get("PYTHONPATH") is None:
        os.environ["PYTHONPATH"] = str(PROJECT_ROOT)
    main()
