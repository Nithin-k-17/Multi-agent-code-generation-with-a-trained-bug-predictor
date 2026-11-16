"""Phase C: fine-tunes CodeBERT (or another encoder) as a binary bug classifier.

Settings from the report: AdamW, lr 2e-5, 10% linear warm-up, batch 16, fp16 on GPU,
early stopping on validation ROC-AUC (patience 2). The champion model is
CodeBERT, max_len 512, seed 123.

    python main.py train                                   # champion settings
    python main.py train --seed 42 --max-len 256           # another run
    python main.py train --model microsoft/unixcoder-base  # architecture comparison
    python main.py train --train-file train_combined       # combined-data ablation

The best checkpoint (by validation ROC-AUC) is saved to runs/<run-name>/ together with
threshold.json (F1-optimal threshold on validation) and val_metrics.json.
To use it in the rest of the project, copy its files into models/bug_predictor/.
A GPU is strongly recommended (about 12 minutes on an RTX 4060; several hours on CPU).
"""
import argparse
import json
import os
import random
import time

os.environ.setdefault("USE_TF", "0")  # PyTorch only: skip slow TensorFlow import inside transformers
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from config import BASE_ENCODER, MAX_LEN, TRAINING_RUNS_DIR
from training.common import best_f1_threshold, classification_metrics, load_split


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(tok, codes, labels, max_len, batch_size, shuffle):
    def collate(batch):
        texts, ys = zip(*batch)
        enc = tok(list(texts), truncation=True, max_length=max_len, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor(ys, dtype=torch.long)
        return enc
    return DataLoader(list(zip(codes, labels.tolist())), batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(**batch).logits
        probs += torch.softmax(logits.float(), dim=-1)[:, 1].tolist()
    return np.array(probs)


def main(argv=None):
    p = argparse.ArgumentParser(description="Fine-tune CodeBERT as a bug classifier.")
    p.add_argument("--model", default=BASE_ENCODER)
    p.add_argument("--train-file", default="train")
    p.add_argument("--val-file", default="val")
    p.add_argument("--max-len", type=int, default=MAX_LEN)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--run-name", default=None)
    p.add_argument("--limit", type=int, default=None, help="use only N training samples (quick smoke test)")
    args = p.parse_args(argv)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or f"{args.model.split('/')[-1]}_len{args.max_len}_seed{args.seed}"
    out_dir = os.path.join(TRAINING_RUNS_DIR, run_name)
    os.makedirs(out_dir, exist_ok=True)

    _, x_tr, y_tr, _ = load_split(args.train_file)
    _, x_va, y_va, t_va = load_split(args.val_file)
    if args.limit:
        x_tr, y_tr = x_tr[: args.limit], y_tr[: args.limit]
    print(f"Device {device} | train {len(y_tr)} | val {len(y_va)} | -> {out_dir}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2).to(device)
    train_loader = make_loader(tok, x_tr, y_tr, args.max_len, args.batch_size, shuffle=True)
    val_loader = make_loader(tok, x_va, y_va, args.max_len, args.batch_size * 2, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup * total_steps), total_steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    best_auc, bad_epochs, t0 = -1.0, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = model(**batch).loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(loss.item())
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{len(train_loader)} loss {np.mean(losses[-50:]):.4f}", flush=True)

        scores = predict(model, val_loader, device)
        threshold = best_f1_threshold(scores, y_va)
        metrics = classification_metrics(scores, y_va, t_va, threshold)
        print(f"Epoch {epoch}: loss {np.mean(losses):.4f} | val ROC-AUC {metrics['roc_auc']:.4f} "
              f"| F1 {metrics['f1']:.4f} @ tau={threshold}")

        if metrics["roc_auc"] > best_auc:
            best_auc, bad_epochs = metrics["roc_auc"], 0
            model.save_pretrained(out_dir)
            tok.save_pretrained(out_dir)
            meta = {"optimal_threshold": threshold, "best_threshold": threshold, "max_len": args.max_len,
                    "roc_auc": metrics["roc_auc"], "f1": metrics["f1"], "accuracy": metrics["accuracy"],
                    "model_name": args.model, "seed": args.seed}
            with open(os.path.join(out_dir, "threshold.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)
            with open(os.path.join(out_dir, "val_metrics.json"), "w", encoding="utf-8") as f:
                json.dump({**metrics, "epoch": epoch, "training_time_sec": time.time() - t0, "seed": args.seed}, f, indent=2)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("Early stopping.")
                break

    print(f"\nBest val ROC-AUC {best_auc:.4f}. Checkpoint: {out_dir}")


if __name__ == "__main__":
    main()
