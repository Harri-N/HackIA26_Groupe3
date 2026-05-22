"""
Multi-seed, multi-metric benchmark of pretrained architectures on augmented database.

For each architecture in MODEL_NAMES:
    for each seed in SEEDS:
        train from ImageNet weights, evaluate on test split
    aggregate per metric (mean ± std across seeds)

Metrics reported per model
--------------------------
    - test accuracy
    - macro precision, recall, F1  (unweighted average over the 3 classes)
    - per-class F1                  (no_fire, start_fire, fire)

Run: hackia/bin/python3 compare_models_v2_multiseed.py
"""

from __future__ import annotations

import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SEEDS = [42, 123, 456]
DATA_ROOT = Path("clean_database_v2")
CLASSES = ["no_fire", "start_fire", "fire"]
NUM_CLASSES = len(CLASSES)

IMAGE_SIZE = 224
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = Path("compare_models_v2_multiseed_results.json")

MODEL_NAMES = [
    "resnet18",
    "resnet50",
    "mobilenet_v3_small",
    "efficientnet_b0",
    "efficientnet_b1",
    "densenet121",
    "convnext_tiny",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Splits, data, model factory  (same as compare_models_v2.py)
# ---------------------------------------------------------------------------

def source_id_of(p: Path) -> str:
    return p.stem.split("_aug")[0]


def build_splits(root: Path, seed: int, ratios=(0.70, 0.15, 0.15)):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for label_idx, cls in enumerate(CLASSES):
        groups = defaultdict(list)
        for p in sorted((root / cls).iterdir()):
            if p.is_file():
                groups[source_id_of(p)].append(p)
        img_ids = sorted(k for k in groups if k.startswith("img_"))
        extra_ids = sorted(k for k in groups if k.startswith("extra_"))
        rng.shuffle(img_ids)
        n = len(img_ids)
        n_tr = int(n * ratios[0])
        n_va = int(n * ratios[1])
        for sid in img_ids[:n_tr]:
            train.extend((p, label_idx) for p in groups[sid])
        for sid in img_ids[n_tr:n_tr + n_va]:
            val.extend((p, label_idx) for p in groups[sid])
        for sid in img_ids[n_tr + n_va:]:
            test.extend((p, label_idx) for p in groups[sid])
        for sid in extra_ids:
            train.extend((p, label_idx) for p in groups[sid])
    rng.shuffle(train)
    return train, val, test


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FireDataset(Dataset):
    def __init__(self, items, transform):
        self.items, self.transform = items, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        return self.transform(Image.open(path).convert("RGB")), label


def build_model(name: str):
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        head = list(m.fc.parameters())
    elif name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, NUM_CLASSES)
        head = list(m.fc.parameters())
    elif name == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
        head = list(m.classifier[-1].parameters())
    elif name == "efficientnet_b0":
        m = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
        head = list(m.classifier[-1].parameters())
    elif name == "efficientnet_b1":
        m = models.efficientnet_b1(
            weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
        head = list(m.classifier[-1].parameters())
    elif name == "densenet121":
        m = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, NUM_CLASSES)
        head = list(m.classifier.parameters())
    elif name == "convnext_tiny":
        m = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_f = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_f, NUM_CLASSES)
        head = list(m.classifier[-1].parameters())
    else:
        raise ValueError(name)
    n_params = sum(p.numel() for p in m.parameters())
    return m, head, n_params


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def confusion_matrix(y, p, n):
    cm = np.zeros((n, n), dtype=int)
    for t, pr in zip(y, p):
        cm[t, pr] += 1
    return cm


def per_class_prf(cm):
    """Per-class precision, recall, F1 (no averaging)."""
    out = {}
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        out[CLASSES[i]] = {"precision": p, "recall": r, "f1": f1}
    return out


def macro_prf(per_class):
    p = statistics.mean(c["precision"] for c in per_class.values())
    r = statistics.mean(c["recall"]    for c in per_class.values())
    f1 = statistics.mean(c["f1"]       for c in per_class.values())
    return {"precision": p, "recall": r, "f1": f1}


def run_epoch(model, loader, criterion, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total = correct = 0
    ys, ps = [], []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total += x.size(0)
            ys.append(y.cpu().numpy())
            ps.append(pred.cpu().numpy())
    return {"acc": correct / total,
            "y": np.concatenate(ys), "p": np.concatenate(ps)}


def train_one(name, seed, train_loader, val_loader, test_loader):
    seed_everything(seed)
    model, head_params, n_params = build_model(name)
    model.to(DEVICE)
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": LR_BACKBONE},
        {"params": head_params, "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_state = None
    t0 = time.time()
    for _ in range(NUM_EPOCHS):
        run_epoch(model, train_loader, criterion, optimizer)
        va = run_epoch(model, val_loader, criterion)
        if va["acc"] > best_val_acc:
            best_val_acc = va["acc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
    train_time = time.time() - t0
    model.load_state_dict(best_state)

    te = run_epoch(model, test_loader, criterion)
    cm = confusion_matrix(te["y"], te["p"], NUM_CLASSES)
    pc = per_class_prf(cm)
    mac = macro_prf(pc)
    return {
        "seed": seed,
        "n_params": n_params,
        "train_time_sec": train_time,
        "best_val_acc": best_val_acc,
        "test_acc": te["acc"],
        "macro_precision": mac["precision"],
        "macro_recall": mac["recall"],
        "macro_f1": mac["f1"],
        "per_class": pc,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------

def stat(values):
    return {"mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "runs": [round(v, 4) for v in values]}


def aggregate(runs):
    accs = [r["test_acc"] for r in runs]
    macP = [r["macro_precision"] for r in runs]
    macR = [r["macro_recall"] for r in runs]
    macF = [r["macro_f1"] for r in runs]
    per_class = {}
    for cls in CLASSES:
        per_class[cls] = {
            "precision": stat([r["per_class"][cls]["precision"] for r in runs]),
            "recall":    stat([r["per_class"][cls]["recall"]    for r in runs]),
            "f1":        stat([r["per_class"][cls]["f1"]        for r in runs]),
        }
    return {
        "n_params_M": round(runs[0]["n_params"] / 1e6, 2),
        "train_time_sec_mean": round(statistics.mean(r["train_time_sec"] for r in runs), 1),
        "test_acc": stat(accs),
        "macro_precision": stat(macP),
        "macro_recall": stat(macR),
        "macro_f1": stat(macF),
        "per_class": per_class,
    }


def fmt(s):
    return f"{s['mean']:.4f} ± {s['std']:.4f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}  data: {DATA_ROOT}  seeds: {SEEDS}")
    print(f"Plan: {len(MODEL_NAMES)} models × {len(SEEDS)} seeds = "
          f"{len(MODEL_NAMES)*len(SEEDS)} trainings\n")

    # Build a separate split per seed so the test set varies (each seed
    # is a fresh experiment) — but the split is consistent across models
    # within one seed because seed_everything is called before each train.
    splits_by_seed = {}
    for s in SEEDS:
        splits_by_seed[s] = build_splits(DATA_ROOT, s)
        tr, va, te = splits_by_seed[s]
        print(f"  seed {s}: train={len(tr)} val={len(va)} test={len(te)}")
    print()

    all_results = {}  # model_name -> {"runs": [...], "agg": {...}}
    t0_overall = time.time()
    for name in MODEL_NAMES:
        print(f"\n========== {name} ==========")
        runs = []
        for s in SEEDS:
            tr, va, te = splits_by_seed[s]
            common = dict(num_workers=NUM_WORKERS, pin_memory=True)
            train_loader = DataLoader(FireDataset(tr, train_tf),
                                      batch_size=BATCH_SIZE, shuffle=True,
                                      **common)
            val_loader = DataLoader(FireDataset(va, eval_tf),
                                    batch_size=BATCH_SIZE, **common)
            test_loader = DataLoader(FireDataset(te, eval_tf),
                                     batch_size=BATCH_SIZE, **common)
            r = train_one(name, s, train_loader, val_loader, test_loader)
            runs.append(r)
            print(f"  seed {s}: acc={r['test_acc']:.4f}  "
                  f"macroF1={r['macro_f1']:.4f}  "
                  f"({r['train_time_sec']:.1f}s)")
        agg = aggregate(runs)
        all_results[name] = {"runs": runs, "agg": agg}
        print(f"  agg:    acc={fmt(agg['test_acc'])}  "
              f"macroF1={fmt(agg['macro_f1'])}")

    print(f"\nAll done in {(time.time()-t0_overall)/60:.1f} min")

    # ---------- Final table ----------
    print("\n" + "=" * 110)
    print(" AGGREGATE RESULTS (mean ± std over 3 seeds)")
    print("=" * 110)
    print(f"{'model':<22}{'params(M)':>10}  {'acc':>15}  "
          f"{'macroP':>15}  {'macroR':>15}  {'macroF1':>15}")
    sortable = sorted(all_results.items(),
                      key=lambda kv: -kv[1]["agg"]["test_acc"]["mean"])
    for name, d in sortable:
        a = d["agg"]
        print(f"{name:<22}{a['n_params_M']:>10.2f}  "
              f"{fmt(a['test_acc']):>15}  {fmt(a['macro_precision']):>15}  "
              f"{fmt(a['macro_recall']):>15}  {fmt(a['macro_f1']):>15}")

    print("\nPer-class F1 (mean ± std over 3 seeds)")
    print(f"{'model':<22}  {'no_fire F1':>17}  {'start_fire F1':>17}  "
          f"{'fire F1':>17}")
    for name, d in sortable:
        pc = d["agg"]["per_class"]
        print(f"{name:<22}  {fmt(pc['no_fire']['f1']):>17}  "
              f"{fmt(pc['start_fire']['f1']):>17}  "
              f"{fmt(pc['fire']['f1']):>17}")

    # Persist
    serializable = {
        name: {
            "agg": d["agg"],
            "runs": [{k: v for k, v in r.items() if k != "best_state"}
                     for r in d["runs"]],
        }
        for name, d in all_results.items()
    }
    RESULTS_FILE.write_text(json.dumps(serializable, indent=2))
    print(f"\nFull results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
