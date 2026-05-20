"""
Train a Swin Transformer fire/no-fire classifier with PyTorch Lightning.

All hyperparameters live in `configs/train_fire.yaml`. CLI flags override.

Target hardware: NVIDIA RTX 4090 (24 GB) for training,
deploy on NVIDIA Jetson Orin.

Conda env: hackia.

Run:
    conda activate hackia
    python src/train_fire.py
    python src/train_fire.py --epochs 50 --batch_size 64

Live monitoring (separate terminal):
    python src/monitor_fire.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
import yaml
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import (
    BinaryAccuracy, BinaryF1Score,
    MulticlassAccuracy, MulticlassF1Score,
)
from torchvision import transforms
from torchvision.datasets import ImageFolder

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "configs" / "train_fire.yaml"

# Map config name -> (torchvision constructor, default weights enum string).
_SWIN_MODELS = {
    "swin_t":    (torchvision.models.swin_t,    "Swin_T_Weights.IMAGENET1K_V1"),
    "swin_s":    (torchvision.models.swin_s,    "Swin_S_Weights.IMAGENET1K_V1"),
    "swin_b":    (torchvision.models.swin_b,    "Swin_B_Weights.IMAGENET1K_V1"),
    "swin_v2_t": (torchvision.models.swin_v2_t, "Swin_V2_T_Weights.IMAGENET1K_V1"),
    "swin_v2_s": (torchvision.models.swin_v2_s, "Swin_V2_S_Weights.IMAGENET1K_V1"),
}

# EfficientNet family. Head lives at `model.classifier[1]` (a Linear preceded
# by Dropout at index 0). V2 variants use IMAGENET1K_V1 weights.
_EFFNET_MODELS = {
    "efficientnet_b0":    (torchvision.models.efficientnet_b0,    "EfficientNet_B0_Weights.IMAGENET1K_V1"),
    "efficientnet_b3":    (torchvision.models.efficientnet_b3,    "EfficientNet_B3_Weights.IMAGENET1K_V1"),
    "efficientnet_b4":    (torchvision.models.efficientnet_b4,    "EfficientNet_B4_Weights.IMAGENET1K_V1"),
    "efficientnet_v2_s":  (torchvision.models.efficientnet_v2_s,  "EfficientNet_V2_S_Weights.IMAGENET1K_V1"),
    "efficientnet_v2_m":  (torchvision.models.efficientnet_v2_m,  "EfficientNet_V2_M_Weights.IMAGENET1K_V1"),
}

# ResNet family. Head lives at `model.fc` (the final Linear).
# Use the V2 IMAGENET1K weights (significantly better than V1 on torchvision).
_RESNET_MODELS = {
    "resnet18":  (torchvision.models.resnet18,  "ResNet18_Weights.IMAGENET1K_V1"),
    "resnet50":  (torchvision.models.resnet50,  "ResNet50_Weights.IMAGENET1K_V2"),
    "resnet101": (torchvision.models.resnet101, "ResNet101_Weights.IMAGENET1K_V2"),
}


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def build_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CFG))
    for key, typ in [
        ("model", str), ("run_name", str), ("data_dir", str),
        ("epochs", int), ("batch_size", int), ("imgsz", int),
        ("lr", float), ("weight_decay", float), ("workers", int),
        ("precision", str), ("patience", int),
    ]:
        p.add_argument(f"--{key}", type=typ, default=None)
    return p.parse_args()


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def _resolve_weights(weights_str: str):
    mod_path, attr = weights_str.split(".", 1)
    return getattr(getattr(torchvision.models, mod_path), attr)


def build_model(name: str, num_classes: int, pretrained: bool) -> nn.Module:
    """Build a torchvision backbone with the classifier head replaced.

    Supports Swin (head at `model.head`) and EfficientNet (head at
    `model.classifier[1]`).
    """
    if name in _SWIN_MODELS:
        ctor, weights_str = _SWIN_MODELS[name]
        weights = _resolve_weights(weights_str) if pretrained else None
        model = ctor(weights=weights)
        in_feat = model.head.in_features
        model.head = nn.Linear(in_feat, num_classes)
        return model
    if name in _EFFNET_MODELS:
        ctor, weights_str = _EFFNET_MODELS[name]
        weights = _resolve_weights(weights_str) if pretrained else None
        model = ctor(weights=weights)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, num_classes)
        return model
    if name in _RESNET_MODELS:
        ctor, weights_str = _RESNET_MODELS[name]
        weights = _resolve_weights(weights_str) if pretrained else None
        model = ctor(weights=weights)
        in_feat = model.fc.in_features
        model.fc = nn.Linear(in_feat, num_classes)
        return model
    raise ValueError(
        f"Unknown model: {name}. "
        f"Choose from "
        f"{list(_SWIN_MODELS) + list(_EFFNET_MODELS) + list(_RESNET_MODELS)}"
    )


# Back-compat alias (older code/tests may import `build_swin`).
build_swin = build_model


class FireClassifier(pl.LightningModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.save_hyperparameters(cfg)
        self.cfg = cfg
        self.model = build_model(
            cfg["model"], cfg["num_classes"], cfg.get("pretrained", True)
        )
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(cfg.get("label_smoothing", 0.0))
        )
        n = int(cfg["num_classes"])
        if n == 2:
            # Positive class = index 1, set by forced class_names ordering.
            self.train_acc = BinaryAccuracy()
            self.val_acc   = BinaryAccuracy()
            self.val_f1    = BinaryF1Score()
        else:
            self.train_acc = MulticlassAccuracy(num_classes=n, average="micro")
            self.val_acc   = MulticlassAccuracy(num_classes=n, average="micro")
            self.val_f1    = MulticlassF1Score(num_classes=n, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _step(self, batch, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        if stage == "train":
            self.train_acc(preds, y)
            self.log("train_loss", loss, prog_bar=True, on_step=True,
                     on_epoch=True)
            self.log("train_acc", self.train_acc, prog_bar=True,
                     on_step=False, on_epoch=True)
        else:
            self.val_acc(preds, y)
            self.val_f1(preds, y)
            self.log("val_loss", loss, prog_bar=True, on_epoch=True)
            self.log("val_acc",  self.val_acc, prog_bar=True, on_epoch=True)
            self.log("val_f1",   self.val_f1,  prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        opt_name = self.cfg.get("optimizer", "adamw").lower()
        if opt_name == "adamw":
            opt = torch.optim.AdamW(
                self.parameters(),
                lr=float(self.cfg["lr"]),
                weight_decay=float(self.cfg.get("weight_decay", 0.05)),
            )
        else:
            opt = torch.optim.SGD(
                self.parameters(),
                lr=float(self.cfg["lr"]),
                momentum=0.9,
                weight_decay=float(self.cfg.get("weight_decay", 0.0001)),
            )

        total_epochs = int(self.cfg["epochs"])
        warmup = int(self.cfg.get("warmup_epochs", 0))

        def lr_lambda(epoch: int) -> float:
            if warmup and epoch < warmup:
                return (epoch + 1) / max(1, warmup)
            if not self.cfg.get("cos_lr", True):
                return 1.0
            progress = (epoch - warmup) / max(1, total_epochs - warmup)
            return 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler,
                                                   "interval": "epoch"}}


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
@dataclass
class DataBundle:
    train: DataLoader
    val:   DataLoader
    classes: list[str]
    n_train: int
    n_val: int


def make_transforms(cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    sz = int(cfg["imgsz"])
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose([
        transforms.Resize(int(sz * 1.15)),
        transforms.RandomCrop(sz),
        transforms.RandomHorizontalFlip(p=float(cfg.get("hflip", 0.5))),
        transforms.ColorJitter(
            brightness=float(cfg.get("color_jitter", 0.2)),
            contrast=float(cfg.get("color_jitter", 0.2)),
            saturation=float(cfg.get("color_jitter", 0.2)),
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        transforms.RandomErasing(p=float(cfg.get("random_erasing", 0.1))),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(sz * 1.15)),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    return train_tf, val_tf


def stratified_split(targets: list[int], val_frac: float,
                     seed: int) -> tuple[list[int], list[int]]:
    rng = random.Random(seed)
    by_class: dict[int, list[int]] = {}
    for idx, y in enumerate(targets):
        by_class.setdefault(y, []).append(idx)
    train_idx, val_idx = [], []
    for cls, idxs in by_class.items():
        rng.shuffle(idxs)
        n_val = int(round(len(idxs) * val_frac))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def _force_class_order(ds: ImageFolder, class_names: list[str]) -> None:
    """Reindex an ImageFolder so its label order matches `class_names`.

    ImageFolder uses alphabetical class order by default. For a fire
    detector we want positive class (index 1) = "fire" so that the
    Binary* torchmetrics report precision/recall *of fires*, not of
    no-fires. This rewrites `samples`, `targets`, `classes`, `class_to_idx`
    in place.
    """
    if list(ds.classes) == list(class_names):
        return
    if set(ds.classes) != set(class_names):
        raise ValueError(
            f"class_names {class_names} don't match folders {ds.classes}")
    remap = {ds.class_to_idx[c]: i for i, c in enumerate(class_names)}
    ds.samples = [(p, remap[y]) for (p, y) in ds.samples]
    ds.targets = [remap[y] for y in ds.targets]
    ds.classes = list(class_names)
    ds.class_to_idx = {c: i for i, c in enumerate(class_names)}


def build_dataloaders(cfg: dict) -> DataBundle:
    data_dir = REPO / cfg["data_dir"]
    train_tf, val_tf = make_transforms(cfg)

    # Two layouts are supported:
    #   A) flat:        data_dir/{class}/...                -> stratified split
    #   B) pre-split:   data_dir/{train,val[,test]}/{class}/...  -> use as-is
    if (data_dir / "train").is_dir() and (data_dir / "val").is_dir():
        full_train = ImageFolder(str(data_dir / "train"), transform=train_tf)
        full_val   = ImageFolder(str(data_dir / "val"),   transform=val_tf)
        if cfg.get("class_names"):
            _force_class_order(full_train, list(cfg["class_names"]))
            _force_class_order(full_val,   list(cfg["class_names"]))
        assert full_train.classes == full_val.classes
        train_ds, val_ds = full_train, full_val
    else:
        full_train = ImageFolder(str(data_dir), transform=train_tf)
        full_val   = ImageFolder(str(data_dir), transform=val_tf)
        if cfg.get("class_names"):
            _force_class_order(full_train, list(cfg["class_names"]))
            _force_class_order(full_val,   list(cfg["class_names"]))
        assert full_train.classes == full_val.classes
        train_idx, val_idx = stratified_split(
            [y for _, y in full_train.samples],
            float(cfg["val_split"]),
            int(cfg.get("seed", 42)),
        )
        train_ds = Subset(full_train, train_idx)
        val_ds   = Subset(full_val,   val_idx)

    bs = int(cfg["batch_size"])
    nw = int(cfg.get("workers", 0))
    common = dict(num_workers=nw, pin_memory=(nw == 0))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              drop_last=True, **common)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False,
                              drop_last=False, **common)
    return DataBundle(train_loader, val_loader, full_train.classes,
                      len(train_ds), len(val_ds))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def assert_gpu_has_room(min_free_gb: float = 4.0) -> None:
    """Refuse to launch if <min_free_gb is available on GPU 0."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available — training requires a GPU.")
    free, total = torch.cuda.mem_get_info(0)
    free_gb  = free  / 1e9
    total_gb = total / 1e9
    print(f"[gpu] device={torch.cuda.get_device_name(0)} "
          f"free={free_gb:.1f}GB / {total_gb:.1f}GB")
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Only {free_gb:.1f} GB free on GPU 0 (need ≥ {min_free_gb} GB). "
            f"Wait for the firezone run to finish or pick another device."
        )


def main():
    args = build_args()
    cfg = load_config(Path(args.config))
    for k, v in vars(args).items():
        if k == "config" or v is None:
            continue
        cfg[k] = v

    pl.seed_everything(int(cfg.get("seed", 42)), workers=True)
    assert_gpu_has_room(min_free_gb=4.0)

    project_dir = REPO / cfg["project_dir"]
    run_dir = project_dir / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the resolved config alongside the checkpoints.
    with (run_dir / "config_used.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    data = build_dataloaders(cfg)
    pos_note = (f"positive class = idx 1 ({data.classes[1]})"
                if len(data.classes) == 2
                else f"{len(data.classes)}-way classification")
    print(f"[data] classes={data.classes}  ({pos_note})  "
          f"train={data.n_train}  val={data.n_val}")

    # Persist info the monitor needs to draw a step-level progress bar.
    steps_per_epoch = data.n_train // int(cfg["batch_size"])  # drop_last=True
    with (run_dir / "run_info.json").open("w") as f:
        json.dump({
            "n_train": data.n_train,
            "n_val": data.n_val,
            "batch_size": int(cfg["batch_size"]),
            "steps_per_epoch": steps_per_epoch,
        }, f, indent=2)

    model = FireClassifier(cfg)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(run_dir / "checkpoints"),
        filename="best-{epoch:02d}-{val_acc:.4f}",
        monitor="val_acc",
        mode="max",
        save_top_k=int(cfg.get("save_top_k", 1)),
        save_last=True,
        auto_insert_metric_name=False,
    )
    early_cb = EarlyStopping(
        monitor="val_acc",
        mode="max",
        patience=int(cfg.get("patience", 8)),
        verbose=True,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    csv_logger = CSVLogger(save_dir=str(project_dir), name=cfg["run_name"],
                           version="")

    trainer = pl.Trainer(
        max_epochs=int(cfg["epochs"]),
        accelerator="gpu",
        devices=int(cfg.get("gpus", 1)),
        precision=cfg.get("precision", "16-mixed"),
        callbacks=[ckpt_cb, early_cb, lr_cb],
        logger=csv_logger,
        log_every_n_steps=int(cfg.get("log_every_n_steps", 5)),
        deterministic=False,
        enable_progress_bar=True,
    )

    print("[train] starting…")
    trainer.fit(model, data.train, data.val)

    # Persist a small "best" summary that monitor_fire.py can read.
    best_ckpt = ckpt_cb.best_model_path
    best_score = float(ckpt_cb.best_model_score) if ckpt_cb.best_model_score is not None else float("nan")
    summary = {
        "best_ckpt": best_ckpt,
        "best_val_acc": best_score,
        "classes": data.classes,
        "model": cfg["model"],
        "imgsz": int(cfg["imgsz"]),
    }
    with (run_dir / "best.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[done] best ckpt: {best_ckpt}  val_acc={best_score:.4f}")

    # ---- automatic report generation -----------------------------------
    if best_ckpt:
        try:
            import sys as _sys
            _sys.path.insert(0, str(REPO / "src"))
            from report_fire import (
                plot_training_curves, plot_confusion_matrix,
                plot_misclassified, evaluate_on_val, write_report,
            )
            plots_dir = run_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)

            curves_path = plots_dir / "training_curves.png"
            if (run_dir / "metrics.csv").exists():
                plot_training_curves(run_dir / "metrics.csv", curves_path)

            print("[report] evaluating best checkpoint on val set…")
            metrics = evaluate_on_val(run_dir, cfg, Path(best_ckpt))

            cm_path = plots_dir / "confusion_matrix.png"
            plot_confusion_matrix(metrics["confusion"], metrics["classes"],
                                  cm_path)

            plot_paths = {"cm": cm_path, "curves": curves_path}
            if metrics["miss_images"]:
                miss_path = plots_dir / "misclassified.png"
                plot_misclassified(metrics["miss_images"],
                                   metrics["miss_titles"], miss_path)
                plot_paths["miss"] = miss_path

            report_path = write_report(run_dir, cfg, Path(best_ckpt),
                                       metrics, plot_paths)
            print(f"[report] {report_path}")
            if isinstance(metrics["val_precision"], list):
                pc = ", ".join(f"{p:.3f}" for p in metrics["val_precision"])
                rc = ", ".join(f"{r:.3f}" for r in metrics["val_recall"])
                print(f"[report] val_acc={metrics['val_acc']:.4f}  "
                      f"f1_macro={metrics['val_f1']:.4f}  "
                      f"prec=[{pc}]  rec=[{rc}]")
            else:
                print(f"[report] val_acc={metrics['val_acc']:.4f}  "
                      f"f1={metrics['val_f1']:.4f}  "
                      f"prec={metrics['val_precision']:.4f}  "
                      f"rec={metrics['val_recall']:.4f}")
        except Exception as e:  # noqa: BLE001 — report must not fail the run
            print(f"[report] failed: {e}")


if __name__ == "__main__":
    main()
