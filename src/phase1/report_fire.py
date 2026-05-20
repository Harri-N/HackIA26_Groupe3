"""
Build evaluation plots + a markdown report for a trained fire classifier.

Reads:
  models/fire/<run_name>/checkpoints/best-*.ckpt   (best by val_acc)
  models/fire/<run_name>/metrics.csv               (CSVLogger output)
  models/fire/<run_name>/config_used.yaml          (resolved config snapshot)

Writes (under the same run dir):
  plots/training_curves.png        (loss + acc per epoch, train vs val)
  plots/confusion_matrix.png       (val set, normalized + counts)
  plots/misclassified.png          (up to 16 hardest val mistakes)
  REPORT.md                        (markdown summary of the run)

Called automatically at the end of `train_fire.py`. Can also be run standalone:

    conda activate hackia
    python src/report_fire.py --run-dir models/fire/swin_b_v1
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)
from torchvision.datasets import ImageFolder

REPO = Path(__file__).resolve().parents[1]


def _load_yaml(p: Path) -> dict:
    with p.open() as f:
        return yaml.safe_load(f) or {}


def find_best_ckpt(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    candidates = sorted(ckpt_dir.glob("best-*.ckpt"))
    if not candidates:
        candidates = sorted(ckpt_dir.glob("*.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    # If we have a best.json, prefer the path it points to.
    bj = run_dir / "best.json"
    if bj.exists():
        meta = json.loads(bj.read_text())
        p = Path(meta.get("best_ckpt", ""))
        if p.is_file():
            return p
    # Otherwise pick the one with the highest val_acc embedded in the filename.
    def _score(p: Path) -> float:
        try:
            return float(p.stem.split("-")[-1])
        except ValueError:
            return 0.0
    return max(candidates, key=_score)


def plot_training_curves(metrics_csv: Path, out_path: Path) -> None:
    df = pd.read_csv(metrics_csv)
    cols = ["train_loss_epoch", "train_acc_epoch",
            "val_loss", "val_acc", "val_f1"]
    present = [c for c in cols if c in df.columns]
    # Drop rows with no epoch number (CSVLogger writes some step-only rows),
    # then collapse to one row per epoch keeping the last non-null value.
    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)
    by_epoch = df.groupby("epoch")[present].last().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    if "train_loss_epoch" in by_epoch:
        ax.plot(by_epoch["epoch"], by_epoch["train_loss_epoch"],
                marker="o", label="train")
    if "val_loss" in by_epoch:
        ax.plot(by_epoch["epoch"], by_epoch["val_loss"],
                marker="s", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Loss")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    if "train_acc_epoch" in by_epoch:
        ax.plot(by_epoch["epoch"], by_epoch["train_acc_epoch"] * 100,
                marker="o", label="train_acc")
    if "val_acc" in by_epoch:
        ax.plot(by_epoch["epoch"], by_epoch["val_acc"] * 100,
                marker="s", label="val_acc")
    if "val_f1" in by_epoch:
        ax.plot(by_epoch["epoch"], by_epoch["val_f1"] * 100,
                marker="^", label="val_f1")
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric (%)")
    ax.set_title("Accuracy / F1")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.suptitle("Training curves", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_confusion_matrix(cm: np.ndarray, classes: list[str],
                          out_path: Path) -> None:
    cm = cm.astype(int)
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    titles = ("Counts", "Row-normalized")
    matrices = (cm, norm)
    fmts = ("d", ".2f")

    for ax, mat, title, fmt in zip(axes, matrices, titles, fmts):
        im = ax.imshow(mat, cmap="Blues", vmin=0,
                       vmax=mat.max() if mat.max() > 0 else 1)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(title)
        thr = mat.max() / 2 if mat.max() > 0 else 0.5
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                ax.text(j, i, format(v, fmt),
                        ha="center", va="center",
                        color="white" if v > thr else "black",
                        fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Confusion matrix (validation set)", fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_misclassified(images: list[np.ndarray], titles: list[str],
                       out_path: Path, max_imgs: int = 16) -> None:
    if not images:
        return
    n = min(len(images), max_imgs)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        if i < n:
            ax.imshow(images[i])
            ax.set_title(titles[i], fontsize=9)
        ax.axis("off")
    fig.suptitle("Hardest validation mistakes (by predicted-class confidence)",
                 fontweight="bold")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _unnormalize(t: torch.Tensor) -> np.ndarray:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = (t.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
    return img


def evaluate_on_val(run_dir: Path, cfg: dict, ckpt_path: Path) -> dict:
    # Local import keeps the script importable even when training script changes.
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from train_fire import FireClassifier, build_dataloaders  # type: ignore

    # PyTorch 2.6 flipped torch.load default to weights_only=True, which
    # rejects Lightning checkpoints (numpy scalars, etc.). Force the old
    # behavior for this trusted local file. (See feedback_pytorch26_load.md.)
    _orig_torch_load = torch.load

    def _trusting_load(*args, **kwargs):
        # Lightning's loader passes weights_only=True explicitly, so we
        # have to overwrite (not setdefault) for the patch to take effect.
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _trusting_load  # type: ignore[assignment]
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FireClassifier.load_from_checkpoint(
            str(ckpt_path), cfg=cfg, map_location=device, strict=False,
        )
    finally:
        torch.load = _orig_torch_load  # type: ignore[assignment]
    model.eval()
    model.to(device)

    data = build_dataloaders(cfg)
    val_loader: DataLoader = data.val
    n = len(data.classes)

    if n == 2:
        acc  = BinaryAccuracy().to(device)
        f1   = BinaryF1Score().to(device)
        prec = BinaryPrecision().to(device)
        rec  = BinaryRecall().to(device)
        cm   = BinaryConfusionMatrix().to(device)
    else:
        acc  = MulticlassAccuracy(num_classes=n, average="micro").to(device)
        f1   = MulticlassF1Score(num_classes=n, average="macro").to(device)
        prec = MulticlassPrecision(num_classes=n, average=None).to(device)
        rec  = MulticlassRecall(num_classes=n, average=None).to(device)
        cm   = MulticlassConfusionMatrix(num_classes=n).to(device)

    all_probs:  list[torch.Tensor] = []
    all_preds:  list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_imgs:   list[torch.Tensor] = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            acc.update(preds, y)
            f1.update(preds, y)
            prec.update(preds, y)
            rec.update(preds, y)
            cm.update(preds, y)
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())
            all_imgs.append(x.cpu())

    probs_t = torch.cat(all_probs)
    preds_t = torch.cat(all_preds)
    labels_t = torch.cat(all_labels)
    imgs_t = torch.cat(all_imgs)

    prec_val = prec.compute()
    rec_val  = rec.compute()
    metrics = {
        "val_acc":       float(acc.compute()),
        "val_f1":        float(f1.compute()),
        # For multiclass we store the per-class vector; for binary it stays scalar.
        "val_precision": prec_val.cpu().tolist() if prec_val.dim() else float(prec_val),
        "val_recall":    rec_val.cpu().tolist()  if rec_val.dim()  else float(rec_val),
        "confusion":     cm.compute().cpu().numpy(),
        "n_val":         int(labels_t.numel()),
        "classes":       data.classes,
    }

    # Hardest mistakes: misclassified samples ranked by the confidence the
    # model placed on the (wrong) predicted class.
    wrong_mask = preds_t != labels_t
    if wrong_mask.any():
        wrong_idx = wrong_mask.nonzero(as_tuple=True)[0]
        confs = probs_t[wrong_idx].gather(1, preds_t[wrong_idx].unsqueeze(1)).squeeze(1)
        order = torch.argsort(confs, descending=True)[:16]
        sel = wrong_idx[order]
        miss_images = [_unnormalize(imgs_t[i]) for i in sel]
        miss_titles = [
            f"true={data.classes[int(labels_t[i])]}  "
            f"pred={data.classes[int(preds_t[i])]} ({float(probs_t[i, preds_t[i]]):.2f})"
            for i in sel
        ]
    else:
        miss_images, miss_titles = [], []

    metrics["miss_images"] = miss_images
    metrics["miss_titles"] = miss_titles
    return metrics


def write_report(run_dir: Path, cfg: dict, ckpt_path: Path,
                 metrics: dict, plot_paths: dict) -> Path:
    cm = metrics["confusion"]
    classes = metrics["classes"]
    n_val = metrics["n_val"]
    n = len(classes)
    # cm[i, j] = count of true=i, pred=j

    md = []
    md.append(f"# Fire classifier — training report\n")
    md.append(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n")
    md.append("## Run\n")
    md.append(f"- **Run name:** `{cfg.get('run_name')}`")
    md.append(f"- **Run dir:** `{run_dir.relative_to(REPO)}`")
    md.append(f"- **Best checkpoint:** `{ckpt_path.relative_to(REPO)}`")
    md.append(f"- **Classes (alphabetical):** {classes}")
    md.append("")

    md.append("## Model & training config\n")
    md.append(f"- **Architecture:** {cfg.get('model')} (torchvision, "
              f"ImageNet pretrained={cfg.get('pretrained')})")
    md.append(f"- **Input size:** {cfg.get('imgsz')} × {cfg.get('imgsz')}")
    md.append(f"- **Batch size:** {cfg.get('batch_size')}")
    md.append(f"- **Optimizer:** {cfg.get('optimizer')} "
              f"(lr={cfg.get('lr')}, weight_decay={cfg.get('weight_decay')}, "
              f"warmup={cfg.get('warmup_epochs')} epochs, cos_lr={cfg.get('cos_lr')})")
    md.append(f"- **Label smoothing:** {cfg.get('label_smoothing')}")
    md.append(f"- **Augmentations:** RandomCrop, HFlip(p={cfg.get('hflip')}), "
              f"ColorJitter({cfg.get('color_jitter')}), "
              f"RandomErasing(p={cfg.get('random_erasing')})")
    md.append(f"- **Precision:** {cfg.get('precision')}")
    md.append(f"- **Epochs configured:** {cfg.get('epochs')} "
              f"(early-stop patience={cfg.get('patience')})")
    md.append("")

    md.append("## Final validation metrics (best checkpoint)\n")
    md.append(f"- **Accuracy:**  {metrics['val_acc']*100:.2f} %")
    md.append(f"- **F1 (macro):** {metrics['val_f1']*100:.2f} %")
    md.append(f"- **N(val):**    {n_val}")
    md.append("")

    md.append("### Per-class precision / recall\n")
    md.append("| class | precision % | recall % | support |")
    md.append("| ----- | ----------: | -------: | ------: |")
    if isinstance(metrics["val_precision"], list):
        precs, recs = metrics["val_precision"], metrics["val_recall"]
    else:
        # Binary: precision/recall refer to class 1.
        precs = [1.0 - metrics["val_precision"], metrics["val_precision"]]
        recs  = [1.0 - metrics["val_recall"],    metrics["val_recall"]]
    for i, c in enumerate(classes):
        support = int(cm[i].sum())
        md.append(f"| `{c}` | {precs[i]*100:.2f} | {recs[i]*100:.2f} | {support} |")
    md.append("")

    md.append("## Confusion matrix (validation set)\n")
    md.append("Rows = true class, columns = predicted class.\n")
    header = "| true \\ pred | " + " | ".join(f"`{c}`" for c in classes) + " |"
    sep    = "| ----------- | " + " | ".join("--------:" for _ in classes) + " |"
    md.append(header)
    md.append(sep)
    for i, c in enumerate(classes):
        row = " | ".join(str(int(cm[i, j])) for j in range(n))
        md.append(f"| **`{c}`** | {row} |")
    md.append("")

    md.append(f"![Confusion matrix]({plot_paths['cm'].relative_to(run_dir).as_posix()})\n")
    md.append(f"![Training curves]({plot_paths['curves'].relative_to(run_dir).as_posix()})\n")
    if "miss" in plot_paths:
        md.append(f"![Misclassified samples]"
                  f"({plot_paths['miss'].relative_to(run_dir).as_posix()})\n")

    md.append("## Interpretation\n")
    msgs = []
    if isinstance(metrics["val_precision"], list):
        # Multiclass: highlight the worst-performing class.
        worst = int(np.argmin(metrics["val_recall"]))
        msgs.append(f"- Lowest-recall class: `{classes[worst]}` at "
                    f"{metrics['val_recall'][worst]*100:.1f}% recall. "
                    "Adding harder examples for this class (or class re-weighting "
                    "in the loss) is the first lever to try.")
    else:
        if metrics["val_recall"] < metrics["val_precision"] - 0.02:
            msgs.append("- Recall lags precision: the model misses some real fires. "
                        "For a safety system this is the costlier error — consider "
                        "lowering the decision threshold for `fire`, oversampling "
                        "the `fire` class, or adding harder positive examples.")
        elif metrics["val_precision"] < metrics["val_recall"] - 0.02:
            msgs.append("- Precision lags recall: the model raises false fire "
                        "alerts. Adding harder `no_fire` examples (sunsets, "
                        "warm-tinted indoor scenes, smoke-like fog) would help.")
        else:
            msgs.append("- Precision and recall are balanced.")
    if metrics["val_acc"] >= 0.95:
        msgs.append("- Validation accuracy ≥ 95 %: with only 1000 images, "
                    "validate on the held-out test split when available before "
                    "trusting this number.")
    elif metrics["val_acc"] < 0.85:
        msgs.append("- Validation accuracy < 85 %: consider longer training, "
                    "a larger backbone, or more / cleaner training data.")
    md.extend(msgs)
    md.append("")

    md.append("## Deployment notes (Jetson Orin)\n")
    md.append("- Export to TorchScript or ONNX before deploying to the Jetson "
              "Orin; FP16 or INT8 quantization is recommended for real-time "
              "inference on the Orin's GPU.")
    md.append("- Inference preprocessing must match training: "
              f"Resize({int(cfg.get('imgsz', 224) * 1.15)}) → "
              f"CenterCrop({cfg.get('imgsz')}) → "
              "Normalize(ImageNet mean/std).")
    md.append("")

    out_path = run_dir / "REPORT.md"
    out_path.write_text("\n".join(md))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(REPO / "models/fire/swin_b_v1"))
    ap.add_argument("--ckpt", default=None,
                    help="Override best checkpoint path.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    cfg = _load_yaml(run_dir / "config_used.yaml")
    if not cfg:
        cfg = _load_yaml(REPO / "configs" / "train_fire.yaml")

    ckpt_path = Path(args.ckpt) if args.ckpt else find_best_ckpt(run_dir)
    print(f"[report] run_dir={run_dir}")
    print(f"[report] ckpt={ckpt_path}")

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    curves_path = plots_dir / "training_curves.png"
    if (run_dir / "metrics.csv").exists():
        plot_training_curves(run_dir / "metrics.csv", curves_path)
        print(f"[report] wrote {curves_path}")
    else:
        print("[report] no metrics.csv — skipping training curves.")

    metrics = evaluate_on_val(run_dir, cfg, ckpt_path)

    cm_path = plots_dir / "confusion_matrix.png"
    plot_confusion_matrix(metrics["confusion"], metrics["classes"], cm_path)
    print(f"[report] wrote {cm_path}")

    plot_paths = {"cm": cm_path, "curves": curves_path}
    if metrics["miss_images"]:
        miss_path = plots_dir / "misclassified.png"
        plot_misclassified(metrics["miss_images"], metrics["miss_titles"],
                           miss_path)
        plot_paths["miss"] = miss_path
        print(f"[report] wrote {miss_path}")

    report_path = write_report(run_dir, cfg, ckpt_path, metrics, plot_paths)
    print(f"[report] wrote {report_path}")
    print(f"[report] val_acc={metrics['val_acc']:.4f}  "
          f"val_f1={metrics['val_f1']:.4f}  "
          f"val_precision={metrics['val_precision']:.4f}  "
          f"val_recall={metrics['val_recall']:.4f}")


if __name__ == "__main__":
    main()
