"""
Grad-CAM explainability for the Swin Fire classifier.

The torchvision Swin backbone produces a (B, H', W', C) tensor right before
the final LayerNorm (`model.norm`). For swin_b at 224 input, H'=W'=7. We
hook the *output* of `model.norm` for activations and grab gradients via a
retained tensor's `.grad`, then build the CAM as
    ReLU( sum_c (alpha_c * A_c) )       with alpha_c = mean over spatial of d y_target / d A
which is the standard Grad-CAM formulation adapted to the channels-last
tensor layout of torchvision's Swin.

Usage as a script — produces overlay PNGs for a few validation samples:
    conda activate hackia
    python src/explain_fire.py
    python src/explain_fire.py --run-dir models/fire/swin_b_forest_v1 --n 12

Usage as a library: import `load_fire_model` and `SwinGradCAM` from here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ----------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------
def _patch_torch_load_for_lightning():
    """PyTorch 2.6 defaults weights_only=True which breaks Lightning ckpts."""
    orig = torch.load

    def _trusting(*args, **kwargs):
        kwargs["weights_only"] = False
        return orig(*args, **kwargs)
    torch.load = _trusting  # type: ignore[assignment]
    return orig


def load_fire_model(ckpt_path: Path, cfg: dict | None = None,
                    device: str | torch.device = "cuda") -> nn.Module:
    """Load a trained FireClassifier from a Lightning checkpoint."""
    from train_fire import FireClassifier  # local import — same env

    if cfg is None:
        cfg_path = Path(ckpt_path).resolve().parents[1] / "config_used.yaml"
        if cfg_path.exists():
            with cfg_path.open() as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = {}

    orig_load = _patch_torch_load_for_lightning()
    try:
        model = FireClassifier.load_from_checkpoint(
            str(ckpt_path), cfg=cfg, map_location=device, strict=False,
        )
    finally:
        torch.load = orig_load  # type: ignore[assignment]
    model.eval().to(device)
    return model


def find_best_ckpt(run_dir: Path) -> Path:
    bj = run_dir / "best.json"
    if bj.exists():
        p = Path(json.loads(bj.read_text()).get("best_ckpt", ""))
        if p.is_file():
            return p
    cands = sorted((run_dir / "checkpoints").glob("best-*.ckpt"))
    if not cands:
        cands = sorted((run_dir / "checkpoints").glob("*.ckpt"))
    if not cands:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    return cands[-1]


# ----------------------------------------------------------------------
# Grad-CAM
# ----------------------------------------------------------------------
class SwinGradCAM:
    """Grad-CAM for torchvision's SwinTransformer (`swin_t`/`s`/`b`/`v2_*`).

    Hooks the output of `model.model.norm` (shape (B, H', W', C)) and the
    gradients of that tensor w.r.t. a chosen output class. Returns CAMs
    upsampled to the input image resolution.
    """

    def __init__(self, lightning_module: nn.Module):
        # FireClassifier wraps the torchvision SwinTransformer as `.model`.
        if hasattr(lightning_module, "model"):
            self.backbone = lightning_module.model
        else:
            self.backbone = lightning_module
        if not hasattr(self.backbone, "norm"):
            raise AttributeError(
                "Expected a torchvision SwinTransformer (with `.norm`); "
                f"got {type(self.backbone).__name__}.")
        self.module = lightning_module
        self._activations: torch.Tensor | None = None
        self._gradients:   torch.Tensor | None = None
        self._fwd_hook = self.backbone.norm.register_forward_hook(self._save_act)

    def _save_act(self, _module, _inp, out: torch.Tensor) -> None:
        # `out` shape: (B, H, W, C) — channels-last for Swin.
        # The hook also fires during torch.no_grad() forwards (e.g. when the
        # caller computes predictions). Skip those — we only care about the
        # forward that runs inside __call__ which enables grads explicitly.
        if not out.requires_grad:
            return
        out.retain_grad()
        self._activations = out
        out.register_hook(self._save_grad)

    def _save_grad(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach()

    def __call__(self, x: torch.Tensor, target_class: int,
                 upsample_to: tuple[int, int] | None = None) -> torch.Tensor:
        """Run Grad-CAM. Returns (B, H_out, W_out) tensor in [0, 1]."""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.detach().clone().requires_grad_(True)
        self._activations = None
        self._gradients   = None
        self.backbone.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.module(x)
            # Scalar target = sum of the target-class logit over the batch.
            target = logits[:, target_class].sum()
            target.backward(retain_graph=False)

        act  = self._activations  # (B, H, W, C)
        grad = self._gradients     # (B, H, W, C)
        assert act is not None and grad is not None
        # Channel-importance = spatial mean of gradients per channel.
        weights = grad.mean(dim=(1, 2), keepdim=True)  # (B, 1, 1, C)
        cam = (weights * act).sum(dim=-1)              # (B, H, W)
        cam = F.relu(cam)
        # Normalize each map to [0, 1] for visualization.
        cam_min = cam.flatten(1).min(dim=1).values[:, None, None]
        cam_max = cam.flatten(1).max(dim=1).values[:, None, None]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        if upsample_to is not None:
            cam = F.interpolate(cam.unsqueeze(1),
                                size=upsample_to,
                                mode="bilinear", align_corners=False).squeeze(1)
        return cam.detach()

    def close(self) -> None:
        self._fwd_hook.remove()

    def __enter__(self) -> "SwinGradCAM":
        return self

    def __exit__(self, *_):
        self.close()


class EffNetGradCAM:
    """Grad-CAM for torchvision EfficientNet / EfficientNet-V2.

    Hooks the output of `backbone.features` (the last conv block — a
    (B, C, H, W) channels-first tensor) and the gradients w.r.t. that
    tensor, then assembles the standard Grad-CAM map.
    """

    def __init__(self, lightning_module: nn.Module):
        if hasattr(lightning_module, "model"):
            self.backbone = lightning_module.model
        else:
            self.backbone = lightning_module
        if not hasattr(self.backbone, "features"):
            raise AttributeError(
                "Expected a torchvision EfficientNet (with `.features`); "
                f"got {type(self.backbone).__name__}.")
        self.module = lightning_module
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_hook = self.backbone.features.register_forward_hook(
            self._save_act
        )

    def _save_act(self, _module, _inp, out: torch.Tensor) -> None:
        if not out.requires_grad:
            return
        out.retain_grad()
        self._activations = out
        out.register_hook(self._save_grad)

    def _save_grad(self, grad: torch.Tensor) -> None:
        self._gradients = grad.detach()

    def __call__(self, x: torch.Tensor, target_class: int,
                 upsample_to: tuple[int, int] | None = None) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = x.detach().clone().requires_grad_(True)
        self._activations = None
        self._gradients = None
        self.backbone.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.module(x)
            target = logits[:, target_class].sum()
            target.backward(retain_graph=False)
        act = self._activations   # (B, C, H, W)
        grad = self._gradients    # (B, C, H, W)
        assert act is not None and grad is not None
        weights = grad.mean(dim=(2, 3), keepdim=True)   # (B, C, 1, 1)
        cam = (weights * act).sum(dim=1)                # (B, H, W)
        cam = F.relu(cam)
        cam_min = cam.flatten(1).min(dim=1).values[:, None, None]
        cam_max = cam.flatten(1).max(dim=1).values[:, None, None]
        cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
        if upsample_to is not None:
            cam = F.interpolate(cam.unsqueeze(1), size=upsample_to,
                                mode="bilinear", align_corners=False).squeeze(1)
        return cam.detach()

    def close(self) -> None:
        self._fwd_hook.remove()

    def __enter__(self) -> "EffNetGradCAM":
        return self

    def __exit__(self, *_):
        self.close()


class ResNetGradCAM(EffNetGradCAM):
    """Grad-CAM for torchvision ResNet (-18 / -50 / -101).

    Hooks the output of `backbone.layer4` (last residual stage,
    shape (B, C, H, W) channels-first). The CAM assembly is identical to
    EfficientNet's once the feature map is in (B, C, H, W).
    """

    def __init__(self, lightning_module: nn.Module):
        # Don't call super().__init__ — it hooks `.features` which doesn't
        # exist on ResNet. We replicate the parent attributes manually.
        nn.Module.__init__ if False else None  # noqa: B015 — intentional
        if hasattr(lightning_module, "model"):
            self.backbone = lightning_module.model
        else:
            self.backbone = lightning_module
        if not hasattr(self.backbone, "layer4"):
            raise AttributeError(
                "Expected a torchvision ResNet (with `.layer4`); "
                f"got {type(self.backbone).__name__}.")
        self.module = lightning_module
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._fwd_hook = self.backbone.layer4.register_forward_hook(
            self._save_act
        )


def build_gradcam(lightning_module: nn.Module):
    """Dispatch GradCAM by backbone type. Swin / EfficientNet / ResNet."""
    backbone = (lightning_module.model
                if hasattr(lightning_module, "model") else lightning_module)
    if hasattr(backbone, "norm") and not hasattr(backbone, "classifier"):
        return SwinGradCAM(lightning_module)
    if hasattr(backbone, "features") and hasattr(backbone, "classifier"):
        return EffNetGradCAM(lightning_module)
    if hasattr(backbone, "layer4") and hasattr(backbone, "fc"):
        return ResNetGradCAM(lightning_module)
    raise ValueError(f"No GradCAM available for {type(backbone).__name__}")


# ----------------------------------------------------------------------
# Image utils
# ----------------------------------------------------------------------
def preprocess_pil(img: Image.Image, imgsz: int) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize(int(imgsz * 1.15)),
        transforms.CenterCrop(imgsz),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return tf(img.convert("RGB"))


def overlay_cam_on_image(rgb01: np.ndarray, cam01: np.ndarray,
                         alpha: float = 0.45,
                         cmap: str = "jet") -> np.ndarray:
    """Blend a heatmap onto a [0,1] RGB image. Returns uint8 HWC."""
    cmap_fn = plt.get_cmap(cmap)
    heat = cmap_fn(cam01)[..., :3]  # RGB float
    blended = (1 - alpha) * rgb01 + alpha * heat
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------
# Standalone: produce N example overlays from the val set
# ----------------------------------------------------------------------
def _pick_examples(run_dir: Path, cfg: dict, n: int) -> list[tuple[str, int]]:
    """Pick `n` validation image paths + their true label (idx).

    Balanced across all classes present in the dataset.
    """
    from train_fire import build_dataloaders  # type: ignore

    data = build_dataloaders(cfg)
    val_ds = data.val.dataset  # Subset
    samples = []
    for i in val_ds.indices:  # type: ignore[attr-defined]
        path, label = val_ds.dataset.samples[i]  # type: ignore[attr-defined]
        samples.append((path, int(label)))

    n_classes = len(data.classes)
    per_class = max(1, n // n_classes)
    out: list[tuple[str, int]] = []
    for cls in range(n_classes):
        out.extend([s for s in samples if s[1] == cls][:per_class])
    return out[:n]


def _explain_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",
                    default=str(REPO / "models/fire/swin_b_forest_v1"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n", type=int, default=8,
                    help="Number of example overlays to render.")
    ap.add_argument("--target", choices=["pred", "fire", "true"], default="fire",
                    help="Which class to explain.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    with (run_dir / "config_used.yaml").open() as f:
        cfg = yaml.safe_load(f)
    ckpt = Path(args.ckpt) if args.ckpt else find_best_ckpt(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_fire_model(ckpt, cfg, device)
    imgsz = int(cfg["imgsz"])
    classes = cfg.get("class_names", ["nofire", "fire"])
    print(f"[gradcam] model={cfg.get('model')} ckpt={ckpt.name} imgsz={imgsz}")

    # Map cfg target name to model index for the --target=fire shortcut.
    name_to_idx = {c: i for i, c in enumerate(classes)}
    examples = _pick_examples(run_dir, cfg, args.n)
    counts = {c: sum(1 for _, y in examples if classes[y] == c) for c in classes}
    print(f"[gradcam] picked {len(examples)} val samples — "
          + ", ".join(f"{n} {c}" for c, n in counts.items()))

    out_dir = run_dir / "plots" / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = 4
    rows = (len(examples) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.4))
    axes = np.atleast_2d(axes)

    with SwinGradCAM(model) as cam_fn:
        for idx, (path, y_true) in enumerate(examples):
            img = Image.open(path)
            x = preprocess_pil(img, imgsz).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
                pred = int(probs.argmax())

            if args.target == "fire":
                tgt = name_to_idx.get("fire", len(classes) - 1)
            elif args.target == "true":
                tgt = y_true
            else:
                tgt = pred

            cam = cam_fn(x, target_class=tgt,
                         upsample_to=(imgsz, imgsz))[0].cpu().numpy()

            # Build the "input image" view (same crop as the model saw),
            # then overlay the CAM.
            view = transforms.Compose([
                transforms.Resize(int(imgsz * 1.15)),
                transforms.CenterCrop(imgsz),
            ])(img.convert("RGB"))
            rgb01 = np.asarray(view).astype(np.float32) / 255.0
            overlaid = overlay_cam_on_image(rgb01, cam)

            ax = axes[idx // cols, idx % cols]
            ax.imshow(overlaid)
            tag = "✓" if pred == y_true else "✗"
            ax.set_title(
                f"true={classes[y_true]} "
                f"pred={classes[pred]} ({probs[pred]:.2f}) {tag}\n"
                f"CAM for `{classes[tgt]}`",
                fontsize=9,
            )
            ax.axis("off")

    for j in range(len(examples), rows * cols):
        axes[j // cols, j % cols].axis("off")

    fig.suptitle(f"Grad-CAM — {cfg.get('model')} on val "
                 f"(target class = {args.target})",
                 fontweight="bold")
    fig.tight_layout()
    out_path = out_dir / f"gradcam_val_{args.target}.png"
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"[gradcam] wrote {out_path}")


if __name__ == "__main__":
    _explain_main()
