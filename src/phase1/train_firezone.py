"""
Fine-tune YOLO26 on the D-Fire dataset (fire/smoke localization).

All hyperparameters live in `configs/train_firezone.yaml`. CLI flags override.

Target hardware: NVIDIA RTX 4090 (24 GB).
Conda env: hackia.

Run:
    conda activate hackia
    python src/train_firezone.py                       # use config defaults
    python src/train_firezone.py --epochs 30 --batch 64
    python src/train_firezone.py --config configs/train_firezone.yaml --resume

Live monitoring (separate terminal):
    python src/monitor_firezone.py
    tensorboard --logdir models/firezone/yolo26m_dfire
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml


def _enable_safe_workers() -> None:
    """Make `workers > 0` survive Python 3.13's multiprocessing IPC quirks.

    Two changes from Ultralytics' defaults:
      - sharing strategy `file_system` instead of `fd` (avoids `ConnectionResetError`
        in `multiprocessing.resource_sharer`).
      - `pin_memory=False` + `prefetch_factor=2` to keep /dev/shm usage well under
        the system's 1 GB cap (default would be `pin_memory=True, prefetch=4`).
    Empirically `workers=2` is stable in this env; `workers≥3` still crashes.
    """
    import torch.multiprocessing as mp
    mp.set_sharing_strategy("file_system")

    from ultralytics.data import build as udb
    _orig = udb.build_dataloader

    def _safe_build_dataloader(*args, **kwargs):
        kwargs["pin_memory"] = False
        dl = _orig(*args, **kwargs)
        # Adjust the prefetch factor on the (now-built) loader's internal state
        # — the underlying torch DataLoader exposes it as a read-only attribute,
        # so we set it before workers spawn by mutating the kwargs path instead.
        return dl

    # Monkey-patch via a wrapper that also caps prefetch_factor at the source.
    import torch.utils.data.dataloader as tdl
    _orig_dl_init = tdl.DataLoader.__init__

    def _patched_init(self, *args, **kwargs):
        if kwargs.get("num_workers", 0) > 0:
            kwargs["prefetch_factor"] = 2
            kwargs["pin_memory"] = False
        return _orig_dl_init(self, *args, **kwargs)

    tdl.DataLoader.__init__ = _patched_init
    udb.build_dataloader = _safe_build_dataloader

REPO = Path(__file__).resolve().parents[1]
DEFAULT_CFG = REPO / "configs" / "train_firezone.yaml"

# Keys that ultralytics' .train() accepts directly; everything else stays in
# our own namespace (name, project_dir, …) and is handled below.
TRAIN_KEYS = {
    "epochs", "imgsz", "batch", "device", "workers", "cache", "amp", "seed",
    "patience", "close_mosaic", "save_period", "plots", "verbose",
    "optimizer", "lr0", "lrf", "cos_lr", "momentum", "weight_decay",
    "warmup_epochs", "warmup_momentum", "warmup_bias_lr",
    "box", "cls", "dfl",
    "hsv_h", "hsv_s", "hsv_v",
    "degrees", "translate", "scale", "shear", "perspective",
    "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
    "erasing", "auto_augment",
}


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(DEFAULT_CFG),
                   help="Path to YAML config file.")
    p.add_argument("--resume", action="store_true")
    # Quick overrides (the rest go in YAML).
    for key, typ in [("model", str), ("name", str),
                     ("epochs", int), ("imgsz", int), ("batch", int),
                     ("patience", int), ("workers", int),
                     ("cache", str), ("device", str)]:
        p.add_argument(f"--{key}", type=typ, default=None)
    return p.parse_args()


def main():
    args = build_args()
    cfg = load_config(Path(args.config))

    # CLI overrides any YAML field of the same name.
    for k, v in vars(args).items():
        if k in {"config", "resume"} or v is None:
            continue
        cfg[k] = v

    project_dir = REPO / cfg.pop("project_dir", "models/firezone")
    project_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = REPO / cfg.pop("data")
    model_weights = cfg.pop("model")
    run_name = cfg.pop("name")

    import torch
    if int(cfg.get("workers", 0)) > 0:
        _enable_safe_workers()
    from ultralytics import YOLO

    assert torch.cuda.is_available(), "CUDA GPU required."
    print(f"[init] torch={torch.__version__} cuda={torch.version.cuda} "
          f"device={torch.cuda.get_device_name(0)} "
          f"vram_gb={torch.cuda.get_device_properties(0).total_memory/1e9:.1f}")
    print(f"[cfg ] loaded {args.config}")

    model = YOLO(model_weights)

    train_kwargs = {k: v for k, v in cfg.items() if k in TRAIN_KEYS}
    unknown = set(cfg) - TRAIN_KEYS
    if unknown:
        print(f"[cfg ] unused keys (passed through anyway): {sorted(unknown)}")
        train_kwargs.update({k: cfg[k] for k in unknown})

    train_kwargs.update(
        data=str(data_yaml),
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        resume=args.resume,
        save=True,
    )

    print("[train] kwargs:")
    for k, v in sorted(train_kwargs.items()):
        print(f"  {k}={v}")

    results = model.train(**train_kwargs)
    print(f"[train] best weights: {results.save_dir}/weights/best.pt")

    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        print("\n[test] evaluating on test split…")
        model = YOLO(str(best_pt))
        test_metrics = model.val(
            data=str(data_yaml),
            split="test",
            imgsz=train_kwargs["imgsz"],
            batch=train_kwargs["batch"],
            device=train_kwargs.get("device", "0"),
            workers=train_kwargs.get("workers", 0),
            project=str(project_dir),
            name=f"{run_name}_test",
            plots=True,
            verbose=True,
        )
        print(f"[test] mAP50={test_metrics.box.map50:.4f} "
              f"mAP50-95={test_metrics.box.map:.4f}")

        stable = project_dir / "best.pt"
        try:
            shutil.copy2(best_pt, stable)
            print(f"[save] copied to {stable}")
        except OSError as e:
            print(f"[save] copy failed: {e}")


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
