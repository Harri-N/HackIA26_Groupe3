"""Extract clean backbone weights (.pt) from Lightning checkpoints.

For each `models/fire/*fire4*` run, reads `best.json`, opens the best
Lightning ckpt, strips the `model.` prefix from every state_dict key so
the resulting file is loadable straight into a fresh torchvision backbone:

    backbone = torchvision.models.resnet50()
    backbone.fc = nn.Linear(2048, 3)
    backbone.load_state_dict(torch.load("weights.pt"))   # no rename needed

Drops anything outside `model.` (e.g. `criterion.*`, `train_acc.*`).

Output: `<run_dir>/weights.pt`

Run:
    python src/export_weights.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO / "models" / "fire"


def export_one(run_dir: Path) -> None:
    best = run_dir / "best.json"
    if not best.exists():
        print(f"[skip] {run_dir.name}: no best.json")
        return
    ckpt_path = Path(json.loads(best.read_text())["best_ckpt"])
    if not ckpt_path.is_file():
        print(f"[skip] {run_dir.name}: ckpt missing at {ckpt_path}")
        return

    # PyTorch 2.6 default weights_only=True trips on Lightning ckpts.
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    PREFIX = "model."
    cleaned: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for k, v in sd.items():
        if k.startswith(PREFIX):
            cleaned[k[len(PREFIX):]] = v
        else:
            dropped.append(k)

    out = run_dir / "weights.pt"
    torch.save(cleaned, str(out))
    n_params = sum(v.numel() for v in cleaned.values()) / 1e6
    size_mb = out.stat().st_size / 1e6
    print(f"[ok]  {run_dir.name}: kept {len(cleaned)} tensors "
          f"({n_params:.2f}M params, {size_mb:.1f} MB) -> {out.name}"
          + (f"  dropped {len(dropped)} non-backbone keys" if dropped else ""))


def main() -> None:
    targets = sorted(p for p in MODELS_DIR.iterdir()
                     if p.is_dir() and "fire4" in p.name)
    if not targets:
        raise SystemExit(f"No *fire4* runs under {MODELS_DIR}")
    print(f"[export] {len(targets)} run(s) to process")
    for run_dir in targets:
        export_one(run_dir)


if __name__ == "__main__":
    main()
