"""
Pointing Game evaluation for the Swin Fire classifier on D-Fire bboxes.

Protocol (Zhang et al., 2018):
  For each image with at least one ground-truth bbox of the target class,
  compute a Grad-CAM heatmap for that class, find the pixel of maximum
  activation, and count a hit if that pixel falls inside any GT bbox of
  the target class.

      pointing_game_acc = hits / total_images_with_target_class

D-Fire labels are YOLO-format: `class cx cy w h` (normalized [0, 1]).
  class 0 = smoke
  class 1 = fire

The fire/no-fire classifier was trained on a different dataset; here we
ask "where does the Swin classifier think the fire is, when it sees a
D-Fire image?" — i.e. does its explanation align with the FireZone-style
fire annotations?

Usage:
    conda activate hackia
    python src/pointing_game.py                          # defaults
    python src/pointing_game.py --split valid --n-vis 12 # also dump examples
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from explain_fire import (  # noqa: E402
    SwinGradCAM, find_best_ckpt, load_fire_model, preprocess_pil,
    overlay_cam_on_image, build_gradcam,
)

DFIRE_NAMES = {0: "smoke", 1: "fire"}
DFIRE_ROOT = REPO / "data" / "firezone" / "D-Fire"


def parse_yolo_label(label_path: Path, img_w: int, img_h: int
                     ) -> list[tuple[int, int, int, int, int]]:
    """Return list of (class, x1, y1, x2, y2) in absolute pixels."""
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []
    out = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(float(parts[0]))
        cx, cy, w, h = map(float, parts[1:])
        x1 = int((cx - w / 2) * img_w)
        y1 = int((cy - h / 2) * img_h)
        x2 = int((cx + w / 2) * img_w)
        y2 = int((cy + h / 2) * img_h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        if x2 > x1 and y2 > y1:
            out.append((cls, x1, y1, x2, y2))
    return out


def cropped_bbox_to_view(bbox: tuple[int, int, int, int],
                         orig_w: int, orig_h: int,
                         imgsz: int) -> tuple[int, int, int, int] | None:
    """Map a bbox from the original image space into the CenterCrop view
    used by the classifier.

    The preprocessing is `Resize(int(imgsz*1.15)) -> CenterCrop(imgsz)`.
    `Resize` keeps the aspect ratio (resizes the *shorter* side to the
    target). Returns None if the bbox falls fully outside the crop.
    """
    short = min(orig_w, orig_h)
    scale = int(imgsz * 1.15) / short
    new_w = round(orig_w * scale)
    new_h = round(orig_h * scale)
    # Center crop offsets
    off_x = (new_w - imgsz) // 2
    off_y = (new_h - imgsz) // 2

    x1, y1, x2, y2 = bbox
    rx1 = int(round(x1 * scale)) - off_x
    ry1 = int(round(y1 * scale)) - off_y
    rx2 = int(round(x2 * scale)) - off_x
    ry2 = int(round(y2 * scale)) - off_y
    # Clip to the view.
    rx1, ry1 = max(0, rx1), max(0, ry1)
    rx2, ry2 = min(imgsz, rx2), min(imgsz, ry2)
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    return rx1, ry1, rx2, ry2


def point_in_any_bbox(px: int, py: int,
                      bboxes: list[tuple[int, int, int, int]]) -> bool:
    return any(x1 <= px < x2 and y1 <= py < y2 for x1, y1, x2, y2 in bboxes)


def run_pointing_game(model, cfg: dict, split: str, target_class: int,
                      cam_class: int,
                      device: torch.device, max_images: int | None,
                      n_vis: int, out_dir: Path) -> dict:
    """Iterate D-Fire images that contain `target_class` (D-Fire bbox class);
    count CAM-argmax hits inside any GT bbox of that class (within the
    model's view). The CAM is explained for `cam_class` of the trained
    classifier — typically the model index whose semantics match the
    D-Fire target class.
    """
    img_dir   = DFIRE_ROOT / "images" / split
    label_dir = DFIRE_ROOT / "labels" / split
    if not img_dir.is_dir():
        raise FileNotFoundError(f"D-Fire split not found: {img_dir}")

    imgsz = int(cfg["imgsz"])
    image_paths = sorted(p for p in img_dir.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    hits = 0
    total = 0
    skipped_outside_crop = 0
    vis_examples: list[dict] = []
    t0 = time.time()

    cam_fn = build_gradcam(model)
    try:
        for k, img_path in enumerate(image_paths):
            if max_images and total >= max_images:
                break
            label_path = label_dir / (img_path.stem + ".txt")
            try:
                with Image.open(img_path) as im:
                    orig_w, orig_h = im.size
                    bboxes = parse_yolo_label(label_path, orig_w, orig_h)
            except (OSError, ValueError):
                continue

            tgt_bboxes_orig = [b for b in bboxes if b[0] == target_class]
            if not tgt_bboxes_orig:
                continue
            # Map to the model's crop space.
            tgt_view = []
            for _, x1, y1, x2, y2 in tgt_bboxes_orig:
                m = cropped_bbox_to_view((x1, y1, x2, y2),
                                         orig_w, orig_h, imgsz)
                if m is not None:
                    tgt_view.append(m)
            if not tgt_view:
                # The crop dropped all target bboxes — exclude from denom.
                skipped_outside_crop += 1
                continue

            with Image.open(img_path) as im:
                pil = im.convert("RGB")
                x = preprocess_pil(pil, imgsz).unsqueeze(0).to(device)
            cam = cam_fn(x, target_class=cam_class,
                         upsample_to=(imgsz, imgsz))[0].cpu().numpy()
            py, px = np.unravel_index(int(cam.argmax()), cam.shape)
            hit = point_in_any_bbox(int(px), int(py), tgt_view)
            hits  += int(hit)
            total += 1

            if len(vis_examples) < n_vis:
                # Save (path, view image, cam, bboxes_in_view, hit, click) for plotting.
                from torchvision import transforms as T
                view = T.Compose([
                    T.Resize(int(imgsz * 1.15)),
                    T.CenterCrop(imgsz),
                ])(pil)
                vis_examples.append({
                    "path": str(img_path),
                    "view": np.asarray(view).astype(np.float32) / 255.0,
                    "cam": cam,
                    "bboxes": tgt_view,
                    "hit": hit,
                    "px": int(px), "py": int(py),
                })

            if (k + 1) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  scanned {k+1}/{len(image_paths)}  "
                      f"used={total}  hits={hits}  "
                      f"acc={hits / max(1, total):.4f}  "
                      f"{elapsed:.0f}s")
    finally:
        cam_fn.close()

    elapsed = time.time() - t0
    classes_model = cfg.get("class_names", [])
    cam_class_name = (classes_model[cam_class]
                      if cam_class < len(classes_model) else str(cam_class))
    result = {
        "split":              split,
        "dfire_class_idx":    target_class,
        "dfire_class_name":   DFIRE_NAMES[target_class],
        "model_cam_class":    cam_class,
        "model_cam_class_name": cam_class_name,
        "n_images_scanned":   len(image_paths),
        "n_images_with_tgt":  total + skipped_outside_crop,
        "n_evaluated":        total,
        "n_skipped_crop":     skipped_outside_crop,
        "hits":               hits,
        "pointing_game_acc":  hits / total if total > 0 else 0.0,
        "elapsed_sec":        elapsed,
    }
    print(f"[pg] dfire=`{result['dfire_class_name']}` vs "
          f"model_cam=`{cam_class_name}`  "
          f"hits={hits}/{total}  "
          f"acc={result['pointing_game_acc']:.4f}  "
          f"({skipped_outside_crop} skipped — bbox lost in CenterCrop)")

    # Dump example visualizations.
    if vis_examples:
        cols = 4
        rows = (len(vis_examples) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.6))
        axes = np.atleast_2d(axes)
        for i, ex in enumerate(vis_examples):
            ax = axes[i // cols, i % cols]
            ax.imshow(overlay_cam_on_image(ex["view"], ex["cam"]))
            for x1, y1, x2, y2 in ex["bboxes"]:
                ax.add_patch(patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, fill=False,
                    edgecolor="lime", linewidth=2,
                ))
            ax.scatter([ex["px"]], [ex["py"]],
                       c="white", s=80, edgecolors="black", linewidths=1.5,
                       marker="x")
            tag = "HIT ✓" if ex["hit"] else "MISS ✗"
            color = "green" if ex["hit"] else "red"
            ax.set_title(f"{Path(ex['path']).name}\n{tag}",
                         fontsize=9, color=color)
            ax.axis("off")
        for j in range(len(vis_examples), rows * cols):
            axes[j // cols, j % cols].axis("off")
        fig.suptitle(
            f"Pointing Game — CAM(`{cam_class_name}`) vs D-Fire "
            f"`{result['dfire_class_name']}` bboxes ({split})\n"
            f"acc = {result['pointing_game_acc']:.3f}  "
            f"({hits}/{total})",
            fontweight="bold",
        )
        fig.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = (out_dir
                    / f"pointing_game_{split}_{DFIRE_NAMES[target_class]}.png")
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"[pg] wrote {out_path}")
        result["examples_png"] = str(out_path)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",
                    default=str(REPO / "models/fire/swin_b_forest_v1"))
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="valid",
                    choices=["train", "valid", "test"])
    ap.add_argument("--targets", default="fire",
                    help="Comma-separated D-Fire classes to evaluate: "
                         "fire,smoke (default: fire).")
    ap.add_argument("--mapping", default=None,
                    help="D-Fire -> model class mapping, "
                         "e.g. `fire:fire,smoke:start_fire`. "
                         "If omitted: each D-Fire class is matched to the "
                         "model class of the same name when present, else "
                         "to `fire` for 2-class models.")
    ap.add_argument("--max-images", type=int, default=None,
                    help="Cap evaluation to N images per class (for sanity runs).")
    ap.add_argument("--n-vis", type=int, default=12,
                    help="Number of example overlays to render per class.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    with (run_dir / "config_used.yaml").open() as f:
        cfg = yaml.safe_load(f)
    ckpt = Path(args.ckpt) if args.ckpt else find_best_ckpt(run_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pg] device={device}  ckpt={ckpt.name}  split={args.split}")
    model = load_fire_model(ckpt, cfg, device)

    name_to_idx = {v: k for k, v in DFIRE_NAMES.items()}
    classes_model = cfg.get("class_names", [])
    model_idx = {c: i for i, c in enumerate(classes_model)}

    # Build dfire -> model_class_idx mapping.
    if args.mapping:
        pairs = [p.split(":") for p in args.mapping.split(",")]
        mapping = {a.strip(): b.strip() for a, b in pairs}
    else:
        mapping = {}
        for d in DFIRE_NAMES.values():
            # 1) Same-name match (works for 2-class fire/no_fire model w/ fire bboxes).
            if d in model_idx:
                mapping[d] = d
            # 2) Heuristic for the 3-class model (fire / no_fire / start_fire).
            elif d == "smoke" and "start_fire" in model_idx:
                mapping[d] = "start_fire"
            elif d == "fire":
                mapping[d] = "fire" if "fire" in model_idx else classes_model[-1]

    print(f"[pg] model classes={classes_model}")
    print(f"[pg] D-Fire -> model CAM mapping: {mapping}")

    out_dir = run_dir / "plots" / "pointing_game"
    results = []
    for tgt in [t.strip() for t in args.targets.split(",") if t.strip()]:
        if tgt not in name_to_idx:
            print(f"[pg] unknown target `{tgt}` (choose from {list(name_to_idx)})")
            continue
        cam_class_name = mapping.get(tgt)
        if cam_class_name not in model_idx:
            print(f"[pg] mapping for `{tgt}` -> `{cam_class_name}` not in "
                  f"model classes {classes_model}. Skipping.")
            continue
        cam_class = model_idx[cam_class_name]
        print(f"\n[pg] evaluating D-Fire=`{tgt}` (idx {name_to_idx[tgt]}) "
              f"vs CAM(model=`{cam_class_name}`, idx {cam_class})")
        r = run_pointing_game(
            model, cfg, args.split, name_to_idx[tgt], cam_class,
            device, args.max_images, args.n_vis, out_dir,
        )
        results.append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"pointing_game_{args.split}.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\n[pg] wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
