"""
Build clean_database_v2/: same recipe as build_clean_database.py, plus
the extra images dropped into ./Images/.

Naming convention
-----------------
- Files coming from FIRE_DATABASE_3/ + Videos/ are written as
  `img_NNNNN.jpg` / `img_NNNNN_augK.jpg` (same as v1).
- Files coming from Images/<class>/ are written as
  `extra_NNNNN.jpg` / `extra_NNNNN_augK.jpg`.

The `extra_` prefix lets the training script send those sources to the
train split exclusively, so v1 vs v2 can be compared on an identical
test set.

For the augmentation :Apply the SAME classical, label-preserving augmentations to every
   image (5 variants per source: hflip, rotation, brightness/contrast,
   gaussian noise, blur). Every class is treated identically, so the
   per-class totals stay naturally balanced.

Run: python build_clean_database_v2.py
"""

from __future__ import annotations

import hashlib
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SRC_IMAGES = Path("FIRE_DATABASE_3")
SRC_VIDEOS = Path("Videos")
SRC_EXTRA = Path("Images")
DST_ROOT = Path("clean_database_v2")

CLASSES = ["no_fire", "start_fire", "fire"]

FRAMES_PER_VIDEO = 3
AUGS_PER_IMAGE = 5
RNG_SEED = 42


# ---------------------------------------------------------------------------
# MD5 helpers
# ---------------------------------------------------------------------------

def md5_of_file(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def md5_of_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# ---------------------------------------------------------------------------
# Augmentations (identical to v1)
# ---------------------------------------------------------------------------

def aug_hflip(img, rng):
    return cv2.flip(img, 1)


def aug_rotate(img, rng):
    angle = rng.uniform(-15, 15)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def aug_brightness_contrast(img, rng):
    alpha = rng.uniform(0.8, 1.2)
    beta = rng.uniform(-25, 25)
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def aug_gaussian_noise(img, rng):
    sigma = rng.uniform(5, 15)
    noise = np.random.default_rng(rng.randint(0, 2**31)).normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def aug_blur(img, rng):
    k = rng.choice([3, 5])
    return cv2.GaussianBlur(img, (k, k), 0)


AUG_FUNCS = [aug_hflip, aug_rotate, aug_brightness_contrast,
             aug_gaussian_noise, aug_blur]


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_image_any(path: Path):
    """Read with OpenCV; fall back to PIL for formats cv2 doesn't handle
    (e.g. .avif). Returns a BGR np.ndarray or None on failure."""
    img = cv2.imread(str(path))
    if img is not None:
        return img
    try:
        pil = Image.open(path).convert("RGB")
        arr = np.array(pil)[:, :, ::-1]  # RGB -> BGR
        return np.ascontiguousarray(arr)
    except Exception:
        return None


def collect_unique_images(class_dir: Path,
                          seen_md5: set[str]) -> list[np.ndarray]:
    """Read images from disk, dropping byte-identical duplicates only."""
    if not class_dir.is_dir():
        return []
    kept = []
    files = sorted(class_dir.iterdir())
    exact_dups = 0
    for f in files:
        if not f.is_file():
            continue
        digest = md5_of_file(f)
        if digest in seen_md5:
            exact_dups += 1
            continue
        seen_md5.add(digest)
        img = load_image_any(f)
        if img is None:
            continue
        kept.append(img)
    print(f"  scanned {len(files)} files from {class_dir} -> kept {len(kept)} "
          f"(exact dups dropped: {exact_dups})")
    return kept


def extract_video_frames(class_videos_dir: Path,
                         seen_md5: set[str]) -> list[np.ndarray]:
    if not class_videos_dir.is_dir():
        return []
    frames = []
    for vid_path in sorted(class_videos_dir.iterdir()):
        if not vid_path.is_file():
            continue
        cap = cv2.VideoCapture(str(vid_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            print(f"    {vid_path.name}: unreadable, skipped")
            continue
        targets = [int(total * (i + 1) / (FRAMES_PER_VIDEO + 1))
                   for i in range(FRAMES_PER_VIDEO)]
        added = 0
        for t in targets:
            cap.set(cv2.CAP_PROP_POS_FRAMES, t)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            ok, buf = cv2.imencode(".jpg", frame)
            if not ok:
                continue
            digest = md5_of_bytes(buf.tobytes())
            if digest in seen_md5:
                continue
            seen_md5.add(digest)
            frames.append(frame)
            added += 1
        cap.release()
        print(f"    {vid_path.name}: +{added} frames")
    return frames


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

def write_with_augmentations(images, dst_dir: Path, prefix: str,
                             rng: random.Random) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for idx, img in enumerate(images):
        base = f"{prefix}_{idx:05d}"
        cv2.imwrite(str(dst_dir / f"{base}.jpg"), img)
        written += 1
        for a_idx in range(AUGS_PER_IMAGE):
            fn = AUG_FUNCS[a_idx % len(AUG_FUNCS)]
            aug = fn(img, rng)
            cv2.imwrite(str(dst_dir / f"{base}_aug{a_idx}.jpg"), aug)
            written += 1
    return written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    rng = random.Random(RNG_SEED)
    if DST_ROOT.exists():
        shutil.rmtree(DST_ROOT)

    summary = []
    for cls in CLASSES:
        print(f"\n[{cls}]")
        seen_md5: set[str] = set()

        # 1. Main pool — same as v1.
        print("  - scanning FIRE_DATABASE_3 images...")
        main_imgs = collect_unique_images(SRC_IMAGES / cls, seen_md5)
        print("  - extracting video frames...")
        video_frames = extract_video_frames(SRC_VIDEOS / cls, seen_md5)
        main_imgs.extend(video_frames)

        # 2. Extra pool — only added in v2.
        print("  - scanning Images/ extras...")
        extra_imgs = collect_unique_images(SRC_EXTRA / cls, seen_md5)

        # 3. Write both pools with distinct prefixes.
        n_main = write_with_augmentations(main_imgs, DST_ROOT / cls,
                                          "img", rng)
        n_extra = write_with_augmentations(extra_imgs, DST_ROOT / cls,
                                           "extra", rng)
        print(f"  -> wrote {n_main} `img_*` + {n_extra} `extra_*` "
              f"to {DST_ROOT / cls}")
        summary.append((cls, len(main_imgs), len(extra_imgs),
                        n_main, n_extra, n_main + n_extra))

    print("\n=== Summary ===")
    print(f"{'class':<12} {'main_src':>9} {'extra_src':>10} "
          f"{'main_out':>9} {'extra_out':>10} {'total':>8}")
    for cls, n_ms, n_es, n_mo, n_eo, tot in summary:
        print(f"{cls:<12} {n_ms:>9} {n_es:>10} "
              f"{n_mo:>9} {n_eo:>10} {tot:>8}")


if __name__ == "__main__":
    main()
