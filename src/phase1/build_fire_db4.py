"""Build `data/FIRE_DATABASE_4` — a domain-diversified 3-class fire dataset
with a 70/15/15 train/val/test split (per class).

Why: the previous DB3-only training gave 99 % val / 67 % test. Cause: DB3
has very uniform image style; the test images (from D-Fire) are outside
that distribution. This script mixes DB3 + D-Fire train (sampled, balanced)
so the model sees the broader distribution that deployment will encounter.

Layout produced (ImageFolder-compatible under each split):

    data/FIRE_DATABASE_4/
        train/{no_fire, start_fire, fire}/    1400 / class
        val/{no_fire, start_fire, fire}/       300 / class
        test/{no_fire, start_fire, fire}/      300 / class
        manifest.json

Sources (all class assignments cross-checked by MD5 to prevent leakage):
    train + val : DB3 (~500 / class) + D-Fire train (top-up to 1700 / class)
                  -> stratified 1400 train / 300 val
    test        : DB1 unique-fire + D-Fire test
                  -> never in DB3, never overlap with train / val

Run:
    python src/build_fire_db4.py
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DB1 = REPO / "data" / "FIRE_DATABASE_1"
DB3 = REPO / "data" / "FIRE_DATABASE_3"
DFIRE_TRAIN_IMG = REPO / "data" / "firezone" / "D-Fire" / "images" / "train"
DFIRE_TRAIN_LAB = REPO / "data" / "firezone" / "D-Fire" / "labels" / "train"
DFIRE_TEST_IMG  = REPO / "data" / "firezone" / "D-Fire" / "images" / "test"
DFIRE_TEST_LAB  = REPO / "data" / "firezone" / "D-Fire" / "labels" / "test"
OUT = REPO / "data" / "FIRE_DATABASE_4"

CLASSES = ["no_fire", "start_fire", "fire"]

# 70 / 15 / 15 on a per-class total of 2000:
TRAIN_PER_CLASS = 1400
VAL_PER_CLASS   = 300
TEST_PER_CLASS  = 300
TRAINVAL_PER_CLASS = TRAIN_PER_CLASS + VAL_PER_CLASS  # 1700
SEED = 42


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def dfire_label_classes(lab_dir: Path, p_img: Path) -> set[int]:
    lab = lab_dir / (p_img.stem + ".txt")
    if not lab.exists() or lab.stat().st_size == 0:
        return set()
    out: set[int] = set()
    for ln in lab.read_text().splitlines():
        parts = ln.strip().split()
        if len(parts) == 5:
            out.add(int(float(parts[0])))
    return out


def dfire_class_of(lab_dir: Path, p_img: Path) -> str | None:
    cs = dfire_label_classes(lab_dir, p_img)
    if not cs:
        return "no_fire"
    if cs == {0}:
        return "start_fire"
    if 1 in cs:
        return "fire"
    return None


# ----------------------------------------------------------------------
# Test pool (held-out from training sources by construction + MD5 check)
# ----------------------------------------------------------------------
def collect_test_pool(rng: random.Random,
                      blocked: set[str]) -> dict[str, list[tuple[Path, str]]]:
    """Sources:
        fire       : DB1/fire ∖ DB3  (up to 220 unique)
                     +  D-Fire test fire (top-up to TEST_PER_CLASS)
        no_fire    : D-Fire test  no_label
        start_fire : D-Fire test  smoke_only
    `blocked` = hashes that must not appear in any class of the test pool
    (we'll add DB3 + D-Fire train hashes later).
    """
    test_pool: dict[str, list[tuple[Path, str]]] = {c: [] for c in CLASSES}
    seen_test: dict[str, set[str]] = {c: set() for c in CLASSES}

    # ---- fire from DB1
    db1_fire_unique: list[tuple[Path, str]] = []
    for p in sorted((DB1 / "fire").iterdir()):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        h = md5_of(p)
        if h in blocked or h in seen_test["fire"]:
            continue
        seen_test["fire"].add(h)
        db1_fire_unique.append((p, h))
    rng.shuffle(db1_fire_unique)
    test_pool["fire"].extend(db1_fire_unique[:TEST_PER_CLASS])

    # ---- D-Fire test, all 3 classes
    dfire_test_paths = sorted(p for p in DFIRE_TEST_IMG.iterdir()
                              if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng.shuffle(dfire_test_paths)

    for p in dfire_test_paths:
        cls = dfire_class_of(DFIRE_TEST_LAB, p)
        if cls is None:
            continue
        if len(test_pool[cls]) >= TEST_PER_CLASS:
            continue
        h = md5_of(p)
        if h in blocked or h in seen_test[cls]:
            continue
        seen_test[cls].add(h)
        test_pool[cls].append((p, h))
        if all(len(test_pool[c]) >= TEST_PER_CLASS for c in CLASSES):
            break

    return test_pool


# ----------------------------------------------------------------------
# Train/val pool
# ----------------------------------------------------------------------
def collect_trainval_pool(rng: random.Random, blocked_test_hashes: set[str]
                          ) -> tuple[dict[str, list[Path]],
                                     dict[str, dict[str, int]]]:
    pool: dict[str, list[Path]] = {c: [] for c in CLASSES}
    seen: dict[str, set[str]] = {c: set() for c in CLASSES}

    # 1) DB3 (all)
    db3_counts = {c: 0 for c in CLASSES}
    for c in CLASSES:
        for p in sorted((DB3 / c).iterdir()):
            if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            h = md5_of(p)
            if h in blocked_test_hashes or h in seen[c]:
                continue
            seen[c].add(h)
            pool[c].append(p)
            db3_counts[c] += 1

    # 2) D-Fire train, top-up to TRAINVAL_PER_CLASS
    dfire_paths = sorted(p for p in DFIRE_TRAIN_IMG.iterdir()
                         if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    rng.shuffle(dfire_paths)
    dfire_counts = {c: 0 for c in CLASSES}
    for p in dfire_paths:
        cls = dfire_class_of(DFIRE_TRAIN_LAB, p)
        if cls is None:
            continue
        if len(pool[cls]) >= TRAINVAL_PER_CLASS:
            continue
        h = md5_of(p)
        if h in blocked_test_hashes or h in seen[cls]:
            continue
        seen[cls].add(h)
        pool[cls].append(p)
        dfire_counts[cls] += 1
        if all(len(pool[c]) >= TRAINVAL_PER_CLASS for c in CLASSES):
            break

    sources = {c: {"db3": db3_counts[c],
                   "dfire_train": dfire_counts[c],
                   "total": len(pool[c])} for c in CLASSES}
    return pool, sources


# ----------------------------------------------------------------------
# Materialize splits
# ----------------------------------------------------------------------
def copy_one(p: Path, dst_dir: Path) -> str:
    h = md5_of(p)
    dst = dst_dir / f"{h[:16]}{p.suffix.lower()}"
    dst.write_bytes(p.read_bytes())
    return h


def main() -> None:
    if OUT.exists():
        print(f"[build] removing existing {OUT}")
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rng = random.Random(SEED)

    # Pre-block training sources from test pool (avoid sampling test images
    # that happen to be byte-identical to a DB3 / D-Fire-train image).
    print("[build] hashing DB3 + D-Fire train (blocked for test)…")
    blocked: set[str] = set()
    for c in CLASSES:
        for p in (DB3 / c).iterdir():
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                blocked.add(md5_of(p))
    for p in DFIRE_TRAIN_IMG.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            blocked.add(md5_of(p))
    print(f"[build] {len(blocked)} blocked training hashes")

    print(f"[build] collecting test pool ({TEST_PER_CLASS}/class) …")
    test_pool = collect_test_pool(rng, blocked)
    test_hashes: set[str] = set()
    for c in CLASSES:
        for _, h in test_pool[c]:
            test_hashes.add(h)
        print(f"[build] test/{c}: {len(test_pool[c])}")

    print(f"[build] collecting train+val pool ({TRAINVAL_PER_CLASS}/class) …")
    tv_pool, tv_sources = collect_trainval_pool(rng, test_hashes)
    for c in CLASSES:
        s = tv_sources[c]
        print(f"[build] train+val/{c}: db3={s['db3']}  "
              f"dfire_train={s['dfire_train']}  total={s['total']}")

    # ---- materialize test
    for c in CLASSES:
        d = OUT / "test" / c
        d.mkdir(parents=True, exist_ok=True)
        for p, _ in test_pool[c]:
            copy_one(p, d)

    # ---- materialize train + val (stratified)
    tv_counts: dict[str, dict[str, int]] = {}
    for c in CLASSES:
        imgs = list(tv_pool[c])
        rng.shuffle(imgs)
        val_imgs   = imgs[:VAL_PER_CLASS]
        train_imgs = imgs[VAL_PER_CLASS:VAL_PER_CLASS + TRAIN_PER_CLASS]
        for tag, group in (("train", train_imgs), ("val", val_imgs)):
            dst = OUT / tag / c
            dst.mkdir(parents=True, exist_ok=True)
            for p in group:
                copy_one(p, dst)
        tv_counts[c] = {"train": len(train_imgs), "val": len(val_imgs)}

    # Final hard check: no MD5 collision between any pair of splits.
    all_hashes: dict[str, set[str]] = {"train": set(), "val": set(),
                                       "test": set()}
    for split in ("train", "val", "test"):
        for c in CLASSES:
            for p in (OUT / split / c).iterdir():
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    all_hashes[split].add(md5_of(p))
    leak = {
        "train_in_val":  len(all_hashes["train"] & all_hashes["val"]),
        "train_in_test": len(all_hashes["train"] & all_hashes["test"]),
        "val_in_test":   len(all_hashes["val"]   & all_hashes["test"]),
    }
    print(f"[build] leakage check: {leak}")
    assert all(v == 0 for v in leak.values()), "Leak detected"

    total_train = sum(d["train"] for d in tv_counts.values())
    total_val   = sum(d["val"]   for d in tv_counts.values())
    total_test  = sum(len(test_pool[c]) for c in CLASSES)
    ratios = (total_train, total_val, total_test)
    s = sum(ratios)
    print(f"[build] totals  train={total_train}  val={total_val}  "
          f"test={total_test}  -> ratios "
          f"{total_train/s:.2f}/{total_val/s:.2f}/{total_test/s:.2f}")

    manifest = {
        "split_ratio":      f"{TRAIN_PER_CLASS}/{VAL_PER_CLASS}/{TEST_PER_CLASS}"
                            " per class  (70/15/15)",
        "train_per_class":  TRAIN_PER_CLASS,
        "val_per_class":    VAL_PER_CLASS,
        "test_per_class":   TEST_PER_CLASS,
        "seed":             SEED,
        "classes":          CLASSES,
        "trainval_sources": tv_sources,
        "test_sources": {
            "fire":       "FIRE_DATABASE_1/fire ∖ DB3, then D-Fire test/fire",
            "no_fire":    "D-Fire test (no labels)",
            "start_fire": "D-Fire test (smoke-only)",
        },
        "leakage_check":    leak,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[build] wrote {OUT}/manifest.json")
    print(f"[build] done -> {OUT}")


if __name__ == "__main__":
    main()
