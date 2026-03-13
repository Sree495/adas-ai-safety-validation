"""
One-time data fix script. Run this ONCE before retraining or testing.

Fixes two issues:
1. Renames data/bdd-sample/{train,test}/img/ -> images/ so YOLO can resolve labels
2. Renames *.jpg.txt label files -> *.txt in train, val, and test splits
"""
import os
import glob
import shutil

from config import DATA_DIR
BASE = str(DATA_DIR)


def fix_label_names(labels_dir):
    bad = glob.glob(os.path.join(labels_dir, "*.jpg.txt"))
    if not bad:
        print(f"  Labels already correct in: {labels_dir}")
        return
    print(f"  Renaming {len(bad)} label files in: {labels_dir}")
    for src in bad:
        dst = src.replace(".jpg.txt", ".txt")
        os.rename(src, dst)


def fix_image_folder(split):
    old = os.path.join(BASE, split, "img")
    new = os.path.join(BASE, split, "images")
    if os.path.isdir(new):
        print(f"  {split}/images/ already exists, skipping rename.")
        return
    if not os.path.isdir(old):
        print(f"  WARNING: {split}/img/ not found at {old}")
        return
    print(f"  Renaming {split}/img/ -> {split}/images/")
    shutil.move(old, new)


if __name__ == "__main__":
    print("=== Fixing data structure ===")

    print("\n[1] Renaming train/img -> train/images")
    fix_image_folder("train")

    print("\n[2] Renaming test/img -> test/images")
    fix_image_folder("test")

    print("\n[3] Fixing train label filenames")
    fix_label_names(os.path.join(BASE, "train", "labels"))

    print("\n[4] Fixing val label filenames")
    fix_label_names(os.path.join(BASE, "val", "labels"))

    print("\n[5] Fixing test label filenames")
    fix_label_names(os.path.join(BASE, "test", "labels"))

    print("\nDone.")
