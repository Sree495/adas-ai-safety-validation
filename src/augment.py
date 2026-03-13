"""
Generates 5 augmented val datasets for SOTIF boundary testing.

Conditions: rain, night, fog, noise, glare
Labels are reused unchanged — augmentations only affect pixels, not box geometry.

Prerequisites:
    pip install "albumentations>=1.3.0,<2.0"

Usage:
    python augment.py
"""

import os
import glob
import shutil
import cv2
import albumentations as A

from config import VAL_IMAGES, VAL_LABELS, AUG_BASE

CONDITIONS = {
    "rain": A.Compose([
        A.RandomRain(
            slant_lower=-10, slant_upper=10,
            drop_length=15, drop_width=1,
            drop_color=(180, 180, 180),
            blur_value=3,
            brightness_coefficient=0.75,
            rain_type="heavy",
            p=1.0,
        ),
    ]),

    "night": A.Compose([
        A.RandomBrightnessContrast(
            brightness_limit=(-0.65, -0.45),
            contrast_limit=(-0.1, 0.1),
            p=1.0,
        ),
    ]),

    "fog": A.Compose([
        A.RandomFog(
            fog_coef_lower=0.4,
            fog_coef_upper=0.7,
            alpha_coef=0.1,
            p=1.0,
        ),
    ]),

    "noise": A.Compose([
        A.GaussNoise(var_limit=(2000, 5000), p=1.0),
        A.ImageCompression(quality_lower=20, quality_upper=40, p=1.0),
    ]),

    "glare": A.Compose([
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_lower=0,
            num_flare_circles_lower=6,
            num_flare_circles_upper=10,
            src_radius=400,
            src_color=(255, 255, 255),
            p=1.0,
        ),
    ]),
}


def write_yaml(condition):
    """Write a YOLO-compatible dataset yaml for one augmented condition."""
    path = os.path.join(AUG_BASE, condition).replace("\\", "/")
    # train points to nominal train images (required by YOLO, not actually used here)
    from config import TRAIN_IMAGES
    nominal_train = TRAIN_IMAGES.replace("\\", "/")
    yaml_path = os.path.join(AUG_BASE, f"{condition}.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {path}\n\n")
        f.write(f"train: {nominal_train}\n")
        f.write(f"val: images\n\n")
        f.write(f"nc: 2\n\n")
        f.write(f"names:\n  0: person\n  1: car\n")
    return yaml_path


def augment_condition(condition, transform):
    out_images = os.path.join(AUG_BASE, condition, "images")
    out_labels = os.path.join(AUG_BASE, condition, "labels")
    os.makedirs(out_images, exist_ok=True)
    os.makedirs(out_labels, exist_ok=True)

    images = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg")))
    ok, failed = 0, 0

    for img_path in images:
        fname = os.path.basename(img_path)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            failed += 1
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        try:
            aug_rgb = transform(image=img_rgb)["image"]
        except Exception as e:
            print(f"    Warning: {fname} skipped — {e}")
            aug_rgb = img_rgb
            failed += 1

        cv2.imwrite(
            os.path.join(out_images, fname),
            cv2.cvtColor(aug_rgb, cv2.COLOR_RGB2BGR),
        )

        # Reuse the original label (pixel transform = same box coordinates)
        label_src = os.path.join(VAL_LABELS, fname.replace(".jpg", ".txt"))
        if os.path.exists(label_src):
            shutil.copy2(label_src, os.path.join(out_labels, fname.replace(".jpg", ".txt")))

        ok += 1

    yaml_path = write_yaml(condition)
    print(f"  {ok} images saved  ({failed} failed)  →  {yaml_path}")


if __name__ == "__main__":
    print("=== Generating augmented SOTIF test sets ===\n")

    for condition, transform in CONDITIONS.items():
        print(f"[{condition.upper()}]")
        augment_condition(condition, transform)
        print()

    print("Done. Run: python safety_report.py")
