"""
Model Explainability — GradCAM + EigenCAM  (ADAS/AD V&V — ISO 26262 ASIL)

Answers: "What pixels does the model use to make each detection?"

Two complementary methods:
  - EigenCAM  : PCA of backbone feature maps — fast, no gradient needed,
                shows overall scene attention
  - GradCAM   : Gradient of detection score w.r.t. feature maps — shows
                which regions matter for a specific detection

Output sets:
  A. Random sample — 8 daytime images (normal behaviour baseline)
  B. Night sample  — 8 night images (known low-recall scenario)
  C. Danger zone   — images flagged by uncertainty_eval (high conf + high std)
  D. Side-by-side  — daytime vs night comparison pairs (same scene type)
  E. Per-class     — person-focused vs car-focused GradCAM

Usage:
    venv/Scripts/python.exe src/explainability.py
"""

import os
import json
import glob
import csv
import random
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
from config import (MODEL_PATH, VAL_IMAGES, ANN_DIR,
                    DANGER_CSV, RESULTS_EXPLAINABILITY as RESULTS_DIR)

CONF         = 0.30
TARGET_LAYER = 9          # SPPF — end of backbone
IMG_SIZE     = 640
CLASS_NAMES  = {0: "person", 1: "car"}
CLASS_COLORS = {0: (255, 100, 100), 1: (100, 180, 255)}   # BGR: red=person, blue=car
SAMPLE_N     = 8
random.seed(42)

# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(img_bgr, size=IMG_SIZE):
    """Resize to square, convert BGR→RGB, return tensor (1,3,H,W) and scale factors."""
    h0, w0 = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (size, size))
    t = torch.from_numpy(img_res).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return t, w0, h0


def tensor_to_bgr(t):
    """Convert preprocessed tensor back to uint8 BGR for overlay."""
    arr = (t.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

# ── EigenCAM ──────────────────────────────────────────────────────────────────

def eigen_cam(feat_map):
    """
    feat_map: (1, C, H, W) torch tensor
    Returns normalised (H, W) saliency map in [0, 1].
    """
    fm = feat_map.squeeze(0).detach()           # (C, H, W)
    C, H, W = fm.shape
    fm_flat = fm.reshape(C, H * W)              # (C, H*W)
    # SVD: first right-singular vector = dominant spatial pattern
    _, _, Vt = torch.linalg.svd(fm_flat, full_matrices=False)
    cam = Vt[0].reshape(H, W).numpy()
    cam = np.maximum(cam, 0)                    # ReLU
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


# ── GradCAM ───────────────────────────────────────────────────────────────────

def grad_cam(model_nn, img_tensor, target_cls=None):
    """
    Compute GradCAM w.r.t. backbone layer 9 (SPPF).
    target_cls: 0=person, 1=car, None=all classes
    Returns normalised (H, W) saliency map in [0, 1].
    """
    feat_store = {}
    grad_store = {}

    def fhook(m, i, o):
        feat_store['f'] = o

    def bhook(m, gi, go):
        grad_store['g'] = go[0]

    h1 = model_nn.model[TARGET_LAYER].register_forward_hook(fhook)
    h2 = model_nn.model[TARGET_LAYER].register_full_backward_hook(bhook)

    img_tensor = img_tensor.clone().requires_grad_(True)

    model_nn.eval()
    with torch.enable_grad():
        out = model_nn(img_tensor)           # (1, 6, 8400): [cx,cy,w,h, s0, s1]
        preds = out[0]                       # (1, 6, 8400)
        cls_logits = preds[0, 4:, :]        # (2, 8400)
        cls_scores = cls_logits.sigmoid()   # (2, 8400)

        if target_cls is not None:
            scores = cls_scores[target_cls]
        else:
            scores = cls_scores.max(dim=0)[0]

        # Use top-k anchor scores as the scalar signal
        k = min(30, scores.numel())
        score = scores.topk(k)[0].sum()
        model_nn.zero_grad()
        score.backward()

    h1.remove()
    h2.remove()

    fm = feat_store['f'].squeeze(0).detach()     # (C, H, W)
    gr = grad_store['g'].squeeze(0).detach()     # (C, H, W)

    # Global average pool gradients across spatial dims → per-channel weights
    weights = gr.mean(dim=[1, 2])                # (C,)
    cam = (weights[:, None, None] * fm).sum(dim=0)  # (H, W)
    cam = torch.relu(cam).numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


# ── Overlay helpers ───────────────────────────────────────────────────────────

def overlay_cam(img_bgr, cam, alpha=0.45, colormap=cv2.COLORMAP_JET):
    """Resize CAM to image size and blend with original image."""
    h, w = img_bgr.shape[:2]
    cam_u8  = (cam * 255).astype(np.uint8)
    cam_res = cv2.resize(cam_u8, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap(cam_res, colormap)
    blended = cv2.addWeighted(img_bgr, 1 - alpha, heatmap, alpha, 0)
    return blended


def draw_detections(img_bgr, model, path, w0, h0):
    """Run predict and draw boxes on image (scaled back to original size)."""
    r = model.predict(path, conf=CONF, verbose=False)[0]
    out = img_bgr.copy()
    if r.boxes is None:
        return out
    sx = w0 / IMG_SIZE
    sy = h0 / IMG_SIZE
    for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(),
                               r.boxes.conf.cpu().numpy(),
                               r.boxes.cls.cpu().numpy().astype(int)):
        x1, y1, x2, y2 = box
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        cv2.rectangle(out, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), color, 2)
        label = f"{CLASS_NAMES.get(cls,'?')} {conf:.2f}"
        cv2.putText(out, label, (int(x1*sx), max(int(y1*sy)-6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out


# ── Grid saving ───────────────────────────────────────────────────────────────

def save_cam_grid(entries, save_path, title, cols=4):
    """
    entries: list of (title_str, img_bgr_with_overlay)
    Saves a matplotlib grid figure.
    """
    rows = (len(entries) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.2))
    axes = np.array(axes).reshape(-1)
    for ax, (t, img) in zip(axes, entries):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(t, fontsize=7, pad=3)
        ax.axis("off")
    for ax in axes[len(entries):]:
        ax.axis("off")
    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {os.path.basename(save_path)}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load metadata
    img_tags = {}
    for ann_file in glob.glob(os.path.join(ANN_DIR, "*.json")):
        stem = os.path.basename(ann_file).replace(".jpg.json", "")
        with open(ann_file, encoding="utf-8") as f:
            d = json.load(f)
        img_tags[stem] = {t["name"]: t["value"] for t in d.get("tags", [])}

    # Load danger-zone stems from uncertainty_eval
    danger_stems = set()
    if os.path.exists(DANGER_CSV):
        with open(DANGER_CSV, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                if row.get("Danger") == "DANGER":
                    danger_stems.add(row["Image"])

    all_paths = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg")))
    stem_to_path = {os.path.splitext(os.path.basename(p))[0]: p for p in all_paths}

    day_paths   = [p for p in all_paths
                   if img_tags.get(os.path.splitext(os.path.basename(p))[0], {}).get("timeofday") == "daytime"]
    night_paths = [p for p in all_paths
                   if img_tags.get(os.path.splitext(os.path.basename(p))[0], {}).get("timeofday") == "night"]
    danger_paths = [stem_to_path[s] for s in danger_stems if s in stem_to_path]

    print(f"Daytime images : {len(day_paths)}")
    print(f"Night images   : {len(night_paths)}")
    print(f"Danger-zone    : {len(danger_paths)}")

    print(f"\nLoading model...")
    model   = YOLO(MODEL_PATH)
    model_nn = model.model

    # ── A: Daytime baseline ───────────────────────────────────────────────────
    print("\n[A] EigenCAM — Daytime baseline (8 images)...")
    sample_day = random.sample(day_paths, min(SAMPLE_N, len(day_paths)))
    entries = []
    for path in sample_day:
        stem = os.path.splitext(os.path.basename(path))[0]
        img_bgr = cv2.imread(path)
        t, w0, h0 = preprocess(img_bgr)

        feat = {}
        h = model_nn.model[TARGET_LAYER].register_forward_hook(lambda m,i,o: feat.update({'f': o}))
        with torch.no_grad():
            model_nn(t)
        h.remove()

        cam  = eigen_cam(feat['f'])
        base = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        over = overlay_cam(base, cam)
        over = draw_detections(over, model, path, IMG_SIZE, IMG_SIZE)
        tags = img_tags.get(stem, {})
        entries.append((f"{tags.get('weather','?')} | {tags.get('scene','?')}", over))

    save_cam_grid(entries, os.path.join(RESULTS_DIR, "A_eigencam_daytime.png"),
                  "EigenCAM — Daytime Baseline (backbone attention)", cols=4)

    # ── B: Night scenario ─────────────────────────────────────────────────────
    print("[B] EigenCAM — Night scenario (8 images)...")
    sample_night = random.sample(night_paths, min(SAMPLE_N, len(night_paths)))
    entries = []
    for path in sample_night:
        stem = os.path.splitext(os.path.basename(path))[0]
        img_bgr = cv2.imread(path)
        t, w0, h0 = preprocess(img_bgr)

        feat = {}
        h = model_nn.model[TARGET_LAYER].register_forward_hook(lambda m,i,o: feat.update({'f': o}))
        with torch.no_grad():
            model_nn(t)
        h.remove()

        cam  = eigen_cam(feat['f'])
        base = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
        over = overlay_cam(base, cam)
        over = draw_detections(over, model, path, IMG_SIZE, IMG_SIZE)
        tags = img_tags.get(stem, {})
        entries.append((f"{tags.get('weather','?')} | {tags.get('scene','?')}", over))

    save_cam_grid(entries, os.path.join(RESULTS_DIR, "B_eigencam_night.png"),
                  "EigenCAM — Night Scenario (backbone attention)", cols=4)

    # ── C: Danger zone GradCAM ────────────────────────────────────────────────
    print("[C] GradCAM — Danger-zone images...")
    if danger_paths:
        entries = []
        for path in danger_paths:
            stem = os.path.splitext(os.path.basename(path))[0]
            img_bgr = cv2.imread(path)
            t, w0, h0 = preprocess(img_bgr)

            cam  = grad_cam(model_nn, t, target_cls=None)
            base = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
            over = overlay_cam(base, cam, alpha=0.50)
            over = draw_detections(over, model, path, IMG_SIZE, IMG_SIZE)
            tags = img_tags.get(stem, {})
            title = (f"DANGER | {tags.get('timeofday','?')} | "
                     f"{tags.get('scene','?')}")
            entries.append((title, over))
        save_cam_grid(entries, os.path.join(RESULTS_DIR, "C_gradcam_danger_zone.png"),
                      "GradCAM — Danger Zone (high confidence + high uncertainty)", cols=3)
    else:
        print("  No danger-zone images found — skipping C.")

    # ── D: Daytime vs Night side-by-side ──────────────────────────────────────
    print("[D] GradCAM — Day vs Night side-by-side comparison...")
    n_pairs = 4
    day_sample   = random.sample(day_paths,   min(n_pairs, len(day_paths)))
    night_sample = random.sample(night_paths, min(n_pairs, len(night_paths)))

    fig, axes = plt.subplots(n_pairs, 4, figsize=(16, n_pairs * 3.5))
    col_titles = ["Daytime — Original+Boxes", "Daytime — GradCAM",
                  "Night — Original+Boxes",   "Night — GradCAM"]
    for ax, ct in zip(axes[0], col_titles):
        ax.set_title(ct, fontsize=9, fontweight="bold")

    for row, (dp, np_) in enumerate(zip(day_sample, night_sample)):
        for col, (path, label) in enumerate([(dp, "day"), (dp, "day_cam"),
                                              (np_, "night"), (np_, "night_cam")]):
            img_bgr = cv2.imread(path)
            t, w0, h0 = preprocess(img_bgr)
            base = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

            if "cam" in label:
                cam  = grad_cam(model_nn, t)
                disp = overlay_cam(base, cam, alpha=0.50)
            else:
                disp = draw_detections(base.copy(), model, path, IMG_SIZE, IMG_SIZE)

            axes[row, col].imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            axes[row, col].axis("off")

    plt.suptitle("GradCAM: Daytime vs Night — What the model attends to", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "D_gradcam_day_vs_night.png"),
                dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved: D_gradcam_day_vs_night.png")

    # ── E: Per-class GradCAM (person vs car) ─────────────────────────────────
    print("[E] GradCAM — Per-class (person vs car) on 4 city-street images...")
    city_paths = [p for p in all_paths
                  if img_tags.get(os.path.splitext(os.path.basename(p))[0], {}).get("scene") == "city street"
                  and img_tags.get(os.path.splitext(os.path.basename(p))[0], {}).get("timeofday") == "daytime"]
    city_sample = random.sample(city_paths, min(4, len(city_paths)))

    fig, axes = plt.subplots(len(city_sample), 3, figsize=(12, len(city_sample) * 3.5))
    col_titles = ["Original + Detections", "GradCAM: Person (class 0)", "GradCAM: Car (class 1)"]
    for ax, ct in zip(axes[0], col_titles):
        ax.set_title(ct, fontsize=9, fontweight="bold")

    for row, path in enumerate(city_sample):
        img_bgr = cv2.imread(path)
        t, w0, h0 = preprocess(img_bgr)
        base = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

        orig = draw_detections(base.copy(), model, path, IMG_SIZE, IMG_SIZE)
        cam_person = grad_cam(model_nn, t, target_cls=0)
        cam_car    = grad_cam(model_nn, t, target_cls=1)

        over_person = overlay_cam(base, cam_person, alpha=0.50, colormap=cv2.COLORMAP_HOT)
        over_car    = overlay_cam(base, cam_car,    alpha=0.50, colormap=cv2.COLORMAP_OCEAN)

        for col, disp in enumerate([orig, over_person, over_car]):
            axes[row, col].imshow(cv2.cvtColor(disp, cv2.COLOR_BGR2RGB))
            axes[row, col].axis("off")

    plt.suptitle("Per-Class GradCAM — Person vs Car Attention (City Street, Daytime)", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "E_gradcam_per_class.png"),
                dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  Saved: E_gradcam_per_class.png")

    # ── Summary report ────────────────────────────────────────────────────────
    W = 80
    print("\n" + "="*W)
    print("         EXPLAINABILITY REPORT  (XAI — ISO 26262 ASIL Justification)")
    print("="*W)
    print("""
  METHOD SUMMARY
  ┌─────────────┬───────────────────────────────────────────────────────┐
  │ EigenCAM    │ PCA of backbone feature maps. No gradient needed.     │
  │             │ Shows overall scene regions the backbone encodes.     │
  ├─────────────┼───────────────────────────────────────────────────────┤
  │ GradCAM     │ Gradient of detection score → feature map weights.    │
  │             │ Shows which regions drive specific class predictions. │
  └─────────────┴───────────────────────────────────────────────────────┘

  OUTPUT FILES
    A_eigencam_daytime.png     — Baseline attention on well-lit scenes
    B_eigencam_night.png       — Attention shift on night scenes
    C_gradcam_danger_zone.png  — What model attends to on uncertain inputs
    D_gradcam_day_vs_night.png — Direct comparison of attention quality
    E_gradcam_per_class.png    — Person vs car attention separation

  INTERPRETATION FOR ISO 26262
    • If heatmaps concentrate on object boundaries → expected behaviour
    • If heatmaps fire on backgrounds (sky, road surface) → spurious correlations
    • Night heatmaps diffuse/low-contrast → backbone losing discriminative signal
    • Danger-zone heatmaps scattered → model guessing, not attending to objects
    • Per-class separation (E) confirms model distinguishes person vs car features

  SOTIF LINKAGE
    Compare A vs B: daytime heatmaps should be tighter, more object-centred.
    If night heatmaps are diffuse → confirms ODD restriction on night driving.
    Danger-zone heatmaps (C) are the visual evidence for the UQ findings.
""")
    print("="*W)
    print(f"\n  All plots saved to: {RESULTS_DIR}/")
