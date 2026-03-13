"""
Out-of-Distribution (OOD) Detection  (ADAS/AD V&V — SOTIF ISO 21448)

Extracts 512-dim backbone features from every val image via a forward hook
on YOLOv8s layer 9 (SPPF — end of backbone, before neck/head).

Nominal distribution = weather:clear + timeofday:daytime images.

Pipeline:
  1. Extract 512-dim GAP features for all val images
  2. Fit PCA (n=50) on nominal features only
  3. Project ALL images into PCA space
  4. Compute Mahalanobis distance (= normalised Euclidean in PCA space)
     for each image relative to the nominal centroid
  5. Flag images/scenarios with distance > threshold as OOD
  6. Plot & save: score histogram, PCA scatter (weather / timeofday / scene),
     per-scenario box-plot, ranked OOD table
  7. Save CSV

Usage:
    venv/Scripts/python.exe src/ood_detector.py
"""

import os
import json
import glob
import csv
import collections
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
from config import MODEL_PATH, VAL_IMAGES, ANN_DIR, RESULTS_OOD as RESULTS_DIR

PCA_COMPONENTS  = 50      # PCA dims (≤ min(n_nominal, 512))
OOD_PERCENTILE  = 90      # images above this nominal-score percentile → OOD flag
TAG_DIMS        = ["weather", "timeofday", "scene"]
NOMINAL_TAGS    = {"weather": "clear", "timeofday": "daytime"}  # in-distribution definition

# ── Feature extraction ────────────────────────────────────────────────────────

def build_feature_extractor(model):
    """Return a function that extracts a 512-dim GAP vector from layer 9 (SPPF)."""
    cache = {}

    def hook(module, inp, out):
        # Global Average Pool: (1, 512, H, W) → (512,)
        cache["feat"] = out.mean(dim=[2, 3]).squeeze(0).detach().cpu().numpy()

    handle = model.model.model[9].register_forward_hook(hook)

    def extract(img_bgr):
        model.predict(img_bgr, verbose=False)
        return cache["feat"].copy()

    return extract, handle


def extract_all_features(model, image_paths):
    """Run extraction on all images. Returns (N, 512) array + list of stems."""
    extractor, handle = build_feature_extractor(model)
    feats, stems = [], []
    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            continue
        feats.append(extractor(img))
        stems.append(os.path.splitext(os.path.basename(path))[0])
        if (i + 1) % 20 == 0 or (i + 1) == len(image_paths):
            print(f"  Extracted {i+1}/{len(image_paths)}", end="\r", flush=True)
    handle.remove()
    print()
    return np.array(feats, dtype=np.float32), stems


# ── PCA (manual, no sklearn dependency) ───────────────────────────────────────

def fit_pca(X, n_components):
    """Fit PCA on X. Returns (mean, components, explained_var)."""
    mu = X.mean(axis=0)
    Xc = X - mu
    # SVD on centred data
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    components = Vt[:n_components]          # (n_components, 512)
    explained_var = (S[:n_components] ** 2) / (len(X) - 1)
    return mu, components, explained_var


def apply_pca(X, mu, components):
    return (X - mu) @ components.T          # (N, n_components)


# ── Mahalanobis in PCA space (= normalised Euclidean) ─────────────────────────

def mahalanobis_pca(X_pca, nominal_pca):
    """
    In PCA space the covariance is diagonal (eigenvalues).
    Mahalanobis ≈ Euclidean normalised by per-PC standard deviation.
    """
    mu  = nominal_pca.mean(axis=0)
    std = nominal_pca.std(axis=0) + 1e-8   # avoid div-by-zero
    return np.sqrt(((X_pca - mu) / std) ** 2).mean(axis=1)


# ── Plotting helpers ──────────────────────────────────────────────────────────

PALETTE = [
    "#e6194b","#3cb44b","#4363d8","#f58231","#911eb4",
    "#42d4f4","#f032e6","#bfef45","#fabed4","#469990",
    "#dcbeff","#9A6324","#800000","#aaffc3",
]


def scatter_pca(X_pca, stems, img_tags, colour_by, save_path):
    labels = [img_tags.get(s, {}).get(colour_by, "undefined") for s in stems]
    unique = sorted(set(labels))
    cmap   = {v: PALETTE[i % len(PALETTE)] for i, v in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(9, 6))
    for lbl in unique:
        idx = [i for i, l in enumerate(labels) if l == lbl]
        ax.scatter(X_pca[idx, 0], X_pca[idx, 1],
                   c=cmap[lbl], label=lbl, alpha=0.75, s=40, edgecolors="none")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title(f"PCA Feature Space — coloured by {colour_by}")
    ax.legend(fontsize=8, markerscale=1.4, bbox_to_anchor=(1.01, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def score_histogram(scores, threshold, save_path):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(scores, bins=30, color="#4363d8", alpha=0.75, edgecolor="white")
    ax.axvline(threshold, color="#e6194b", linewidth=2, linestyle="--",
               label=f"OOD threshold ({OOD_PERCENTILE}th pct = {threshold:.2f})")
    ax.set_xlabel("Mahalanobis OOD Score")
    ax.set_ylabel("Number of images")
    ax.set_title("OOD Score Distribution — all val images")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def scenario_boxplot(scores, stems, img_tags, dim, threshold, save_path):
    tag_scores = collections.defaultdict(list)
    for s, sc in zip(stems, scores):
        val = img_tags.get(s, {}).get(dim, "undefined")
        tag_scores[val].append(sc)

    labels = sorted(tag_scores, key=lambda x: -np.median(tag_scores[x]))
    data   = [tag_scores[l] for l in labels]
    colors = [("#e6194b" if np.median(d) > threshold else "#3cb44b") for d in data]

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax.axhline(threshold, color="#e6194b", linewidth=1.5, linestyle="--",
               label=f"OOD threshold = {threshold:.2f}")
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Mahalanobis OOD Score")
    ax.set_title(f"OOD Score per {dim} scenario")
    ax.legend()
    red_p   = mpatches.Patch(color="#e6194b", alpha=0.7, label="OOD (median > threshold)")
    green_p = mpatches.Patch(color="#3cb44b", alpha=0.7, label="In-distribution")
    ax.legend(handles=[red_p, green_p], fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Parse annotation metadata
    print("Parsing BDD annotation metadata...")
    img_tags = {}
    for ann_file in glob.glob(os.path.join(ANN_DIR, "*.json")):
        stem = os.path.basename(ann_file).replace(".jpg.json", "")
        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)
        img_tags[stem] = {t["name"]: t["value"] for t in data.get("tags", [])
                          if t["name"] in TAG_DIMS}

    # 2. Collect all val image paths
    all_paths = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg")))
    print(f"  {len(all_paths)} val images found.")

    # 3. Identify nominal images (in-distribution definition)
    def is_nominal(stem):
        tags = img_tags.get(stem, {})
        return all(tags.get(k) == v for k, v in NOMINAL_TAGS.items())

    nominal_stems = [os.path.splitext(os.path.basename(p))[0]
                     for p in all_paths if is_nominal(os.path.splitext(os.path.basename(p))[0])]
    nominal_paths = [p for p in all_paths
                     if os.path.splitext(os.path.basename(p))[0] in set(nominal_stems)]
    print(f"  Nominal images (clear + daytime): {len(nominal_paths)}")

    # 4. Load model and extract features
    print(f"\nLoading model...")
    model = YOLO(MODEL_PATH)

    print("Extracting features from all val images...")
    all_feats, all_stems = extract_all_features(model, all_paths)
    print(f"  Feature matrix: {all_feats.shape}")

    # 5. Fit PCA on nominal features only
    nominal_idx = [i for i, s in enumerate(all_stems) if s in set(nominal_stems)]
    nominal_feats = all_feats[nominal_idx]
    n_comp = min(PCA_COMPONENTS, len(nominal_feats) - 1, all_feats.shape[1])
    print(f"\nFitting PCA ({n_comp} components) on {len(nominal_feats)} nominal images...")
    pca_mu, pca_comp, pca_var = fit_pca(nominal_feats, n_comp)
    explained = pca_var.sum() / ((nominal_feats - pca_mu).var(axis=0).sum() * nominal_feats.shape[1] / pca_var.sum() + 1e-8)

    # 6. Project all images into PCA space
    all_pca     = apply_pca(all_feats, pca_mu, pca_comp)
    nominal_pca = all_pca[nominal_idx]

    # 7. Compute OOD scores
    print("Computing Mahalanobis OOD scores...")
    scores    = mahalanobis_pca(all_pca, nominal_pca)
    threshold = float(np.percentile(scores, OOD_PERCENTILE))
    ood_flags = scores > threshold
    print(f"  OOD threshold ({OOD_PERCENTILE}th pct): {threshold:.3f}")
    print(f"  OOD images: {ood_flags.sum()} / {len(scores)}")

    # 8. Per-scenario OOD stats
    print("\nPer-scenario OOD statistics:")
    scenario_stats = {}
    for dim in TAG_DIMS:
        tag_data = collections.defaultdict(list)
        for stem, sc, ood in zip(all_stems, scores, ood_flags):
            val = img_tags.get(stem, {}).get(dim, "undefined")
            tag_data[val].append((sc, ood))
        scenario_stats[dim] = tag_data

    rows = []
    for dim in TAG_DIMS:
        for tag_val, entries in sorted(scenario_stats[dim].items(),
                                       key=lambda x: -np.median([e[0] for e in x[1]])):
            sc_arr  = [e[0] for e in entries]
            ood_arr = [e[1] for e in entries]
            median  = float(np.median(sc_arr))
            mean    = float(np.mean(sc_arr))
            pct_ood = float(np.mean(ood_arr)) * 100
            is_ood  = median > threshold
            rows.append({
                "Dimension": dim,
                "Scenario":  tag_val,
                "N":         len(entries),
                "Median_OOD": median,
                "Mean_OOD":   mean,
                "Pct_OOD":    pct_ood,
                "OOD_Flag":   "OOD" if is_ood else "IN-DIST",
            })

    # 9. Print report
    W = 88
    print("\n" + "=" * W)
    print("         OOD DETECTION REPORT  (YOLOv8s Backbone — SOTIF ISO 21448)")
    print(f"         Nominal: weather=clear + timeofday=daytime  |  Threshold: {threshold:.3f}")
    print("=" * W)

    for dim in TAG_DIMS:
        dim_rows = [r for r in rows if r["Dimension"] == dim]
        print(f"\n  TAG: {dim.upper()}")
        print(f"    {'Scenario':<22} {'N':>4}  {'Median':>8} {'Mean':>8} {'%OOD':>7}  {'Status'}")
        print(f"    {'-'*(W-4)}")
        for r in dim_rows:
            marker = " ← OOD" if r["OOD_Flag"] == "OOD" else ""
            print(f"    {r['Scenario']:<22} {r['N']:>4}  "
                  f"{r['Median_OOD']:>8.3f} {r['Mean_OOD']:>8.3f} "
                  f"{r['Pct_OOD']:>6.1f}%  {r['OOD_Flag']}{marker}")

    print("\n" + "=" * W)

    # 10. Top OOD images
    top_idx  = np.argsort(-scores)[:15]
    print(f"\n  TOP 15 MOST OOD IMAGES:")
    print(f"    {'Image':<40} {'Score':>7}  {'weather':<16} {'timeofday':<12} {'scene'}")
    print(f"    {'-'*(W-4)}")
    for idx in top_idx:
        s  = all_stems[idx]
        t  = img_tags.get(s, {})
        print(f"    {s:<40} {scores[idx]:>7.3f}  "
              f"{t.get('weather','?'):<16} {t.get('timeofday','?'):<12} {t.get('scene','?')}")

    print("=" * W)

    # 11. Cross-reference with SOTIF findings
    print("\n  SOTIF CROSS-REFERENCE:")
    print("  Scenarios flagged as OOD AND known low-recall (from safety_report / scenario_analysis):")
    known_risky = {
        ("weather", "rainy"): "recall=0.287 (safety_report: augmented HIGH risk)",
        ("timeofday", "night"): "recall=0.354 (safety_report: CRITICAL on synthetic night)",
        ("scene", "highway"): "recall=0.234, person_recall=0.000 (scenario_analysis)",
    }
    for dim in TAG_DIMS:
        for r in rows:
            if r["Dimension"] != dim:
                continue
            key = (dim, r["Scenario"])
            if key in known_risky and r["OOD_Flag"] == "OOD":
                print(f"    ✖ {dim}={r['Scenario']:<18} OOD score={r['Median_OOD']:.3f}  "
                      f"→ {known_risky[key]}")

    print("\n  ISO 21448 interpretation:")
    print("  OOD + low-recall = unknown unsafe situation — must be excluded from ODD")
    print("  or mitigated with system-level safety measures (driver handover, speed limit).")
    print("=" * W)

    # 12. Plots
    print("\nGenerating plots...")

    score_histogram(scores, threshold,
                    os.path.join(RESULTS_DIR, "ood_score_histogram.png"))
    print("  ood_score_histogram.png")

    for dim in TAG_DIMS:
        scatter_pca(all_pca, all_stems, img_tags, dim,
                    os.path.join(RESULTS_DIR, f"pca_scatter_{dim}.png"))
        print(f"  pca_scatter_{dim}.png")

        scenario_boxplot(scores, all_stems, img_tags, dim, threshold,
                         os.path.join(RESULTS_DIR, f"ood_boxplot_{dim}.png"))
        print(f"  ood_boxplot_{dim}.png")

    # 13. Save CSV
    csv_path = os.path.join(RESULTS_DIR, "ood_report.csv")
    fieldnames = ["Dimension", "Scenario", "N", "Median_OOD", "Mean_OOD", "Pct_OOD", "OOD_Flag"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]
                             for k in fieldnames})

    # Save per-image scores
    img_csv = os.path.join(RESULTS_DIR, "ood_per_image.csv")
    with open(img_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "OOD_Score", "OOD_Flag", "weather", "timeofday", "scene"])
        for stem, sc, ood in zip(all_stems, scores, ood_flags):
            t = img_tags.get(stem, {})
            writer.writerow([stem, f"{sc:.4f}", "OOD" if ood else "IN-DIST",
                             t.get("weather", "?"), t.get("timeofday", "?"), t.get("scene", "?")])

    print(f"\n  CSV saved to: {csv_path}")
    print(f"  Per-image CSV: {img_csv}")
    print(f"  Plots saved to: {RESULTS_DIR}/")
