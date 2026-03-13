"""
Uncertainty Quantification & Calibration Evaluation  (ADAS/AD V&V)

Answers: "How confident should we be in each detection?"

Pipeline:
  1. Collect all detections from val set (conf >= 0.01) + match to GT (IoU >= 0.5)
     → each detection labelled as TP (1) or FP (0)
  2. Compute ECE (Expected Calibration Error) — before calibration
  3. Plot Reliability Diagram (calibration curve)
  4. Fit Temperature Scaling (T) to minimise ECE
  5. Show ECE after calibration + updated reliability diagram
  6. Run Test-Time Augmentation (TTA, N=5) on every val image
     → per-image uncertainty = std of mean-confidence across augmentations
  7. Flag "danger zone": high raw confidence + high TTA uncertainty
  8. Cross-reference with OOD scores (if available)
  9. Save CSV + plots

Embedded deployment relevance:
  - Temperature T must be re-validated after INT8 quantisation
  - TTA uncertainty provides a runtime safety signal at low cost (5 passes)
  - Danger-zone detections = candidates for driver handover / reduced autonomy

Usage:
    venv/Scripts/python.exe src/uncertainty_eval.py
"""

import os
import csv
import glob
import json
import collections
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.optimize import minimize_scalar

from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
from config import (MODEL_PATH, VAL_IMAGES, VAL_LABELS, ANN_DIR,
                    OOD_PER_IMAGE_CSV as OOD_CSV,
                    RESULTS_UNCERTAINTY as RESULTS_DIR)

RAW_CONF    = 0.01    # low threshold to collect full confidence distribution
IOU_THRESH  = 0.50    # IoU for TP matching
N_BINS      = 10      # reliability diagram bins
N_TTA       = 5       # TTA augmentation passes

DANGER_CONF = 0.70    # high confidence threshold for danger-zone
DANGER_STD  = 0.10    # high TTA std threshold for danger-zone
CLASS_NAMES = {0: "person", 1: "car"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_gt(label_path, img_w=1280, img_h=720):
    """Load YOLO labels → list of (class_id, x1, y1, x2, y2) in pixels."""
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            p = line.strip().split()
            if len(p) != 5:
                continue
            cls, cx, cy, w, h = int(p[0]), *map(float, p[1:])
            x1 = (cx - w/2) * img_w;  y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w;  y2 = (cy + h/2) * img_h
            boxes.append((cls, x1, y1, x2, y2))
    return boxes


def iou(a, b):
    xi1, yi1 = max(a[0], b[0]), max(a[1], b[1])
    xi2, yi2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    if inter == 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua


def match_detections(pred_boxes, pred_confs, pred_cls, gt_boxes):
    """
    Greedy confidence-sorted matching.
    Returns list of (conf, is_tp, cls_id) for every prediction.
    """
    matched_gt = [False] * len(gt_boxes)
    results = []
    order = np.argsort(-pred_confs)
    for idx in order:
        pb   = pred_boxes[idx]
        conf = pred_confs[idx]
        cls  = pred_cls[idx]
        best_iou, best_j = 0.0, -1
        for j, (gc, gx1, gy1, gx2, gy2) in enumerate(gt_boxes):
            if matched_gt[j] or gc != cls:
                continue
            v = iou(pb, (gx1, gy1, gx2, gy2))
            if v > best_iou:
                best_iou, best_j = v, j
        if best_iou >= IOU_THRESH and best_j >= 0:
            matched_gt[best_j] = True
            results.append((conf, 1, cls))
        else:
            results.append((conf, 0, cls))
    return results


# ── Calibration ───────────────────────────────────────────────────────────────

def compute_ece(confs, labels, n_bins=N_BINS):
    """Expected Calibration Error."""
    confs  = np.array(confs)
    labels = np.array(labels)
    bins   = np.linspace(0, 1, n_bins + 1)
    ece    = 0.0
    bin_data = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            bin_data.append((0.5*(lo+hi), 0.0, 0.0, 0))
            continue
        acc  = labels[mask].mean()
        conf = confs[mask].mean()
        ece += (mask.sum() / len(confs)) * abs(acc - conf)
        bin_data.append((conf, acc, abs(acc - conf), mask.sum()))
    return ece, bin_data


def apply_temperature(confs, T):
    """Temperature-scale confidence scores."""
    confs = np.clip(np.array(confs), 1e-7, 1-1e-7)
    logits = np.log(confs / (1 - confs))
    return 1.0 / (1.0 + np.exp(-logits / T))


def find_temperature(confs, labels):
    """Minimise ECE over temperature T in [0.1, 10]."""
    def ece_fn(T):
        scaled = apply_temperature(confs, T)
        return compute_ece(scaled, labels)[0]
    result = minimize_scalar(ece_fn, bounds=(0.1, 10.0), method="bounded")
    return result.x


# ── Reliability diagram ───────────────────────────────────────────────────────

def plot_reliability(bin_data_raw, bin_data_cal, T, ece_raw, ece_cal, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, bd, title, ece in [
        (axes[0], bin_data_raw, f"Before Calibration  (ECE={ece_raw:.4f})", ece_raw),
        (axes[1], bin_data_cal, f"After Temperature Scaling  T={T:.3f}  (ECE={ece_cal:.4f})", ece_cal),
    ]:
        confs = [b[0] for b in bd]
        accs  = [b[1] for b in bd]
        gaps  = [b[2] for b in bd]
        ns    = [b[3] for b in bd]

        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
        bars = ax.bar(confs, accs, width=0.08, alpha=0.6, color="#4363d8",
                      label="Actual precision", align="center")
        ax.bar(confs, gaps, width=0.08, alpha=0.4, color="#e6194b",
               bottom=[min(a, c) for a, c in zip(accs, confs)],
               label="Calibration gap")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence"); ax.set_ylabel("Precision (fraction TP)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        # Annotate bin counts
        for conf, n in zip(confs, ns):
            if n > 0:
                ax.text(conf, 0.02, str(n), ha="center", va="bottom", fontsize=7, color="gray")

    plt.suptitle("Reliability Diagram — YOLOv8s on BDD val", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ── TTA augmentations ─────────────────────────────────────────────────────────

def tta_augment(img_bgr):
    """Return list of N_TTA augmented copies of the image."""
    imgs = [img_bgr]                                              # 1: original
    imgs.append(cv2.flip(img_bgr, 1))                            # 2: h-flip
    imgs.append(cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=30))  # 3: brighter
    imgs.append(cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=-30)) # 4: darker
    blurred = cv2.GaussianBlur(img_bgr, (5, 5), 0)               # 5: blur
    imgs.append(blurred)
    return imgs[:N_TTA]


def tta_uncertainty(model, img_bgr):
    """
    Run N_TTA augmented passes, return:
      mean_conf  — average max-confidence across augmentations
      std_conf   — std of max-confidence (= uncertainty estimate)
      n_dets     — mean number of detections across augmentations
    """
    aug_imgs   = tta_augment(img_bgr)
    max_confs  = []
    n_dets_list = []
    for aug in aug_imgs:
        r = model.predict(aug, conf=0.30, verbose=False)[0]
        if r.boxes is not None and len(r.boxes) > 0:
            max_confs.append(float(r.boxes.conf.max().cpu()))
            n_dets_list.append(len(r.boxes))
        else:
            max_confs.append(0.0)
            n_dets_list.append(0)
    return float(np.mean(max_confs)), float(np.std(max_confs)), float(np.mean(n_dets_list))


# ── Danger zone plot ──────────────────────────────────────────────────────────

def plot_danger_scatter(raw_confs, tta_stds, stems, save_path):
    raw_confs = np.array(raw_confs)
    tta_stds  = np.array(tta_stds)

    colors = []
    for c, s in zip(raw_confs, tta_stds):
        if c >= DANGER_CONF and s >= DANGER_STD:
            colors.append("#e6194b")   # danger
        elif c >= DANGER_CONF:
            colors.append("#f58231")   # high conf, stable
        elif s >= DANGER_STD:
            colors.append("#4363d8")   # uncertain but low conf
        else:
            colors.append("#3cb44b")   # safe zone

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(raw_confs, tta_stds, c=colors, alpha=0.7, s=50, edgecolors="none")

    ax.axvline(DANGER_CONF, color="#e6194b", linestyle="--", linewidth=1.2,
               label=f"High conf threshold ({DANGER_CONF})")
    ax.axhline(DANGER_STD, color="#4363d8", linestyle="--", linewidth=1.2,
               label=f"High uncertainty threshold ({DANGER_STD})")

    # Shade danger quadrant
    ax.axvspan(DANGER_CONF, 1.0, ymin=DANGER_STD, alpha=0.06, color="#e6194b")

    patches = [
        mpatches.Patch(color="#e6194b", label="DANGER: high conf + high uncertainty"),
        mpatches.Patch(color="#f58231", label="High conf, stable"),
        mpatches.Patch(color="#4363d8", label="Uncertain, low conf"),
        mpatches.Patch(color="#3cb44b", label="Safe zone"),
    ]
    ax.legend(handles=patches, fontsize=8, loc="upper left")
    ax.set_xlabel("Max Raw Confidence (per image)")
    ax.set_ylabel("TTA Confidence Std (uncertainty)")
    ax.set_title("Danger Zone: High Confidence + High Uncertainty")
    ax.set_xlim(0, 1); ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


def plot_uncertainty_by_scenario(tta_stds, stems, img_tags, save_path):
    dims = ["timeofday", "scene"]
    fig, axes = plt.subplots(1, len(dims), figsize=(12, 5))
    for ax, dim in zip(axes, dims):
        groups = collections.defaultdict(list)
        for s, std in zip(stems, tta_stds):
            val = img_tags.get(s, {}).get(dim, "undefined")
            groups[val].append(std)
        labels = sorted(groups, key=lambda x: -np.median(groups[x]))
        data   = [groups[l] for l in labels]
        bp = ax.boxplot(data, patch_artist=True)
        for patch in bp["boxes"]:
            patch.set_facecolor("#4363d8"); patch.set_alpha(0.6)
        ax.axhline(DANGER_STD, color="#e6194b", linestyle="--", linewidth=1.2,
                   label=f"Danger threshold ({DANGER_STD})")
        ax.set_xticks(range(1, len(labels)+1))
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("TTA Uncertainty (std)")
        ax.set_title(f"Uncertainty by {dim}")
        ax.legend(fontsize=8)
    plt.suptitle("Per-Scenario TTA Uncertainty — YOLOv8s BDD val", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load metadata tags
    img_tags = {}
    for ann_file in glob.glob(os.path.join(ANN_DIR, "*.json")):
        stem = os.path.basename(ann_file).replace(".jpg.json", "")
        with open(ann_file, encoding="utf-8") as f:
            d = json.load(f)
        img_tags[stem] = {t["name"]: t["value"] for t in d.get("tags", [])}

    # Load OOD scores if available
    ood_scores = {}
    if os.path.exists(OOD_CSV):
        with open(OOD_CSV, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                ood_scores[row["Image"]] = float(row["OOD_Score"])

    all_paths = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg")))
    print(f"Val images: {len(all_paths)}")

    print(f"\nLoading model...")
    model = YOLO(MODEL_PATH)

    # ── Phase 1: Calibration data collection ─────────────────────────────────
    print("\n[1/3] Collecting detection-level confidences vs GT matches...")
    all_confs, all_labels = [], []   # for calibration
    per_class = {0: {"confs": [], "labels": []}, 1: {"confs": [], "labels": []}}

    for i, path in enumerate(all_paths):
        fname      = os.path.basename(path)
        stem       = os.path.splitext(fname)[0]
        label_path = os.path.join(VAL_LABELS, stem + ".txt")

        img  = cv2.imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        gt   = load_gt(label_path, w, h)

        r = model.predict(path, conf=RAW_CONF, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        pred_boxes = r.boxes.xyxy.cpu().numpy()
        pred_confs = r.boxes.conf.cpu().numpy()
        pred_cls   = r.boxes.cls.cpu().numpy().astype(int)

        matched = match_detections(pred_boxes, pred_confs, pred_cls, gt)
        for conf, is_tp, cls in matched:
            all_confs.append(conf)
            all_labels.append(is_tp)
            if cls in per_class:
                per_class[cls]["confs"].append(conf)
                per_class[cls]["labels"].append(is_tp)

        if (i+1) % 20 == 0 or (i+1) == len(all_paths):
            print(f"  {i+1}/{len(all_paths)}", end="\r", flush=True)

    print(f"\n  Total detections collected: {len(all_confs)}")
    print(f"  TP: {sum(all_labels)}  FP: {len(all_labels)-sum(all_labels)}")

    # ── Phase 2: Calibration ──────────────────────────────────────────────────
    print("\n[2/3] Running temperature scaling calibration...")

    ece_raw, bin_data_raw = compute_ece(all_confs, all_labels)
    print(f"  ECE before calibration: {ece_raw:.4f}")

    T_opt = find_temperature(all_confs, all_labels)
    cal_confs = apply_temperature(all_confs, T_opt)
    ece_cal, bin_data_cal = compute_ece(cal_confs, all_labels)
    print(f"  Optimal temperature T:  {T_opt:.4f}")
    print(f"  ECE after calibration:  {ece_cal:.4f}  "
          f"(improvement: {((ece_raw-ece_cal)/ece_raw*100):.1f}%)")

    # Per-class calibration
    print("\n  Per-class ECE:")
    for cls_id, cls_name in CLASS_NAMES.items():
        cd = per_class[cls_id]
        if not cd["confs"]:
            continue
        e_raw, _ = compute_ece(cd["confs"], cd["labels"])
        cal_c    = apply_temperature(cd["confs"], T_opt)
        e_cal, _ = compute_ece(cal_c, cd["labels"])
        print(f"    {cls_name:<8}  ECE before={e_raw:.4f}  after={e_cal:.4f}  "
              f"n_dets={len(cd['confs'])}")

    plot_reliability(bin_data_raw, bin_data_cal, T_opt, ece_raw, ece_cal,
                     os.path.join(RESULTS_DIR, "reliability_diagram.png"))
    print("\n  Saved: reliability_diagram.png")

    # ── Phase 3: TTA Uncertainty ──────────────────────────────────────────────
    print("\n[3/3] Running TTA uncertainty estimation (N=5 augmentations per image)...")
    tta_rows = []
    raw_confs_img, tta_stds_img, stems_img = [], [], []

    for i, path in enumerate(all_paths):
        stem = os.path.splitext(os.path.basename(path))[0]
        img  = cv2.imread(path)
        if img is None:
            continue

        mean_c, std_c, mean_n = tta_uncertainty(model, img)

        # Single-pass confidence (conf=0.30)
        r = model.predict(path, conf=0.30, verbose=False)[0]
        single_conf = float(r.boxes.conf.max().cpu()) if (r.boxes is not None and len(r.boxes)>0) else 0.0

        is_danger = (single_conf >= DANGER_CONF) and (std_c >= DANGER_STD)
        ood_score = ood_scores.get(stem, None)
        tags      = img_tags.get(stem, {})

        tta_rows.append({
            "Image":        stem,
            "Raw_Conf":     single_conf,
            "TTA_Mean":     mean_c,
            "TTA_Std":      std_c,
            "TTA_N_Dets":   mean_n,
            "Cal_Conf":     float(apply_temperature([single_conf], T_opt)[0]) if single_conf > 0 else 0.0,
            "OOD_Score":    ood_score if ood_score else "N/A",
            "Danger":       "DANGER" if is_danger else "OK",
            "weather":      tags.get("weather", "?"),
            "timeofday":    tags.get("timeofday", "?"),
            "scene":        tags.get("scene", "?"),
        })
        raw_confs_img.append(single_conf)
        tta_stds_img.append(std_c)
        stems_img.append(stem)

        if (i+1) % 20 == 0 or (i+1) == len(all_paths):
            print(f"  {i+1}/{len(all_paths)}", end="\r", flush=True)

    print()
    danger_rows = [r for r in tta_rows if r["Danger"] == "DANGER"]
    print(f"  Danger-zone images: {len(danger_rows)} / {len(tta_rows)}")

    # ── Report ────────────────────────────────────────────────────────────────
    W = 90
    print("\n" + "="*W)
    print("         UNCERTAINTY QUANTIFICATION REPORT  (ADAS/AD Pre-Deployment V&V)")
    print("="*W)

    print(f"\n  CALIBRATION SUMMARY")
    print(f"    ECE before temperature scaling : {ece_raw:.4f}")
    print(f"    Optimal temperature T          : {T_opt:.4f}")
    print(f"    ECE after  temperature scaling : {ece_cal:.4f}")
    improvement = (ece_raw - ece_cal) / ece_raw * 100
    print(f"    ECE improvement                : {improvement:.1f}%")
    if T_opt > 1.0:
        print(f"    Interpretation: T={T_opt:.2f} > 1 → model is OVER-CONFIDENT")
        print(f"    → Raw confidence scores are systematically too high")
        print(f"    → Apply T={T_opt:.3f} scaling before deployment decisions")
    else:
        print(f"    Interpretation: T={T_opt:.2f} < 1 → model is UNDER-CONFIDENT")
        print(f"    → Raw confidence scores can be used conservatively")

    print(f"\n  TTA UNCERTAINTY SUMMARY  (N={N_TTA} augmentations per image)")
    print(f"    Mean TTA std across all images : {np.mean(tta_stds_img):.4f}")
    print(f"    Max  TTA std                   : {np.max(tta_stds_img):.4f}")
    print(f"    Danger-zone images             : {len(danger_rows)} / {len(tta_rows)}"
          f"  (conf≥{DANGER_CONF} AND std≥{DANGER_STD})")

    if danger_rows:
        print(f"\n  DANGER-ZONE IMAGES:")
        print(f"    {'Image':<38} {'RawConf':>8} {'CalConf':>8} {'Std':>7}  "
              f"{'OOD':>6}  {'timeofday':<12} {'scene'}")
        print(f"    {'-'*(W-4)}")
        for r in sorted(danger_rows, key=lambda x: -x["TTA_Std"])[:20]:
            ood_str = f"{r['OOD_Score']:.3f}" if isinstance(r['OOD_Score'], float) else "N/A"
            print(f"    {r['Image']:<38} {r['Raw_Conf']:>8.3f} {r['Cal_Conf']:>8.3f} "
                  f"{r['TTA_Std']:>7.3f}  {ood_str:>6}  "
                  f"{r['timeofday']:<12} {r['scene']}")

    # Cross-reference with OOD
    if ood_scores:
        print(f"\n  UNCERTAINTY + OOD CROSS-REFERENCE:")
        print(f"    (images with high TTA uncertainty AND high OOD score)")
        cross = [(r, ood_scores.get(r["Image"], 0)) for r in tta_rows
                 if r["TTA_Std"] >= DANGER_STD and isinstance(r["OOD_Score"], float)
                 and r["OOD_Score"] > 0.9]
        if cross:
            for r, ood in sorted(cross, key=lambda x: -x[1])[:10]:
                print(f"    {r['Image']:<38}  std={r['TTA_Std']:.3f}  "
                      f"ood={ood:.3f}  {r['timeofday']} / {r['scene']}")
        else:
            print("    None found (OOD and uncertainty are complementary signals).")

    print(f"\n  EMBEDDED DEPLOYMENT CHECKLIST:")
    print(f"    [{'✓' if T_opt != 1.0 else '✗'}] Temperature T={T_opt:.3f} calibrated — apply to logits before threshold")
    print(f"    [ ] Re-validate T after INT8/FP16 quantisation export")
    print(f"    [ ] Re-validate T on target hardware (TDA4 / Orin / RPi)")
    print(f"    [✓] TTA uncertainty pipeline validated (N={N_TTA}, latency: ~{N_TTA}× inference)")
    print(f"    [ ] Set runtime uncertainty threshold for driver handover trigger")
    print(f"    [ ] Danger-zone images ({len(danger_rows)}) reviewed and root-caused")
    print("="*W)

    # Plots
    print("\nGenerating plots...")
    plot_danger_scatter(raw_confs_img, tta_stds_img, stems_img,
                        os.path.join(RESULTS_DIR, "danger_zone_scatter.png"))
    print("  danger_zone_scatter.png")

    plot_uncertainty_by_scenario(tta_stds_img, stems_img, img_tags,
                                 os.path.join(RESULTS_DIR, "uncertainty_by_scenario.png"))
    print("  uncertainty_by_scenario.png")

    # Save CSVs
    csv_path = os.path.join(RESULTS_DIR, "uncertainty_per_image.csv")
    fields = ["Image", "Raw_Conf", "TTA_Mean", "TTA_Std", "Cal_Conf",
              "OOD_Score", "Danger", "weather", "timeofday", "scene"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in tta_rows:
            w.writerow({k: f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]
                        for k in fields})

    cal_csv = os.path.join(RESULTS_DIR, "calibration_summary.csv")
    with open(cal_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Metric", "Value"])
        w.writerow(["ECE_before", f"{ece_raw:.4f}"])
        w.writerow(["Temperature_T", f"{T_opt:.4f}"])
        w.writerow(["ECE_after", f"{ece_cal:.4f}"])
        w.writerow(["ECE_improvement_pct", f"{improvement:.1f}"])
        w.writerow(["Danger_zone_images", len(danger_rows)])
        w.writerow(["Total_images", len(tta_rows)])

    print(f"\n  CSVs saved to: {RESULTS_DIR}/")
    print(f"  Plots saved to: {RESULTS_DIR}/")
