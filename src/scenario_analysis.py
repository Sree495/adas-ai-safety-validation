"""
Scenario Coverage Analysis  (ADAS/AD V&V)

Parses BDD val annotation JSON files to extract scenario metadata tags
(weather, timeofday, scene), groups val images by each tag value, runs
per-group inference against ground truth, and reports:
  - Image count (coverage)
  - Per-class Precision, Recall, F1
  - mAP50 (area under precision-recall curve, IoU ≥ 0.5)
  - Coverage gap flag (< MIN_IMAGES images OR recall < RECALL_GAP)

Prerequisites: augment.py is NOT required — runs directly on val images.

Usage:
    python scenario_analysis.py
"""

import os
import json
import glob
import csv
import collections
import numpy as np
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
from config import (MODEL_PATH, VAL_IMAGES, VAL_LABELS, ANN_DIR,
                    RESULTS_SCENARIO as RESULTS_DIR)

CONF        = 0.30
IOU_THRESH  = 0.50     # IoU threshold for TP matching
MIN_IMAGES  = 8        # below this → coverage gap (too few samples)
RECALL_GAP  = 0.35     # below this → performance gap
CLASS_NAMES = {0: "person", 1: "car"}
TAG_DIMS    = ["weather", "timeofday", "scene"]

# ── Utility functions ─────────────────────────────────────────────────────────

def load_gt_labels(label_path):
    """Load YOLO-format ground-truth labels → list of (class_id, cx, cy, w, h)."""
    if not os.path.exists(label_path):
        return []
    boxes = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
                boxes.append((cls, cx, cy, w, h))
    return boxes


def yolo_to_xyxy(cx, cy, w, h, img_w=1280, img_h=720):
    """Convert YOLO normalised cx,cy,w,h to absolute x1,y1,x2,y2."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return x1, y1, x2, y2


def iou(box_a, box_b):
    """Compute IoU between two (x1,y1,x2,y2) boxes."""
    xa1 = max(box_a[0], box_b[0])
    ya1 = max(box_a[1], box_b[1])
    xa2 = min(box_a[2], box_b[2])
    ya2 = min(box_a[3], box_b[3])
    inter = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    if inter == 0:
        return 0.0
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def compute_ap(precisions, recalls):
    """Compute AP using the 11-point interpolation (PASCAL VOC style)."""
    ap = 0.0
    for thr in np.linspace(0, 1, 11):
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= thr]
        ap += (max(p_at_r) if p_at_r else 0.0) / 11
    return ap


def eval_group(model, image_paths):
    """
    Run inference on a list of image paths, compare to GT labels.

    Returns per-class dict with:
        ap50, precision, recall, f1, n_gt, n_pred
    """
    # Accumulate per-class detection lists: list of (conf, is_tp)
    detections = collections.defaultdict(list)   # class_id → [(conf, is_tp), ...]
    n_gt       = collections.defaultdict(int)    # class_id → total GT boxes

    for img_path in image_paths:
        fname      = os.path.basename(img_path)
        label_path = os.path.join(VAL_LABELS, os.path.splitext(fname)[0] + ".txt")
        gt_boxes   = load_gt_labels(label_path)

        # Count GT per class
        gt_by_class = collections.defaultdict(list)
        for cls, cx, cy, w, h in gt_boxes:
            if cls in CLASS_NAMES:
                box_abs = yolo_to_xyxy(cx, cy, w, h)
                gt_by_class[cls].append(box_abs)
                n_gt[cls] += 1

        # Run inference
        results = model.predict(img_path, conf=CONF, verbose=False)
        if not results:
            continue
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs      = r.boxes.conf.cpu().numpy()
        classes    = r.boxes.cls.cpu().numpy().astype(int)

        # Match predictions to GT per class
        for pred_cls in CLASS_NAMES:
            gt_list    = gt_by_class.get(pred_cls, [])
            matched_gt = [False] * len(gt_list)

            pred_mask = classes == pred_cls
            pred_boxes = boxes_xyxy[pred_mask]
            pred_confs = confs[pred_mask]

            # Sort by confidence descending
            order = np.argsort(-pred_confs)
            for idx in order:
                pbox = pred_boxes[idx]
                conf = pred_confs[idx]
                best_iou, best_j = 0.0, -1
                for j, gbox in enumerate(gt_list):
                    if matched_gt[j]:
                        continue
                    v = iou(pbox, gbox)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= IOU_THRESH and best_j >= 0:
                    matched_gt[best_j] = True
                    detections[pred_cls].append((conf, True))
                else:
                    detections[pred_cls].append((conf, False))

    # Compute per-class AP50, precision, recall
    results_out = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        total_gt = n_gt.get(cls_id, 0)
        dets     = sorted(detections.get(cls_id, []), key=lambda x: -x[0])

        if total_gt == 0 and not dets:
            results_out[cls_name] = dict(ap50=0.0, precision=0.0, recall=0.0, f1=0.0,
                                          n_gt=0, n_pred=0)
            continue

        tp_cum, fp_cum = 0, 0
        prec_list, rec_list = [], []
        for _, is_tp in dets:
            if is_tp:
                tp_cum += 1
            else:
                fp_cum += 1
            p = tp_cum / (tp_cum + fp_cum)
            r = tp_cum / total_gt if total_gt > 0 else 0.0
            prec_list.append(p)
            rec_list.append(r)

        ap50 = compute_ap(prec_list, rec_list) if prec_list else 0.0
        final_tp = sum(1 for _, tp in dets if tp)
        final_fp = len(dets) - final_tp
        precision = final_tp / len(dets) if dets else 0.0
        recall    = final_tp / total_gt if total_gt > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results_out[cls_name] = dict(ap50=ap50, precision=precision, recall=recall,
                                      f1=f1, n_gt=total_gt, n_pred=len(dets))

    return results_out


def mean_ap50(cls_metrics):
    """Return mean AP50 across classes that have GT boxes."""
    vals = [v["ap50"] for v in cls_metrics.values() if v["n_gt"] > 0]
    return sum(vals) / len(vals) if vals else 0.0


def mean_recall(cls_metrics):
    vals = [v["recall"] for v in cls_metrics.values() if v["n_gt"] > 0]
    return sum(vals) / len(vals) if vals else 0.0


def gap_flag(n_imgs, recall):
    flags = []
    if n_imgs < MIN_IMAGES:
        flags.append(f"LOW-COVERAGE (<{MIN_IMAGES} imgs)")
    if recall < RECALL_GAP:
        flags.append(f"PERF-GAP (recall<{RECALL_GAP})")
    return " | ".join(flags) if flags else "OK"


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # 1. Parse annotation JSONs → per-image tag dict
    print("Parsing BDD annotation metadata...")
    img_tags = {}   # image_stem → {weather, timeofday, scene}
    for ann_file in glob.glob(os.path.join(ANN_DIR, "*.json")):
        stem = os.path.basename(ann_file).replace(".jpg.json", "")
        with open(ann_file, encoding="utf-8") as f:
            data = json.load(f)
        tags = {t["name"]: t["value"] for t in data.get("tags", [])
                if t["name"] in TAG_DIMS}
        img_tags[stem] = tags

    print(f"  Parsed {len(img_tags)} annotations.")

    # 2. Group image paths by each tag dimension
    groups = {}   # (dimension, value) → [image_path, ...]
    for stem, tags in img_tags.items():
        img_path = os.path.join(VAL_IMAGES, stem + ".jpg")
        if not os.path.exists(img_path):
            continue
        for dim in TAG_DIMS:
            val = tags.get(dim, "undefined")
            key = (dim, val)
            groups.setdefault(key, []).append(img_path)

    print(f"  Found {len(groups)} scenario groups across {len(TAG_DIMS)} tag dimensions.\n")

    # 3. Load model
    print(f"Loading model: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 4. Evaluate each group
    print("\nRunning per-scenario evaluation...\n")
    rows = []
    for (dim, val), paths in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        n = len(paths)
        print(f"  [{dim}={val}]  {n} images", end="  ", flush=True)
        cls_m = eval_group(model, paths)
        map50  = mean_ap50(cls_m)
        recall = mean_recall(cls_m)
        flag   = gap_flag(n, recall)
        print(f"mAP50={map50:.3f}  recall={recall:.3f}  {flag}")
        rows.append({
            "Dimension": dim,
            "Scenario":  val,
            "N_Images":  n,
            "mAP50":     map50,
            "Recall":    recall,
            "Precision": sum(v["precision"] for v in cls_m.values() if v["n_gt"] > 0)
                         / max(1, sum(1 for v in cls_m.values() if v["n_gt"] > 0)),
            "Person_mAP50":   cls_m["person"]["ap50"],
            "Person_Recall":  cls_m["person"]["recall"],
            "Car_mAP50":      cls_m["car"]["ap50"],
            "Car_Recall":     cls_m["car"]["recall"],
            "Flag":      flag,
        })

    # 5. Print formatted report
    W = 100
    print("\n\n" + "=" * W)
    print("         SCENARIO COVERAGE & PERFORMANCE ANALYSIS  (ADAS/AD V&V)")
    print(f"         Model: YOLOv8s  |  Conf: {CONF}  |  IoU: {IOU_THRESH}  |  Dataset: BDD-sample")
    print("=" * W)

    for dim in TAG_DIMS:
        dim_rows = [r for r in rows if r["Dimension"] == dim]
        if not dim_rows:
            continue
        print(f"\n  TAG: {dim.upper()}")
        hdr = f"    {'Scenario':<20} {'N':>5} {'mAP50':>7} {'Recall':>8} {'Precision':>10}  {'Person R':>9} {'Car R':>6}  {'Flag'}"
        print(hdr)
        print(f"    {'-'*(W-4)}")
        for r in sorted(dim_rows, key=lambda x: -x["mAP50"]):
            print(
                f"    {r['Scenario']:<20} {r['N_Images']:>5} "
                f"{r['mAP50']:>7.3f} {r['Recall']:>8.3f} {r['Precision']:>10.3f}  "
                f"{r['Person_Recall']:>9.3f} {r['Car_Recall']:>6.3f}  "
                f"{r['Flag']}"
            )

    print("\n" + "=" * W)

    # 6. Coverage gap summary
    gaps = [r for r in rows if r["Flag"] != "OK"]
    if gaps:
        print("\n  COVERAGE / PERFORMANCE GAPS DETECTED:")
        for r in gaps:
            print(f"    → {r['Dimension']}={r['Scenario']:<18}  N={r['N_Images']:>3}  "
                  f"mAP50={r['mAP50']:.3f}  recall={r['Recall']:.3f}  [{r['Flag']}]")
        print("\n  Recommendation: Collect/augment data for flagged scenarios.")
        print("  Document as known ODD gaps per ISO 21448 (SOTIF) boundary analysis.")
    else:
        print("\n  No coverage or performance gaps detected.")

    print("=" * W)

    # 7. Save CSV
    csv_path = os.path.join(RESULTS_DIR, "scenario_analysis.csv")
    fieldnames = ["Dimension", "Scenario", "N_Images", "mAP50", "Recall", "Precision",
                  "Person_mAP50", "Person_Recall", "Car_mAP50", "Car_Recall", "Flag"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]
                              for k in fieldnames})

    print(f"\n  CSV saved to: {csv_path}")
