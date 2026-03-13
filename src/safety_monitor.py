"""
Runtime Safety Monitor  (ADAS/AD — ISO 21448 SOTIF / ISO 26262)

Integrates all V&V signals into a single per-frame safety verdict:

    ┌─────────────────────────────────────────────────────────────┐
    │  Camera frame                                               │
    │       │                                                     │
    │       ├─► YOLOv8 detection  → calibrated confidence        │
    │       ├─► OOD score         → Mahalanobis distance         │
    │       ├─► TTA uncertainty   → std over 5 augmented passes  │
    │       │                                                     │
    │       └─► SafetyMonitor.assess() → GO / CAUTION / STOP     │
    └─────────────────────────────────────────────────────────────┘

Decision rules:
    STOP    ood_score ≥ OOD_STOP  or  tta_std ≥ UNC_STOP
    CAUTION ood_score ≥ OOD_WARN  or  tta_std ≥ UNC_WARN
              or calibrated_conf < CONF_WARN (detections present but unreliable)
    GO      all signals within safe bounds

Usage:
    # As a module
    from safety_monitor import SafetyMonitor
    monitor = SafetyMonitor()
    verdict = monitor.assess(img_bgr)

    # Standalone demo on val images
    venv/Scripts/python.exe src/safety_monitor.py
"""

import os
import glob
import json
import csv
import random
import numpy as np
import cv2
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────────────
from config import MODEL_PATH, VAL_IMAGES, ANN_DIR, RESULTS_SAFETY_MONITOR as RESULTS_DIR

# Nominal distribution: images to fit the OOD baseline on
NOMINAL_TAGS = {"weather": "clear", "timeofday": "daytime"}

# Temperature scaling factor (from uncertainty_eval.py)
TEMPERATURE  = 0.9936

# Decision thresholds
OOD_STOP    = 1.10   # STOP  if ood_score ≥ this
OOD_WARN    = 0.95   # CAUTION if ood_score ≥ this
UNC_STOP    = 0.20   # STOP  if tta_std ≥ this
UNC_WARN    = 0.10   # CAUTION if tta_std ≥ this
CONF_WARN   = 0.45   # CAUTION if best calibrated conf < this (when dets exist)

CONF_THRESH = 0.30   # detection confidence gate
N_TTA       = 5      # TTA passes
IMG_SIZE    = 640
CLASS_NAMES = {0: "person", 1: "car"}

DECISION_COLOR = {
    "GO":      (34,  139, 34),    # green  (BGR)
    "CAUTION": (0,   165, 255),   # orange
    "STOP":    (0,   0,   220),   # red
}

# ── Pre-processing ────────────────────────────────────────────────────────────

def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return torch.from_numpy(img_res).permute(2, 0, 1).float().unsqueeze(0) / 255.0

# ── TTA augmentations ─────────────────────────────────────────────────────────

def tta_variants(img_bgr):
    return [
        img_bgr,
        cv2.flip(img_bgr, 1),
        cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=30),
        cv2.convertScaleAbs(img_bgr, alpha=1.0, beta=-30),
        cv2.GaussianBlur(img_bgr, (5, 5), 0),
    ][:N_TTA]

# ── Temperature scaling ───────────────────────────────────────────────────────

def calibrate(conf, T=TEMPERATURE):
    c = float(np.clip(conf, 1e-7, 1 - 1e-7))
    logit = np.log(c / (1 - c))
    return float(1.0 / (1.0 + np.exp(-logit / T)))

# ── OOD: PCA + Mahalanobis ────────────────────────────────────────────────────

class OODModel:
    """Fits a Mahalanobis-in-PCA-space OOD detector on nominal images."""

    def __init__(self, model_nn, nominal_paths, n_components=26):
        self._nn   = model_nn
        self._feat = {}
        self._hook = model_nn.model[9].register_forward_hook(
            lambda m, i, o: self._feat.update({"f": o})
        )
        self.mu   = None
        self.comp = None
        self.pca_mu  = None
        self.pca_std = None
        self._fit(nominal_paths, n_components)
        self._hook.remove()

    def _extract(self, img_bgr):
        t = preprocess(img_bgr)
        with torch.no_grad():
            self._nn(t)
        return self._feat["f"].mean(dim=[2, 3]).squeeze(0).detach().cpu().numpy()

    def _fit(self, paths, n_comp):
        feats = np.array([self._extract(cv2.imread(p)) for p in paths
                          if cv2.imread(p) is not None], dtype=np.float32)
        self.mu = feats.mean(axis=0)
        Xc = feats - self.mu
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.comp = Vt[:n_comp]                              # (n_comp, 512)
        pca = (feats - self.mu) @ self.comp.T
        self.pca_mu  = pca.mean(axis=0)
        self.pca_std = pca.std(axis=0) + 1e-8

    def score(self, img_bgr):
        """Return Mahalanobis OOD score for one image."""
        h = self._nn.model[9].register_forward_hook(
            lambda m, i, o: self._feat.update({"f": o})
        )
        t = preprocess(img_bgr)
        with torch.no_grad():
            self._nn(t)
        h.remove()
        feat = self._feat["f"].mean(dim=[2, 3]).squeeze(0).detach().cpu().numpy()
        pca  = (feat - self.mu) @ self.comp.T
        return float(np.sqrt(((pca - self.pca_mu) / self.pca_std) ** 2).mean())

# ── SafetyMonitor ─────────────────────────────────────────────────────────────

class SafetyMonitor:
    """
    Per-frame safety assessor. Call assess(img_bgr) → verdict dict.

    verdict keys:
        decision        : "GO" | "CAUTION" | "STOP"
        reasons         : list of triggered rule strings
        ood_score       : float
        tta_std         : float
        raw_conf        : float  (best detection confidence, 0 if no dets)
        cal_conf        : float  (temperature-scaled)
        n_detections    : int
        detections      : list of {"class", "conf", "cal_conf", "box_xyxy"}
    """

    def __init__(self, val_images_dir=VAL_IMAGES, ann_dir=ANN_DIR,
                 nominal_tags=NOMINAL_TAGS):
        print("Initialising SafetyMonitor...")
        self.model = YOLO(MODEL_PATH)
        self._nn   = self.model.model

        # Identify nominal images for OOD baseline
        nominal_paths = self._nominal_paths(val_images_dir, ann_dir, nominal_tags)
        print(f"  Fitting OOD model on {len(nominal_paths)} nominal images...")
        self.ood = OODModel(self._nn, nominal_paths)
        print("  SafetyMonitor ready.\n")

    @staticmethod
    def _nominal_paths(img_dir, ann_dir, nominal_tags):
        paths = []
        for p in glob.glob(os.path.join(img_dir, "*.jpg")):
            stem = os.path.splitext(os.path.basename(p))[0]
            af   = os.path.join(ann_dir, stem + ".jpg.json")
            if not os.path.exists(af):
                continue
            with open(af, encoding="utf-8") as f:
                tags = {t["name"]: t["value"] for t in json.load(f).get("tags", [])}
            if all(tags.get(k) == v for k, v in nominal_tags.items()):
                paths.append(p)
        return paths

    def assess(self, img_bgr):
        """Run all checks on one frame. Returns verdict dict."""
        reasons = []

        # ── 1. Detection ──────────────────────────────────────────────────────
        r = self.model.predict(img_bgr, conf=CONF_THRESH, verbose=False)[0]
        detections = []
        raw_conf   = 0.0
        if r.boxes is not None and len(r.boxes) > 0:
            for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy(),
                                       r.boxes.cls.cpu().numpy().astype(int)):
                c = float(conf)
                detections.append({
                    "class":    CLASS_NAMES.get(cls, str(cls)),
                    "conf":     c,
                    "cal_conf": calibrate(c),
                    "box_xyxy": box.tolist(),
                })
            raw_conf = max(d["conf"] for d in detections)

        cal_conf = calibrate(raw_conf) if raw_conf > 0 else 0.0

        # ── 2. OOD score ──────────────────────────────────────────────────────
        ood_score = self.ood.score(img_bgr)

        # ── 3. TTA uncertainty ────────────────────────────────────────────────
        tta_confs = []
        for aug in tta_variants(img_bgr):
            ra = self.model.predict(aug, conf=CONF_THRESH, verbose=False)[0]
            if ra.boxes is not None and len(ra.boxes) > 0:
                tta_confs.append(float(ra.boxes.conf.max().cpu()))
            else:
                tta_confs.append(0.0)
        tta_std = float(np.std(tta_confs))

        # ── 4. Decision logic ─────────────────────────────────────────────────
        if ood_score >= OOD_STOP:
            reasons.append(f"OOD score {ood_score:.3f} ≥ STOP threshold {OOD_STOP}")
        if tta_std >= UNC_STOP:
            reasons.append(f"TTA uncertainty {tta_std:.3f} ≥ STOP threshold {UNC_STOP}")

        if reasons:
            decision = "STOP"
        else:
            if ood_score >= OOD_WARN:
                reasons.append(f"OOD score {ood_score:.3f} ≥ CAUTION threshold {OOD_WARN}")
            if tta_std >= UNC_WARN:
                reasons.append(f"TTA uncertainty {tta_std:.3f} ≥ CAUTION threshold {UNC_WARN}")
            if detections and cal_conf < CONF_WARN:
                reasons.append(f"Calibrated confidence {cal_conf:.3f} < {CONF_WARN} (unreliable detections)")
            decision = "CAUTION" if reasons else "GO"

        return {
            "decision":     decision,
            "reasons":      reasons,
            "ood_score":    ood_score,
            "tta_std":      tta_std,
            "raw_conf":     raw_conf,
            "cal_conf":     cal_conf,
            "n_detections": len(detections),
            "detections":   detections,
        }

# ── Visualisation ─────────────────────────────────────────────────────────────

def render_verdict(img_bgr, verdict):
    """Draw detections + safety badge onto image copy."""
    out = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)).copy()
    sx  = IMG_SIZE / img_bgr.shape[1]
    sy  = IMG_SIZE / img_bgr.shape[0]

    # Draw boxes
    for d in verdict["detections"]:
        x1, y1, x2, y2 = [int(v * (sx if i % 2 == 0 else sy))
                           for i, v in enumerate(d["box_xyxy"])]
        col = (100, 255, 100) if d["class"] == "car" else (100, 100, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cv2.putText(out, f"{d['class']} {d['conf']:.2f}",
                    (x1, max(y1 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

    # Safety badge (top-left overlay)
    dec   = verdict["decision"]
    color = DECISION_COLOR[dec]
    badge_h = 70
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (IMG_SIZE, badge_h), color, -1)
    cv2.addWeighted(overlay, 0.75, out, 0.25, 0, out)

    cv2.putText(out, dec, (10, 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out,
                f"OOD:{verdict['ood_score']:.2f}  Unc:{verdict['tta_std']:.2f}  "
                f"Conf:{verdict['cal_conf']:.2f}  Dets:{verdict['n_detections']}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    if verdict["reasons"]:
        reason_short = verdict["reasons"][0][:70]
        cv2.putText(out, reason_short, (10, IMG_SIZE - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def save_verdict_grid(entries, save_path, title, cols=4):
    rows = (len(entries) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.array(axes).reshape(-1)
    for ax, (img, verdict) in zip(axes, entries):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        dec = verdict["decision"]
        ax.set_title(dec, fontsize=9, fontweight="bold",
                     color={"GO": "green", "CAUTION": "darkorange", "STOP": "red"}[dec])
        ax.axis("off")
    for ax in axes[len(entries):]:
        ax.axis("off")
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close()

# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(42)

    monitor = SafetyMonitor()

    # Parse metadata
    img_tags = {}
    for af in glob.glob(os.path.join(ANN_DIR, "*.json")):
        stem = os.path.basename(af).replace(".jpg.json", "")
        with open(af, encoding="utf-8") as f:
            img_tags[stem] = {t["name"]: t["value"]
                              for t in json.load(f).get("tags", [])}

    all_paths = sorted(glob.glob(os.path.join(VAL_IMAGES, "*.jpg")))

    # Run monitor on all val images
    print(f"Running safety monitor on {len(all_paths)} val images...\n")
    results = []
    for i, path in enumerate(all_paths):
        stem    = os.path.splitext(os.path.basename(path))[0]
        img_bgr = cv2.imread(path)
        verdict = monitor.assess(img_bgr)
        tags    = img_tags.get(stem, {})
        results.append({
            "image":     stem,
            "decision":  verdict["decision"],
            "ood_score": verdict["ood_score"],
            "tta_std":   verdict["tta_std"],
            "raw_conf":  verdict["raw_conf"],
            "cal_conf":  verdict["cal_conf"],
            "n_dets":    verdict["n_detections"],
            "reasons":   " | ".join(verdict["reasons"]),
            "weather":   tags.get("weather", "?"),
            "timeofday": tags.get("timeofday", "?"),
            "scene":     tags.get("scene", "?"),
            "_verdict":  verdict,
            "_img":      img_bgr,
        })
        dec = verdict["decision"]
        marker = {"GO": "✓", "CAUTION": "△", "STOP": "✖"}[dec]
        print(f"  [{i+1:>3}/{len(all_paths)}] {marker} {dec:<8}  "
              f"ood={verdict['ood_score']:.3f}  unc={verdict['tta_std']:.3f}  "
              f"conf={verdict['cal_conf']:.3f}  {tags.get('timeofday','?')}/{tags.get('scene','?')}")

    # ── Summary ───────────────────────────────────────────────────────────────
    counts = {d: sum(1 for r in results if r["decision"] == d)
              for d in ["GO", "CAUTION", "STOP"]}

    W = 82
    print("\n" + "=" * W)
    print("         RUNTIME SAFETY MONITOR — BATCH REPORT  (ISO 21448 SOTIF)")
    print("=" * W)
    total = len(results)
    for dec, cnt in counts.items():
        bar = "█" * int(cnt / total * 40)
        print(f"  {dec:<8} {cnt:>4} / {total}  ({cnt/total*100:>5.1f}%)  {bar}")

    # Per scenario breakdown
    for dim in ["timeofday", "scene"]:
        print(f"\n  BY {dim.upper()}:")
        groups = {}
        for r in results:
            k = r[dim]
            groups.setdefault(k, []).append(r["decision"])
        for tag, decs in sorted(groups.items()):
            n = len(decs)
            go = decs.count("GO");  ca = decs.count("CAUTION");  st = decs.count("STOP")
            print(f"    {tag:<18}  n={n:>3}  GO={go:>3}  CAUTION={ca:>3}  STOP={st:>3}")

    # Top STOP/CAUTION reasons
    stop_rows    = [r for r in results if r["decision"] == "STOP"]
    caution_rows = [r for r in results if r["decision"] == "CAUTION"]
    print(f"\n  STOP decisions ({len(stop_rows)}):")
    for r in sorted(stop_rows, key=lambda x: -x["ood_score"])[:10]:
        print(f"    {r['image']:<38}  ood={r['ood_score']:.3f}  unc={r['tta_std']:.3f}"
              f"  {r['timeofday']}/{r['scene']}")
        if r["reasons"]:
            print(f"      → {r['reasons'][:80]}")

    print(f"\n  CAUTION decisions ({len(caution_rows)}):")
    for r in sorted(caution_rows, key=lambda x: -x["ood_score"])[:8]:
        print(f"    {r['image']:<38}  ood={r['ood_score']:.3f}  unc={r['tta_std']:.3f}"
              f"  {r['timeofday']}/{r['scene']}")

    print("\n" + "=" * W)

    # ── Verdict grid plots ────────────────────────────────────────────────────
    print("\nGenerating verdict images...")

    # Plot A: STOP frames
    if stop_rows:
        entries = [(render_verdict(r["_img"], r["_verdict"]), r["_verdict"])
                   for r in stop_rows[:8]]
        save_verdict_grid(entries,
                          os.path.join(RESULTS_DIR, "A_stop_frames.png"),
                          "STOP Decisions — Safety Monitor",
                          cols=min(4, len(entries)))
        print("  A_stop_frames.png")

    # Plot B: CAUTION frames
    if caution_rows:
        sample = random.sample(caution_rows, min(8, len(caution_rows)))
        entries = [(render_verdict(r["_img"], r["_verdict"]), r["_verdict"])
                   for r in sample]
        save_verdict_grid(entries,
                          os.path.join(RESULTS_DIR, "B_caution_frames.png"),
                          "CAUTION Decisions — Safety Monitor",
                          cols=4)
        print("  B_caution_frames.png")

    # Plot C: GO frames (random sample)
    go_rows = [r for r in results if r["decision"] == "GO"]
    if go_rows:
        sample = random.sample(go_rows, min(8, len(go_rows)))
        entries = [(render_verdict(r["_img"], r["_verdict"]), r["_verdict"])
                   for r in sample]
        save_verdict_grid(entries,
                          os.path.join(RESULTS_DIR, "C_go_frames.png"),
                          "GO Decisions — Safety Monitor",
                          cols=4)
        print("  C_go_frames.png")

    # Plot D: Signal dashboard — scatter of OOD vs uncertainty, coloured by decision
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    color_map = {"GO": "#3cb44b", "CAUTION": "#f58231", "STOP": "#e6194b"}
    for dec in ["GO", "CAUTION", "STOP"]:
        sub = [r for r in results if r["decision"] == dec]
        if not sub:
            continue
        axes[0].scatter([r["ood_score"] for r in sub],
                        [r["tta_std"]   for r in sub],
                        c=color_map[dec], label=dec, alpha=0.75, s=50, edgecolors="none")
        axes[1].scatter([r["ood_score"] for r in sub],
                        [r["cal_conf"]  for r in sub],
                        c=color_map[dec], label=dec, alpha=0.75, s=50, edgecolors="none")

    for ax in axes:
        ax.axvline(OOD_WARN, color="#f58231", linestyle="--", linewidth=1, label=f"CAUTION ({OOD_WARN})")
        ax.axvline(OOD_STOP, color="#e6194b", linestyle="--", linewidth=1, label=f"STOP ({OOD_STOP})")
        ax.legend(fontsize=8)

    axes[0].axhline(UNC_WARN, color="#f58231", linestyle=":", linewidth=1)
    axes[0].axhline(UNC_STOP, color="#e6194b", linestyle=":", linewidth=1)
    axes[0].set_xlabel("OOD Score"); axes[0].set_ylabel("TTA Uncertainty (std)")
    axes[0].set_title("OOD Score vs Uncertainty")

    axes[1].axhline(CONF_WARN, color="#f58231", linestyle=":", linewidth=1,
                    label=f"CONF WARN ({CONF_WARN})")
    axes[1].set_xlabel("OOD Score"); axes[1].set_ylabel("Calibrated Confidence")
    axes[1].set_title("OOD Score vs Calibrated Confidence")

    plt.suptitle("Safety Monitor Signal Space — All Val Images", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "D_signal_dashboard.png"), dpi=120)
    plt.close()
    print("  D_signal_dashboard.png")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "safety_monitor_results.csv")
    fields   = ["image", "decision", "ood_score", "tta_std", "raw_conf",
                "cal_conf", "n_dets", "weather", "timeofday", "scene", "reasons"]
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow({k: f"{r[k]:.4f}" if isinstance(r[k], float) else r[k]
                        for k in fields})

    print(f"\n  CSV  → {csv_path}")
    print(f"  Plots → {RESULTS_DIR}/")
