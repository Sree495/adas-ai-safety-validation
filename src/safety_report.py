"""
SOTIF Safety Boundary Report  (ISO 21448)

Runs the trained model on nominal val + 5 adverse augmented conditions,
compares KPIs against baseline, and flags risk levels.

Prerequisites: Run augment.py first.

Usage:
    python safety_report.py
"""

import os
import csv
from ultralytics import YOLO

from config import MODEL_PATH, NOMINAL_YAML, AUG_BASE, RESULTS_SAFETY_REPORT as RESULTS_DIR
CONF         = 0.30

# ── SOTIF / ISO 26262 acceptance thresholds ───────────────────────────────────
# Adjust these to match your HARA / safety requirements document
THRESHOLDS = {
    "recall_critical": 0.20,   # below → CRITICAL
    "recall_high":     0.30,   # below → HIGH
    "recall_medium":   0.40,   # below → MEDIUM
    "map50_critical":  0.25,   # below → CRITICAL
    "map50_high":      0.35,   # below → HIGH
    "map50_medium":    0.50,   # below → MEDIUM
}

CONDITIONS = ["rain", "night", "fog", "noise", "glare"]


def risk_level(map50, recall):
    if recall < THRESHOLDS["recall_critical"] or map50 < THRESHOLDS["map50_critical"]:
        return "CRITICAL ✖"
    if recall < THRESHOLDS["recall_high"] or map50 < THRESHOLDS["map50_high"]:
        return "HIGH ⚠"
    if recall < THRESHOLDS["recall_medium"] or map50 < THRESHOLDS["map50_medium"]:
        return "MEDIUM △"
    return "LOW ✓"


def run_val(model, yaml_path, name):
    m = model.val(
        data=yaml_path,
        split="val",
        device="cpu",
        workers=0,
        conf=CONF,
        plots=True,
        verbose=False,
        project=RESULTS_DIR,
        name=name,
    )
    # per-class: index 0=person, 1=car
    return {
        "map50":       m.box.map50,
        "map5095":     m.box.map,
        "precision":   m.box.mp,
        "recall":      m.box.mr,
        "person_map50": m.box.ap50[0] if len(m.box.ap50) > 0 else 0,
        "car_map50":    m.box.ap50[1] if len(m.box.ap50) > 1 else 0,
        "person_recall": m.box.r[0] if len(m.box.r) > 0 else 0,
        "car_recall":    m.box.r[1] if len(m.box.r) > 1 else 0,
    }


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    rows = []

    # ── Nominal baseline ──────────────────────────────────────────────────────
    print("\n[1/6] Running nominal baseline (val set)...")
    base = run_val(model, NOMINAL_YAML, "nominal")
    rows.append(("Nominal", base, "BASELINE"))

    # ── Augmented conditions ──────────────────────────────────────────────────
    for i, cond in enumerate(CONDITIONS, start=2):
        yaml_path = os.path.join(AUG_BASE, f"{cond}.yaml")
        if not os.path.exists(yaml_path):
            print(f"[{i}/6] Skipping {cond}: YAML not found — run augment.py first.")
            continue
        print(f"[{i}/6] Running {cond}...")
        kpis = run_val(model, yaml_path, cond)
        risk = risk_level(kpis["map50"], kpis["recall"])
        rows.append((cond.capitalize(), kpis, risk))

    # ── Print report ──────────────────────────────────────────────────────────
    W = 82
    print("\n")
    print("=" * W)
    print("         SOTIF SAFETY BOUNDARY REPORT  (ISO 21448 / ISO 26262)")
    print(f"         Model: YOLOv8s  |  Confidence: {CONF}  |  Dataset: BDD-sample")
    print("=" * W)
    header = f"  {'Condition':<13} {'mAP50':>7} {'Recall':>8} {'Precision':>10}  {'Person R':>9} {'Car R':>7}  {'SOTIF Risk':>13}"
    print(header)
    print(f"  {'-'*(W-2)}")

    for name, kpis, risk in rows:
        if name == "Nominal":
            delta = ""
        else:
            dm = base["map50"] - kpis["map50"]
            dr = base["recall"] - kpis["recall"]
            delta = f"  Δmap={-dm:+.3f}  Δrecall={-dr:+.3f}"

        print(
            f"  {name:<13} "
            f"{kpis['map50']:>7.3f} "
            f"{kpis['recall']:>8.3f} "
            f"{kpis['precision']:>10.3f}  "
            f"{kpis['person_recall']:>9.3f} "
            f"{kpis['car_recall']:>7.3f}  "
            f"{risk:>13}"
            f"{delta}"
        )

    print("=" * W)

    # ── SOTIF findings ────────────────────────────────────────────────────────
    findings = [(n, r, k) for n, k, r in rows if "HIGH" in r or "CRITICAL" in r or "MEDIUM" in r]
    if findings:
        print("\n⚠  SOTIF FINDINGS:")
        for name, risk, kpis in findings:
            print(f"   → {name:<10} {risk}  "
                  f"(mAP50={kpis['map50']:.3f}, recall={kpis['recall']:.3f})")
        print("\n   Action: Document as known SOTIF boundary. Trigger condition")
        print("   must be addressed in ODD (Operational Design Domain) definition.")
    else:
        print("\n✓  No SOTIF boundary violations detected across all conditions.")

    # ── Acceptance thresholds used ────────────────────────────────────────────
    print(f"\n   Thresholds used  →  recall: LOW≥{THRESHOLDS['recall_medium']} / "
          f"MEDIUM≥{THRESHOLDS['recall_high']} / HIGH≥{THRESHOLDS['recall_critical']} / CRITICAL<{THRESHOLDS['recall_critical']}")
    print(f"                        mAP50: LOW≥{THRESHOLDS['map50_medium']} / "
          f"MEDIUM≥{THRESHOLDS['map50_high']} / HIGH≥{THRESHOLDS['map50_critical']} / CRITICAL<{THRESHOLDS['map50_critical']}")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    csv_path = os.path.join(RESULTS_DIR, "safety_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["Condition", "mAP50", "mAP50-95", "Precision", "Recall",
                         "Person_mAP50", "Car_mAP50", "Person_Recall", "Car_Recall", "SOTIF_Risk"])
        for name, kpis, risk in rows:
            writer.writerow([
                name, f"{kpis['map50']:.4f}", f"{kpis['map5095']:.4f}",
                f"{kpis['precision']:.4f}", f"{kpis['recall']:.4f}",
                f"{kpis['person_map50']:.4f}", f"{kpis['car_map50']:.4f}",
                f"{kpis['person_recall']:.4f}", f"{kpis['car_recall']:.4f}",
                risk,
            ])

    print(f"\n   CSV saved to: {csv_path}")
    print(f"   Plots saved to: {RESULTS_DIR}/")
