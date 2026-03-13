# ADAS/AD Perception — AI V&V Pipeline
### From Model Training to Runtime Safety Decision

> *"Most ML engineers stop at accuracy. This project builds a safety case."*

A complete end-to-end AI Verification & Validation pipeline for ADAS object detection,
structured around **ISO 26262 (Functional Safety)** and **ISO 21448 (SOTIF)** principles.

Built on YOLOv8s + BDD (Berkeley Deep Drive) dataset. Runs fully on CPU.

---

## Why This Project Exists

Training a detection model is the easy part. The hard question in ADAS/AD is:

> **"How do you know the model is safe enough to deploy — and safe enough to trust at runtime?"**

That question does not have a single answer. It requires a *layered evidence chain*:
systematic testing under nominal conditions, stress testing under adverse conditions,
coverage analysis across real-world scenarios, and runtime signals that tell the system
when it is operating outside its comfort zone.

This project builds that chain, module by module, and converges on a **safety case** —
not just a benchmark number.

---

## Pipeline Architecture

```
Raw Data (BDD Supervisely format)
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 1 — DATA PREPARATION                                          │
│  preprocess.py   Supervisely JSON → YOLO label format                │
│  fix_data.py     Folder structure normalisation                       │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 2 — MODEL TRAINING                                            │
│  train.py        YOLOv8s fine-tuning on BDD (person + car)           │
│                  50 epochs · imgsz=640 · batch=4 · CPU               │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 3 — FUNCTIONAL V&V                                            │
│  validate.py     Confidence threshold sweep → best F1 @ conf=0.30    │
│  kpi.py          Per-class KPI report (mAP50, Recall, Precision, F1) │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 4 — SOTIF BOUNDARY TESTING  (ISO 21448)                       │
│  augment.py         Synthetic adverse conditions via Albumentations   │
│                     5 conditions: rain · night · fog · noise · glare  │
│  safety_report.py   Model KPIs on nominal + all 5 synthetic sets      │
│                     Risk classification: LOW / MEDIUM / HIGH /        │
│                     CRITICAL per ISO 21448 thresholds                 │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 5 — SCENARIO COVERAGE ANALYSIS                                │
│  scenario_analysis.py   Parse BDD metadata tags (weather · timeofday │
│                          · scene) from real val annotations           │
│                          Per-scenario mAP50 / Recall / coverage flag  │
│                          Identifies ODD gaps in real-world data       │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 6 — OUT-OF-DISTRIBUTION DETECTION                             │
│  ood_detector.py    YOLOv8s backbone features (512-dim, layer 9)     │
│                     PCA + Mahalanobis distance vs nominal baseline    │
│                     Per-image and per-scenario OOD scores             │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 7 — UNCERTAINTY QUANTIFICATION                                │
│  uncertainty_eval.py   Reliability diagram + ECE computation         │
│                         Temperature scaling calibration (T = 0.9936) │
│                         TTA uncertainty (5-pass std deviation)        │
│                         Danger-zone flag: high conf + high std        │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 8 — EXPLAINABILITY  (XAI)                                     │
│  explainability.py   EigenCAM  — backbone attention (no gradient)    │
│                      GradCAM   — per-class detection attention        │
│                      Outputs: day vs night comparison, danger zone,  │
│                      per-class (person vs car) attention maps         │
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────────────────────────────────────────┐
│  STAGE 9 — RUNTIME SAFETY MONITOR  (Integration Layer)               │
│  safety_monitor.py   Per-frame: OOD score + TTA uncertainty +        │
│                       calibrated confidence → GO / CAUTION / STOP    │
│                       Importable class for embedded integration       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

| # | Script | Input | Output | Standard |
|---|---|---|---|---|
| 1 | `preprocess.py` | Supervisely JSON annotations | YOLO `.txt` labels | — |
| 2 | `fix_data.py` | Raw dataset folder | Corrected folder/file names | — |
| 3 | `train.py` | BDD dataset | `best.pt` weights | — |
| 4 | `validate.py` | Val split | Confidence sweep, best F1 threshold | ISO 26262 §8 |
| 5 | `kpi.py` | Val split | mAP50, Recall, Precision, F1 per class | ISO 26262 §8 |
| 6 | `augment.py` | Real val images | 5 synthetic adverse-condition datasets | ISO 21448 |
| 7 | `safety_report.py` | Nominal + 5 synthetic sets | SOTIF risk table + CSV | ISO 21448 |
| 8 | `scenario_analysis.py` | Real val + BDD metadata | Per-scenario KPI + coverage gaps | ISO 21448 |
| 9 | `ood_detector.py` | Val images + backbone | OOD scores, PCA scatter plots | ISO 21448 |
| 10 | `uncertainty_eval.py` | Val images + GT | ECE, temperature T, TTA std, danger zone | ISO 26262 |
| 11 | `explainability.py` | Val images | GradCAM / EigenCAM heatmap grids | ISO 26262 |
| 12 | `safety_monitor.py` | Any camera frame | GO / CAUTION / STOP verdict | ISO 21448 |

---

## Key Results

### Functional Performance (Stage 3)
| Metric | Value |
|---|---|
| mAP50 | **0.644** |
| Recall | 0.573 |
| Precision | 0.748 |
| F1 | 0.649 |
| Best conf threshold | 0.30 |

### SOTIF Boundary Results — Synthetic Adverse Conditions (Stage 4)

Synthetic datasets generated by applying photometric transforms to real val images.
Labels unchanged (pixel-only transforms preserve bounding box geometry).

| Condition | mAP50 | Recall | SOTIF Risk |
|---|---|---|---|
| Nominal (baseline) | 0.644 | 0.573 | BASELINE |
| Rain | 0.412 | 0.287 | **HIGH** |
| **Night** | 0.089 | **0.084** | **CRITICAL** |
| Fog | 0.601 | 0.541 | LOW |
| **Noise** | 0.031 | **0.029** | **CRITICAL** |
| Glare | 0.598 | 0.538 | LOW |

> Synthetic night recall drops from 0.573 → 0.084 (-85%).
> These conditions are documented as known SOTIF boundaries and must be addressed in the ODD definition.

### Scenario Coverage Results — Real-World Data (Stage 5)

Real val images grouped by BDD metadata tags and evaluated independently.

| Scenario | N | mAP50 | Recall | Person Recall | Flag |
|---|---|---|---|---|---|
| timeofday=night | 45 | 0.341 | 0.354 | 0.226 | OK (borderline) |
| timeofday=daytime | 59 | 0.526 | 0.551 | 0.467 | OK |
| scene=highway | 21 | 0.217 | 0.234 | **0.000** | **PERF-GAP** |
| scene=residential | 11 | 0.805 | 0.830 | 1.000 | OK |
| weather=rainy | 6 | 0.245 | 0.287 | — | **PERF-GAP + LOW-COVERAGE** |

> Highway scene: person recall = 0.000. Zero pedestrian detections on highways.
> This is a hard ODD restriction — highway pedestrian scenarios must be excluded.

### The Safety Case — Four Methods, Same Answer

The most important result is not a single number.
It is the **convergence of four independent V&V methods** onto the same ODD boundary:

```
                        Night driving    Highway scene
                        ─────────────    ─────────────
SOTIF synthetic test    CRITICAL ✖       —
  (safety_report.py)    recall=0.084

Scenario analysis       recall=0.354     person R=0.000
  (scenario_analysis)   (borderline)     (hard failure)

OOD detection           13.3% flagged    19.0% flagged
  (ood_detector.py)     (highest tod)    (highest scene)

Uncertainty eval        All 5 danger     —
  (uncertainty_eval)    images = night

Runtime monitor         67% STOP/CAUTION  43% STOP/CAUTION
  (safety_monitor.py)   (30 STOP / 45)
```

> **ISO 21448 conclusion:** Night driving and highway scenes are at the boundary
> of the model's Operational Design Domain. Both must either be excluded from the ODD
> or mitigated with system-level safety measures (driver handover, speed restrictions).

---

## Explainability Evidence (Stage 8)

GradCAM analysis reveals *why* night performance degrades:

- **Daytime:** Backbone attention concentrates on vehicle boundaries and pedestrian silhouettes — expected, correct behaviour.
- **Night:** Attention is captured by **light emission sources** (headlights, streetlights) rather than object geometry. The model uses luminance as a proxy for "object present" when it cannot resolve edges.

This visual evidence directly supports the ODD restriction — it is not just a number claim, it is a mechanistic explanation grounded in what the model's internal representations actually encode.

---

## Runtime Safety Monitor (Stage 9)

The integration layer that makes all signals deployable:

```python
from safety_monitor import SafetyMonitor

monitor = SafetyMonitor()              # fits OOD baseline once at startup

img_bgr = cv2.imread("camera_frame.jpg")
verdict = monitor.assess(img_bgr)

# verdict["decision"]  →  "GO" | "CAUTION" | "STOP"
# verdict["reasons"]   →  ["OOD score 1.23 ≥ STOP threshold 1.10"]
# verdict["ood_score"] →  1.23
# verdict["tta_std"]   →  0.03
# verdict["cal_conf"]  →  0.87
```

Decision logic:
```
STOP    →  OOD score ≥ 1.10  OR  TTA uncertainty ≥ 0.20
CAUTION →  OOD score ≥ 0.95  OR  TTA uncertainty ≥ 0.10
              OR calibrated confidence < 0.45 (unreliable detections)
GO      →  all signals within safe bounds
```

Batch results on 109 val images:
- **GO: 57%**  — clear daytime scenes, nominal conditions
- **CAUTION: 13%**  — borderline OOD, moderate uncertainty
- **STOP: 30%**  — predominantly night + highway scenes

---

## What This Demonstrates

| Skill Area | Evidence in this project |
|---|---|
| **ML Engineering** | YOLOv8s training, transfer learning, confidence calibration |
| **ADAS Domain Knowledge** | BDD dataset, ODD definition, per-class detection requirements |
| **ISO 26262 Awareness** | KPI framework, ASIL justification via XAI, calibrated confidence |
| **ISO 21448 / SOTIF** | Synthetic adverse conditions, risk classification, ODD boundary documentation |
| **Validation Engineering** | Systematic V&V pipeline, scenario coverage, regression-ready structure |
| **Safety Architecture** | Multi-signal runtime monitor, auditable decision trail |
| **Explainability** | GradCAM + EigenCAM, mechanistic failure analysis |
| **Uncertainty Quantification** | Temperature scaling, TTA, ECE, danger zone detection |

---

## Setup and Usage

```bash
# 1. Activate virtual environment
venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline in order
python src/preprocess.py
python src/fix_data.py
python src/train.py

python src/validate.py
python src/kpi.py

python src/augment.py
python src/safety_report.py

python src/scenario_analysis.py
python src/ood_detector.py
python src/uncertainty_eval.py
python src/explainability.py
python src/safety_monitor.py
```

Results are saved to `results/` with one subfolder per stage.

---

## Stack

- **Model:** YOLOv8s (Ultralytics 8.1.10)
- **Dataset:** BDD-sample, Supervisely format — 2 classes: person, car
- **Augmentation:** Albumentations ≥1.3.0
- **Compute:** CPU-only (PyTorch 2.3.1+cpu) — no GPU required
- **Python:** 3.x, scipy, numpy, matplotlib, OpenCV

---

## Context

This project was built to demonstrate that AI-based perception for ADAS/AD is not
only an ML problem — it is a **systems engineering and safety problem**.

The gap between "model works on a benchmark" and "model is safe to deploy in a vehicle"
is exactly where ISO 26262 and SOTIF live. This pipeline addresses that gap
systematically, producing artefacts that map directly to what safety-critical
automotive programmes require: evidence, not just accuracy.
