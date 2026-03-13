"""
Central path configuration — all scripts import from here.
Paths are derived from the project root so the repo works on any machine.

Project structure assumed:
    <project_root>/
        src/        ← this file lives here
        data/
            bdd-sample/
            augmented/
        results/
        runs/
"""

from pathlib import Path

# ── Project root (one level up from src/) ────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent

# ── Dataset ───────────────────────────────────────────────────────────────────
DATA_DIR    = BASE_DIR / "data" / "bdd-sample"
VAL_IMAGES  = str(DATA_DIR / "val"   / "images")
VAL_LABELS  = str(DATA_DIR / "val"   / "labels")
ANN_DIR     = str(DATA_DIR / "val"   / "ann")
TRAIN_IMAGES = str(DATA_DIR / "train" / "images")

# ── Augmented (synthetic) data ────────────────────────────────────────────────
AUG_BASE    = str(BASE_DIR / "data" / "augmented")

# ── Model weights ─────────────────────────────────────────────────────────────
MODEL_PATH  = str(BASE_DIR / "src" / "runs" / "detect" / "models"
                  / "yolo_bdd_v2" / "weights" / "best.pt")

# ── YOLO dataset config ───────────────────────────────────────────────────────
NOMINAL_YAML = str(BASE_DIR / "src" / "bdd.yaml")

# ── Results (one subfolder per stage) ────────────────────────────────────────
RESULTS_DIR               = str(BASE_DIR / "results")
RESULTS_VAL               = str(BASE_DIR / "results" / "val_run")
RESULTS_KPI               = str(BASE_DIR / "results" / "test_kpi")
RESULTS_SAFETY_REPORT     = str(BASE_DIR / "results" / "safety_report")
RESULTS_SCENARIO          = str(BASE_DIR / "results" / "scenario_analysis")
RESULTS_OOD               = str(BASE_DIR / "results" / "ood_detector")
RESULTS_UNCERTAINTY       = str(BASE_DIR / "results" / "uncertainty_eval")
RESULTS_EXPLAINABILITY    = str(BASE_DIR / "results" / "explainability")
RESULTS_SAFETY_MONITOR    = str(BASE_DIR / "results" / "safety_monitor")

# ── Cross-module artefacts ────────────────────────────────────────────────────
OOD_PER_IMAGE_CSV   = str(BASE_DIR / "results" / "ood_detector"     / "ood_per_image.csv")
DANGER_CSV          = str(BASE_DIR / "results" / "uncertainty_eval" / "uncertainty_per_image.csv")
