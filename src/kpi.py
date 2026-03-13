"""
KPI report for the test split.
Run after: python preprocess.py && python fix_data.py
"""
from ultralytics import YOLO

from config import MODEL_PATH, NOMINAL_YAML as DATA_YAML, RESULTS_KPI as RESULTS_DIR
CONF = 0.30   # best threshold found during validation sweep

if __name__ == "__main__":
    model = YOLO(MODEL_PATH)

    print("Running inference on test split...")
    metrics = model.val(
        data=DATA_YAML,
        split="test",
        device="cpu",
        workers=0,
        conf=CONF,
        plots=True,
        project=RESULTS_DIR,
        name="test_kpi",
    )

    mp  = metrics.box.mp
    mr  = metrics.box.mr
    f1  = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0

    print("\n" + "=" * 50)
    print("        TEST SET KPI REPORT")
    print("=" * 50)
    print(f"  Confidence Threshold : {CONF}")
    print(f"  mAP50                : {metrics.box.map50:.4f}")
    print(f"  mAP50-95             : {metrics.box.map:.4f}")
    print(f"  Precision            : {mp:.4f}")
    print(f"  Recall               : {mr:.4f}")
    print(f"  F1 Score             : {f1:.4f}")
    print("-" * 50)
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>8} {'F1':>8} {'mAP50':>8}")
    print(f"  {'-'*48}")

    names = {0: "person", 1: "car"}
    for i, (p, r, ap50, ap) in enumerate(zip(
        metrics.box.p, metrics.box.r, metrics.box.ap50, metrics.box.ap
    )):
        f1c = 2 * p * r / (p + r) if (p + r) > 0 else 0
        print(f"  {names.get(i, str(i)):<12} {p:>10.3f} {r:>8.3f} {f1c:>8.3f} {ap50:>8.3f}")

    print("=" * 50)
    print(f"\nPlots saved to: {RESULTS_DIR}/test_kpi/")
    print("  - confusion_matrix_normalized.png")
    print("  - BoxPR_curve.png")
    print("  - BoxF1_curve.png")
