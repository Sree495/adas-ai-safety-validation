from ultralytics import YOLO

from config import MODEL_PATH, NOMINAL_YAML as DATA_YAML, RESULTS_VAL

# Confidence thresholds to sweep
CONF_THRESHOLDS = [0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]


def run_val(model, conf):
    return model.val(
        data=DATA_YAML,
        split="val",
        device="cpu",
        workers=0,
        conf=conf,
        plots=False,   # skip plots during sweep; only save for best
        verbose=False,
    )


if __name__ == "__main__":
    model = YOLO(MODEL_PATH)

    print(f"\n{'Conf':>6}  {'P':>6}  {'R':>6}  {'F1':>6}  {'mAP50':>7}  {'mAP50-95':>9}")
    print("-" * 52)

    best_f1, best_conf, best_metrics = 0, 0, None

    for conf in CONF_THRESHOLDS:
        m = run_val(model, conf)
        p  = m.box.mp
        r  = m.box.mr
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        map50 = m.box.map50
        map   = m.box.map
        print(f"{conf:>6.2f}  {p:>6.3f}  {r:>6.3f}  {f1:>6.3f}  {map50:>7.3f}  {map:>9.3f}")

        if f1 > best_f1:
            best_f1, best_conf, best_metrics = f1, conf, m

    print("-" * 52)
    print(f"\nBest confidence threshold by F1: {best_conf}")

    # Re-run best conf with plots saved
    print(f"\nSaving plots for conf={best_conf} ...")
    final = model.val(
        data=DATA_YAML,
        split="val",
        device="cpu",
        workers=0,
        conf=best_conf,
        plots=True,
        project=RESULTS_VAL,
        name=f"val_conf{int(best_conf*100)}",
    )

    print(f"\n=== Best Results (conf={best_conf}) ===")
    print(f"mAP50:      {final.box.map50:.4f}")
    print(f"mAP50-95:   {final.box.map:.4f}")
    print(f"Precision:  {final.box.mp:.4f}")
    print(f"Recall:     {final.box.mr:.4f}")
    print(f"\nPer-class results:")
    names = {0: "person", 1: "car"}
    for i, (p, r, ap50, ap) in enumerate(zip(
        final.box.p, final.box.r, final.box.ap50, final.box.ap
    )):
        print(f"  {names.get(i, i):10s}  P={p:.3f}  R={r:.3f}  mAP50={ap50:.3f}  mAP50-95={ap:.3f}")

    print(f"\nPlots saved to: results/val_conf{int(best_conf*100)}/")
