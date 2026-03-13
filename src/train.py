from ultralytics import YOLO

if __name__ == "__main__":

    model = YOLO("yolov8s.pt")  # small model: 4x more capacity than nano, still CPU-friendly

    model.train(
        data=str(__import__('pathlib').Path(__file__).resolve().parent / "bdd.yaml"),
        epochs=50,
        imgsz=640,      # standard size; much better for detecting small persons than 416
        batch=4,        # safe for 16 GB RAM at 640 imgsz
        device="cpu",
        workers=0,
        project="models",
        name="yolo_bdd_v2",
        cache=True,     # load images into RAM once; speeds up CPU training significantly
        amp=False,
        patience=30,    # stop early if val mAP doesn't improve for 30 epochs
    )