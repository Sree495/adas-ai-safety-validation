# src/preprocess_all.py
import os
import glob
import json

# -----------------------------
# Class mapping for YOLO
# -----------------------------
CLASS_MAP = {
    "person": 0,
    "car": 1
    # Add more if needed
}

# -----------------------------
# Convert single JSON to YOLO
# -----------------------------
def convert_annotation_json_to_yolo(json_file, save_dir):
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return 0

    img_width = data.get('size', {}).get('width')
    img_height = data.get('size', {}).get('height')

    if not img_width or not img_height:
        print(f"Missing image size info in {json_file}")
        return 0

    labels = []

    for obj in data.get('objects', []):
        class_title = obj.get('classTitle', '').lower()

        if class_title not in CLASS_MAP:
            continue

        class_id = CLASS_MAP[class_title]
        exterior = obj.get('points', {}).get('exterior', [])

        if len(exterior) != 2:
            continue

        try:
            x1, y1 = exterior[0]
            x2, y2 = exterior[1]

            x_center = (x1 + x2) / 2 / img_width
            y_center = (y1 + y2) / 2 / img_height
            w = abs(x2 - x1) / img_width
            h = abs(y2 - y1) / img_height

            labels.append(f"{class_id} {x_center} {y_center} {w} {h}")

        except Exception:
            continue

    # Always create labels folder
    os.makedirs(save_dir, exist_ok=True)

    img_name = os.path.splitext(os.path.basename(json_file))[0]  # strip .json -> "name.jpg"
    base_name = os.path.splitext(img_name)[0] + ".txt"           # strip .jpg  -> "name.txt"
    label_path = os.path.join(save_dir, base_name)

    # Create file even if empty (important for YOLO)
    with open(label_path, 'w') as f:
        if labels:
            f.write("\n".join(labels))

    return len(labels)


# -----------------------------
# Preprocess entire split folder
# -----------------------------
def preprocess_folder(image_folder):
    json_files = glob.glob(
        os.path.join(image_folder, "**", "*.json"),
        recursive=True
    ) + glob.glob(
        os.path.join(image_folder, "**", "*.JSON"),
        recursive=True
    )

    print(f"\nProcessing folder: {image_folder}")
    print(f"Found {len(json_files)} JSON files")

    if not json_files:
        print("No JSON files found, skipping...")
        return

    labels_folder = os.path.join(image_folder, "labels")

    total_objects = 0

    for json_file in json_files:
        count = convert_annotation_json_to_yolo(json_file, labels_folder)
        total_objects += count

    print(f"Created {len(json_files)} label files in: {labels_folder}")
    print(f"Total detected objects written: {total_objects}")


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":

    from config import DATA_DIR
    base_path = str(DATA_DIR)

    for split in ["train", "val", "test"]:
        folder_path = os.path.join(base_path, split)

        if os.path.exists(folder_path):
            preprocess_folder(folder_path)
        else:
            print(f"Folder not found: {folder_path}")

    print("\nAll preprocessing done!")