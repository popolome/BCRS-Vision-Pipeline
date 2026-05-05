from ultralytics import YOLO
from PIL import Image
import io

# This are the BCRS eligible classes based on Singapore's actual schem
ELIGIBLE_CLASSES = {"tin can", "bottle-plastic"}

# This loads the model once when app starts
model = YOLO("model/best.pt")

def predict(image_bytes: bytes, filename: str) -> dict:
    # This converts bytes to PIL Image
    image = Image.open(io.BytesIO(image_bytes))

    # This runs the YOLO inference
    results = model(image, conf=0.5, iou=0.5)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            bcrs_eligible = class_name in ELIGIBLE_CLASSES

            detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": bbox,
                "bcrs_eligible": bcrs_eligible
            })

    eligible_count = sum(1 for d in detections if d["bcrs_eligible"])

    return {
        "filename": filename,
        "detections": detections,
        "total_detections": len(detections),
        "eligible_count": eligible_count,
        "not_eligible_count": len(detections) - eligible_count
    }