from ultralytics import YOLO
from PIL import Image
import io
import cv2
import numpy as np

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

# This is the BCRS color coding
ELIGIBLE_COLOR = (0, 255, 0)    # This is green
NOT_ELIGIBLE_COLOR = (0, 0, 255)    # This is red

def predict_with_visualization(image_bytes: bytes, filename: str) -> tuple:
    # This converts bytes to numpy array for OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # This runs YOLO inference
    results = model(image_cv, conf=0.5, iou=0.5)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].tolist()
            bcrs_eligible = class_name in ELIGIBLE_CLASSES

            # This draws the bound box
            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            color = ELIGIBLE_COLOR if bcrs_eligible else NOT_ELIGIBLE_COLOR
            cv2.rectangle(image_cv, (x1, y1), (x2, y2), color, 3)

            # This draws the label
            label = f"{class_name} {confidence:.2f}"
            eligible_text = "ELIGIBLE" if bcrs_eligible else "NOT ELIGIBLE"
            cv2.putText(image_cv, label, (x1, y1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(image_cv, eligible_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            detections.append({
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox": bbox,
                "bcrs_eligible": bcrs_eligible
            })

    eligible_count = sum(1 for d in detections if d["bcrs_eligible"])

    # This converts back to bytes
    _, buffer = cv2.imencode('.jpg', image_cv)
    image_bytes_out = buffer.tobytes()

    result_data = {
        "filename": filename,
        "detections": detections,
        "total_detections": len(detections),
        "eligible_count": eligible_count,
        "not_eligible_count": len(detections) - eligible_count
    }

    return result_data, image_bytes_out