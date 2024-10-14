# app.py

import io
import sys
import os
from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel
import torch
import numpy as np


# Fixing this new problem I have never seen before??
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


YOLOV5_MODELS_PATH = os.path.join(os.getcwd(), 'yolov5_models')
if YOLOV5_MODELS_PATH not in sys.path:
    sys.path.insert(0, YOLOV5_MODELS_PATH)

app = FastAPI(
    title="Vehicle Damage Detection Assignment API",
    description="API for detecting vehicle damages using a YOLOv5"
)

MODEL_PATH = "app/models/best-m.pt"

# Check if the model file exists
if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(f"Model file not found at {MODEL_PATH}. Please check the path and try again.")
else:
    print(f"Model file found at {MODEL_PATH}")

# YOLO class names
class_names = [
    'minor-dent',
    'minor-scratch',
    'moderate-broken',
    'moderate-dent',
    'moderate-scratch',
    'severe-broken',
    'severe-dent',
    'severe-scratch'
]

# Load the YOLOv5 model using torch.hub with force_reload=True to clear cache
try:
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)
    model.eval()
   
except Exception as e:
    raise RuntimeError(f"Failed to load the YOLOv5 model from {MODEL_PATH}. Error: {e}")


class Detection(BaseModel):
    """
    Detection model representing the details of an object detected in an image.

    Attributes:
        class_id (int): The ID of the detected object's class.
        class_name (str): The name of the detected object's class.
        confidence (float): The confidence score of the detection.
        bbox (List[float]): The bounding box coordinates of the detected object in the format [x_min, y_min, x_max, y_max].
    """
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x_min, y_min, x_max, y_max]

class InferenceResponse(BaseModel):
    """
    InferenceResponse is a model that represents the response of an inference operation.

    Attributes:
        detections (List[Detection]): A list of Detection objects representing the detected items.
    """
    detections: List[Detection]

def run_yolov5_inference(image: Image.Image, confidence_threshold: float = 0.15) -> List[Detection]:
    """
    Runs a YOLOv5 model for object detection on the given image.

    Args:
        image (PIL.Image.Image): The input image for object detection.
        confidence_threshold (float): Minimum confidence to consider a detection valid.
        iou_threshold (float): IoU threshold for Non-Maximum Suppression.

    Returns:
        List[Detection]: A list of Detection objects representing the detected bounding boxes.
    """
    # Perform inference
    results = model(image)

    # Extract detections
    detections = []
    for *box, conf, cls in results.xyxy[0]:  # xyxy (bounding box), confidence, class
        if conf < confidence_threshold:
            continue

        class_id = int(cls.item())
        confidence = float(conf.item())
        x_min, y_min, x_max, y_max = [float(coord.item()) for coord in box]

        # Map the class ID to class name
        class_name = class_names[class_id] if class_id < len(class_names) else "Unknown"

        detection = Detection(
            class_id=class_id,
            class_name=class_name,
            confidence=confidence,
            bbox=[x_min, y_min, x_max, y_max]
        )
        detections.append(detection)

    # Draw bounding boxes on the original image
    draw = ImageDraw.Draw(image)
    try:
        # Attempt to use a truetype font
        font = ImageFont.truetype("arial.ttf", size=16)
    except IOError:
        # Fallback to default font if truetype font is not found
        font = ImageFont.load_default()

    for detection in detections:
        x_min, y_min, x_max, y_max = detection.bbox
        class_name = detection.class_name
        confidence = detection.confidence
        label = f"{class_name}: {confidence:.2f}"

        # Draw bounding box
        draw.rectangle([(x_min, y_min), (x_max, y_max)], outline="red", width=2)

        # Calculate text size
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]

        # Draw rectangle for label background
        draw.rectangle([(x_min, y_min - text_height - 4), (x_min + text_width + 4, y_min)], fill="red")

        # Draw label text
        draw.text((x_min + 2, y_min - text_height - 2), label, fill="white", font=font)



    # Optionally, display the image with bounding boxes
    image.show()

    return detections

@app.post("/predict", response_model=InferenceResponse)
async def predict(file: UploadFile = File(...)):
    """
    Endpoint to perform vehicle damagedetection on an uploaded image
    """
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        # Read image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error processing the image") from e

    # Perform inference using the YOLOv5 model
    try:
        detections = run_yolov5_inference(image)
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error during model inference") from e

    return InferenceResponse(detections=detections)

@app.get("/")
def read_root():
    """
    Root endpoint with a welcome message.
    """
    return {"message": "Vehicle Damage Detection API for Lensor Assignment; use the /predict endpoint to upload images"}
