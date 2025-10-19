import numpy as np
import torch
import os
import json
import io
import cv2
import time
from ultralytics import YOLO

def model_fn(model_dir):
    print("Executing model_fn from inference.py ...")
    try:
        model = YOLO("/opt/ml/model/yolov8n.pt")
        print("Model loaded successfully")
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise e

def input_fn(request_body, request_content_type):
    print("Executing input_fn from inference.py ...")
    print(f"Content type: {request_content_type}")
    print(f"Request body length: {len(request_body)}")
    
    if request_content_type == 'image/jpeg':
        jpg_as_np = np.frombuffer(request_body, dtype=np.uint8)
        img = cv2.imdecode(jpg_as_np, flags=cv2.IMREAD_COLOR)
        print(f"Decoded image shape: {img.shape}")
        return img
        
    elif request_content_type == 'application/json':
        # Handle JSON input with base64 image
        data = json.loads(request_body)
        if 'image' in data:
            import base64
            image_bytes = base64.b64decode(data['image'])
            jpg_as_np = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(jpg_as_np, flags=cv2.IMREAD_COLOR)
            print(f"Decoded JSON image shape: {img.shape}")
            return img
    else:
        raise Exception(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model):
    print("Executing predict_fn from inference.py ...")
    print(f"Input data type: {type(input_data)}, shape: {input_data.shape}")
    
    # Ensure image is in correct format
    if len(input_data.shape) == 3:
        # Convert BGR to RGB if needed
        if input_data.shape[2] == 3:
            input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model.to(device)
    with torch.no_grad():
        print("Running YOLO inference...")
        results = model(input_data)
        print(f"Inference completed, got {len(results)} results")
    
    return results

def output_fn(prediction_output, content_type):
    print("Executing output_fn from inference.py ...")
    
    # Convert results to JSON-serializable format
    output = {
        "detections": [],
        "count": 0
    }
    
    for result in prediction_output:
        if result.boxes:
            boxes_data = []
            for box in result.boxes:
                box_info = {
                    "xyxy": box.xyxy.cpu().numpy().tolist()[0] if hasattr(box.xyxy, 'cpu') else box.xyxy.tolist()[0],
                    "conf": box.conf.cpu().item() if hasattr(box.conf, 'cpu') else box.conf.item(),
                    "cls": box.cls.cpu().item() if hasattr(box.cls, 'cpu') else box.cls.item(),
                    "class_name": result.names[int(box.cls.cpu().item() if hasattr(box.cls, 'cpu') else box.cls.item())]
                }
                boxes_data.append(box_info)
            
            output["detections"] = boxes_data
            output["count"] = len(boxes_data)
            print(f"Found {len(boxes_data)} detections")
    
    return json.dumps(output)