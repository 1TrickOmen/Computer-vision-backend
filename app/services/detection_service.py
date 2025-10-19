"""
Detection service using YOLOv8 for object detection
Supports both local YOLO models and AWS SageMaker endpoints
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import time
import os
import base64
import json
from pathlib import Path

from ultralytics import YOLO
import boto3
from app.core.config import settings
from app.core.database import SessionLocal
from app.models.detection import Detection
from app.models.snapshot import Snapshot

logger = logging.getLogger(__name__)


class DetectionService:
    """Service for object detection using YOLOv8 (local) or AWS SageMaker"""
    
    def __init__(self):
        self.model: Optional[YOLO] = None
        self.model_name = settings.detector_choice
        self.confidence_threshold = settings.confidence_threshold
        self.iou_threshold = settings.iou_threshold
        
        # AWS SageMaker configuration
        self.use_sagemaker = getattr(settings, 'use_sagemaker', False)
        self.sagemaker_endpoint = getattr(settings, 'aws_sagemaker_endpoint', None)
        self.aws_region = getattr(settings, 'aws_region', 'us-east-1')
        self.sagemaker_runtime = None
        
    async def initialize(self):
        """Initialize the detection service (local YOLO or SageMaker)"""
        if self.use_sagemaker and self.sagemaker_endpoint:
            await self._initialize_sagemaker()
        else:
            await self._initialize_local_yolo()
    
    async def _initialize_sagemaker(self):
        """Initialize SageMaker endpoint connection"""
        try:
            # Initialize SageMaker runtime client
            if not self.sagemaker_runtime:
                logger.info(f"🔗 Initializing SageMaker runtime client for region: {self.aws_region}")
                self.sagemaker_runtime = boto3.client('sagemaker-runtime', region_name=self.aws_region)
            
            # Test the endpoint
            logger.info(f"🔗 Testing SageMaker endpoint: {self.sagemaker_endpoint}")
            
            # Create a test image
            test_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', test_image)
            image_bytes = buffer.tobytes()
            
            # Test endpoint with timeout
            import asyncio
            try:
                response = await asyncio.wait_for(
                    asyncio.to_thread(
                        self.sagemaker_runtime.invoke_endpoint,
                        EndpointName=self.sagemaker_endpoint,
                        ContentType='image/jpeg',
                        Body=image_bytes
                    ),
                    timeout=30.0  # 30 second timeout
                )
                
                result = json.loads(response['Body'].read())
                logger.info(f"✅ SageMaker endpoint test completed: {len(result.get('detections', []))} detections")
                
            except asyncio.TimeoutError:
                logger.warning("⚠️ SageMaker endpoint test timed out, but endpoint may still be usable")
                logger.info("✅ SageMaker endpoint initialized (skipped test due to timeout)")
            
        except Exception as e:
            logger.error(f"Failed to initialize SageMaker endpoint: {e}")
            raise
    
    async def _initialize_local_yolo(self):
        """Initialize local YOLO model"""
        try:
            # Patch ultralytics to use weights_only=False for PyTorch 2.6+ compatibility
            import ultralytics.nn.tasks as tasks
            
            # Store original function
            original_torch_safe_load = tasks.torch_safe_load
            
            def patched_torch_safe_load(file):
                import torch
                return torch.load(file, map_location='cpu', weights_only=False), file
            
            # Apply patch
            tasks.torch_safe_load = patched_torch_safe_load
            
            # Load YOLO model
            self.model = YOLO(f"{self.model_name}.pt")
            logger.info(f"🤖 YOLO model '{self.model_name}' loaded successfully")
            
            # Restore original function
            tasks.torch_safe_load = original_torch_safe_load
            
            # Test the model with a dummy image
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False)
            logger.info("✅ YOLO model test completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
            raise
    
    def detect_objects(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """
        Detect objects in the given image using local YOLO or SageMaker
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (detections_list, processing_time_ms)
        """
        if self.use_sagemaker and self.sagemaker_endpoint:
            return self._detect_objects_sagemaker(image)
        else:
            return self._detect_objects_local(image)
    
    def _detect_objects_sagemaker(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """Detect objects using SageMaker endpoint"""
        if not self.sagemaker_runtime:
            raise Exception("SageMaker runtime client not initialized")
        
        start_time = time.time()
        
        try:
            # Optimize image for faster processing
            height, width = image.shape[:2]
            
            # Resize to smaller dimensions for faster processing if image is too large
            if width > 640 or height > 480:
                scale = min(640/width, 480/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image_resized = cv2.resize(image, (new_width, new_height))
            else:
                image_resized = image
                new_width, new_height = width, height
            
            # Encode image as JPEG with optimized quality for faster encoding
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]  # Lower quality for faster processing
            _, buffer = cv2.imencode('.jpg', image_resized, encode_params)
            image_bytes = buffer.tobytes()
            
            # Call SageMaker endpoint with timeout handling
            try:
                import threading
                import queue
                
                # Use threading for timeout instead of asyncio
                result_queue = queue.Queue()
                exception_queue = queue.Queue()
                
                def invoke_endpoint():
                    try:
                        response = self.sagemaker_runtime.invoke_endpoint(
                            EndpointName=self.sagemaker_endpoint,
                            ContentType='image/jpeg',
                            Body=image_bytes
                        )
                        result_queue.put(response)
                    except Exception as e:
                        exception_queue.put(e)
                
                # Start the thread
                thread = threading.Thread(target=invoke_endpoint)
                thread.daemon = True
                thread.start()
                
                # Wait for result with timeout
                thread.join(timeout=3.0)  # 3 second timeout
                
                if thread.is_alive():
                    logger.warning("SageMaker endpoint timeout for live feed")
                    processing_time = (time.time() - start_time) * 1000
                    return [], processing_time
                
                # Check for exceptions
                if not exception_queue.empty():
                    raise exception_queue.get()
                
                # Get the result
                if result_queue.empty():
                    logger.warning("SageMaker endpoint returned no result")
                    processing_time = (time.time() - start_time) * 1000
                    return [], processing_time
                
                response = result_queue.get()
                
                # Parse response
                result = json.loads(response['Body'].read())
                logger.info(f"SageMaker raw response: {result}")
                
            except Exception as e:
                logger.warning(f"SageMaker endpoint error: {e}")
                processing_time = (time.time() - start_time) * 1000
                return [], processing_time
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Convert SageMaker response to our format
            detections = []
            
            # Check different possible response formats
            sage_detections = result.get('detections', [])
            if not sage_detections:
                # Try alternative response format
                sage_detections = result.get('predictions', [])
            if not sage_detections:
                # Try direct array format
                if isinstance(result, list):
                    sage_detections = result
            
            logger.info(f"SageMaker returned {len(sage_detections)} detections")
            
            for detection in sage_detections:
                logger.info(f"Processing detection: {detection}")
                
                # Handle different bbox formats
                bbox = detection.get('bbox', [])
                if not bbox:
                    bbox = detection.get('box', [])
                if not bbox:
                    bbox = detection.get('coordinates', [])
                # Ultralytics-style output from our inference.py uses 'xyxy'
                if not bbox:
                    bbox = detection.get('xyxy', [])
                
                if len(bbox) != 4:
                    logger.warning(f"Invalid bbox format: {bbox}")
                    continue
                
                # Scale bounding boxes back to original image size if resized
                if width != new_width or height != new_height:
                    scale_x = width / new_width
                    scale_y = height / new_height
                    bbox = [
                        bbox[0] * scale_x,
                        bbox[1] * scale_y,
                        bbox[2] * scale_x,
                        bbox[3] * scale_y
                    ]
                
                detections.append({
                    "id": len(detections),
                    # Support multiple field names from different payloads
                    "class_id": int(detection.get('class_id', detection.get('class', detection.get('cls', 0)))) if isinstance(detection.get('class_id', detection.get('class', detection.get('cls', 0))), (int, float)) else 0,
                    "class_name": detection.get('class_name', detection.get('label', 'unknown')),
                    "confidence": float(detection.get('confidence', detection.get('score', detection.get('conf', 0.0)))),
                    "bbox": {
                        "x1": bbox[0],
                        "y1": bbox[1],
                        "x2": bbox[2],
                        "y2": bbox[3]
                    },
                    "center": {
                        "x": (bbox[0] + bbox[2]) / 2,
                        "y": (bbox[1] + bbox[3]) / 2
                    },
                    "width": bbox[2] - bbox[0],
                    "height": bbox[3] - bbox[1]
                })
            
            logger.debug(f"SageMaker detected {len(detections)} objects in {processing_time:.2f}ms")
            return detections, processing_time
            
        except Exception as e:
            logger.error(f"Error during SageMaker detection: {e}")
            processing_time = (time.time() - start_time) * 1000
            return [], processing_time
    
    def _detect_objects_local(self, image: np.ndarray) -> Tuple[List[Dict[str, Any]], float]:
        """Detect objects using local YOLO model"""
        if self.model is None:
            raise Exception("YOLO model not initialized")
        
        start_time = time.time()
        
        try:
            # Run detection
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            # Parse results
            detections = []
            if results and len(results) > 0:
                result = results[0]  # Get first (and only) result
                
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
                    class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
                    class_names = result.names  # Class names mapping
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        detection = {
                            "id": i,
                            "class_id": int(class_id),
                            "class_name": class_names[int(class_id)],
                            "confidence": float(conf),
                            "bbox": {
                                "x1": float(box[0]),
                                "y1": float(box[1]),
                                "x2": float(box[2]),
                                "y2": float(box[3])
                            },
                            "center": {
                                "x": float((box[0] + box[2]) / 2),
                                "y": float((box[1] + box[3]) / 2)
                            },
                            "width": float(box[2] - box[0]),
                            "height": float(box[3] - box[1])
                        }
                        detections.append(detection)
            
            logger.debug(f"Local YOLO detected {len(detections)} objects in {processing_time:.2f}ms")
            return detections, processing_time
            
        except Exception as e:
            logger.error(f"Error during local detection: {e}")
            processing_time = (time.time() - start_time) * 1000
            return [], processing_time
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draw bounding boxes and labels on the image
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            bbox = detection["bbox"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
            # Draw bounding box
            cv2.rectangle(
                result_image,
                (int(bbox["x1"]), int(bbox["y1"])),
                (int(bbox["x2"]), int(bbox["y2"])),
                (0, 255, 0),  # Green color
                2
            )
            
            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            
            cv2.rectangle(
                result_image,
                (int(bbox["x1"]), int(bbox["y1"]) - label_size[1] - 10),
                (int(bbox["x1"]) + label_size[0], int(bbox["y1"])),
                (0, 255, 0),
                -1
            )
            
            # Draw label text
            cv2.putText(
                result_image,
                label,
                (int(bbox["x1"]), int(bbox["y1"]) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),  # Black text
                2
            )
        
        return result_image
    
    async def save_detection_to_db(
        self,
        detections: List[Dict[str, Any]],
        processing_time_ms: float,
        image_path: Optional[str] = None,
        camera_settings: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Save detection results to database
        
        Args:
            detections: List of detection dictionaries
            processing_time_ms: Processing time in milliseconds
            image_path: Path to the processed image
            camera_settings: Camera settings at time of detection
            
        Returns:
            Detection ID
        """
        db = SessionLocal()
        try:
            detection_record = Detection(
                image_path=image_path,
                model_name=self.model_name,
                confidence_threshold=self.confidence_threshold,
                objects_detected=len(detections),
                detection_data=detections,
                camera_settings=camera_settings,
                processing_time_ms=processing_time_ms
            )
            
            db.add(detection_record)
            db.commit()
            db.refresh(detection_record)
            
            logger.info(f"💾 Saved detection record with ID {detection_record.id}")
            return detection_record.id
            
        except Exception as e:
            logger.error(f"Failed to save detection to database: {e}")
            db.rollback()
            raise
        finally:
            db.close()
    
    async def save_snapshot(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        detection_id: Optional[int] = None
    ) -> Optional[str]:
        """
        Save snapshot with bounding boxes
        
        Args:
            image: Image to save
            detections: Detection results
            detection_id: Associated detection ID
            
        Returns:
            Path to saved snapshot or None if failed
        """
        if not settings.enable_snapshots:
            return None
        
        try:
            # Create snapshot directory if it doesn't exist
            snapshot_dir = Path(settings.snapshot_folder)
            snapshot_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
            filename = f"snapshot_{timestamp}.jpg"
            file_path = snapshot_dir / filename
            
            # Draw detections on image
            annotated_image = self.draw_detections(image, detections)
            
            # Save image
            success = cv2.imwrite(str(file_path), annotated_image)
            
            if success:
                # Save snapshot record to database
                db = SessionLocal()
                try:
                    snapshot_record = Snapshot(
                        filename=filename,
                        file_path=str(file_path),
                        file_size_bytes=os.path.getsize(file_path),
                        detection_id=detection_id,
                        image_width=image.shape[1],
                        image_height=image.shape[0],
                        format="jpg",
                        has_bounding_boxes="true" if detections else "false",
                        bounding_box_count=len(detections)
                    )
                    
                    db.add(snapshot_record)
                    db.commit()
                    
                    logger.info(f"📸 Saved snapshot: {filename}")
                    return str(file_path)
                    
                except Exception as e:
                    logger.error(f"Failed to save snapshot record: {e}")
                    db.rollback()
                    return None
                finally:
                    db.close()
            else:
                logger.error(f"Failed to save image to {file_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if self.use_sagemaker and self.sagemaker_endpoint:
            return {
                "model_name": self.model_name,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "status": "sagemaker_endpoint",
                "endpoint": self.sagemaker_endpoint,
                "aws_region": self.aws_region,
                "class_names": []  # SageMaker doesn't provide class names in info
            }
        elif self.model is None:
            return {"status": "not_loaded"}
        else:
            return {
                "model_name": self.model_name,
                "confidence_threshold": self.confidence_threshold,
                "iou_threshold": self.iou_threshold,
                "status": "local_loaded",
                "class_names": list(self.model.names.values()) if hasattr(self.model, 'names') else []
            }
