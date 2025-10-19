"""
API routes for the Computer Vision Backend Service
"""

import cv2
import numpy as np
import logging
from typing import List, Optional
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
import io
import json

from app.core.database import get_db
from app.models.detection import Detection
from app.models.snapshot import Snapshot
from app.services.camera_service import CameraService
from app.services.detection_service import DetectionService
from app.core.config import settings
import os

logger = logging.getLogger(__name__)

router = APIRouter()

# Global services (will be injected)
camera_service: Optional[CameraService] = None
detection_service: Optional[DetectionService] = None


def set_services(cam_service: CameraService, det_service: DetectionService):
    """Set global services"""
    global camera_service, detection_service
    camera_service = cam_service
    detection_service = det_service


@router.get("/video_feed")
async def video_feed():
    """Live video feed with object detection"""
    if not camera_service or not detection_service:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    def generate_frames():
        """Generate video frames with detections"""
        while True:
            try:
                frame = camera_service.get_current_frame()
                if frame is not None:
                    # Run object detection
                    detections, _ = detection_service.detect_objects(frame)
                    
                    # Draw detections
                    annotated_frame = detection_service.draw_detections(frame, detections)
                    
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', annotated_frame)
                    frame_bytes = buffer.tobytes()
                    
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # Send a placeholder frame if no camera input
                    placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(placeholder, "No Camera Input", (200, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', placeholder)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
            except Exception as e:
                logger.error(f"Error generating video frame: {e}")
                break
    
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    save_snapshot: bool = Form(True),
    db: Session = Depends(get_db)
):
    """Upload and process image/video file"""
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    
    # Validate file type
    file_extension = file.filename.split('.')[-1].lower() if file.filename else ""
    if file_extension not in settings.allowed_file_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{file_extension}' not allowed. Allowed types: {settings.allowed_file_types}"
        )
    
    # Validate file size
    content = await file.read()
    file_size_mb = len(content) / (1024 * 1024)
    if file_size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File size {file_size_mb:.2f}MB exceeds maximum {settings.max_file_size_mb}MB"
        )
    
    try:
        # Decode image
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")
        
        # Run object detection
        detections, processing_time = detection_service.detect_objects(image)
        
        # Get camera settings
        camera_settings = camera_service.get_camera_settings() if camera_service else None
        
        # Save detection to database
        detection_id = await detection_service.save_detection_to_db(
            detections, processing_time, file.filename, camera_settings
        )
        
        # Save snapshot if requested
        snapshot_path = None
        if save_snapshot and settings.enable_snapshots:
            snapshot_path = await detection_service.save_snapshot(
                image, detections, detection_id
            )
        
        return {
            "message": "File processed successfully",
            "detection_id": detection_id,
            "objects_detected": len(detections),
            "processing_time_ms": processing_time,
            "detections": detections,
            "snapshot_path": snapshot_path
        }
        
    except Exception as e:
        logger.error(f"Error processing uploaded file: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@router.get("/detections")
async def get_detections(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get detection history"""
    try:
        detections = db.query(Detection).order_by(Detection.timestamp.desc()).offset(offset).limit(limit).all()
        
        return {
            "detections": [
                {
                    "id": d.id,
                    "timestamp": d.timestamp.isoformat(),
                    "image_path": d.image_path,
                    "model_name": d.model_name,
                    "objects_detected": d.objects_detected,
                    "processing_time_ms": d.processing_time_ms,
                    "detection_data": d.detection_data,
                    "camera_settings": d.camera_settings
                }
                for d in detections
            ],
            "total": len(detections),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error retrieving detections: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving detections")


@router.get("/detections/{detection_id}")
async def get_detection(detection_id: int, db: Session = Depends(get_db)):
    """Get specific detection by ID"""
    try:
        detection = db.query(Detection).filter(Detection.id == detection_id).first()
        
        if not detection:
            raise HTTPException(status_code=404, detail="Detection not found")
        
        return {
            "id": detection.id,
            "timestamp": detection.timestamp.isoformat(),
            "image_path": detection.image_path,
            "model_name": detection.model_name,
            "confidence_threshold": detection.confidence_threshold,
            "objects_detected": detection.objects_detected,
            "detection_data": detection.detection_data,
            "camera_settings": detection.camera_settings,
            "processing_time_ms": detection.processing_time_ms,
            "frame_width": detection.frame_width,
            "frame_height": detection.frame_height
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving detection {detection_id}: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving detection")


@router.get("/snapshots")
async def get_snapshots(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get snapshot list"""
    try:
        snapshots = db.query(Snapshot).order_by(Snapshot.timestamp.desc()).offset(offset).limit(limit).all()
        
        return {
            "snapshots": [
                {
                    "id": s.id,
                    "timestamp": s.timestamp.isoformat(),
                    "filename": s.filename,
                    "file_path": s.file_path,
                    "file_size_bytes": s.file_size_bytes,
                    "detection_id": s.detection_id,
                    "image_width": s.image_width,
                    "image_height": s.image_height,
                    "format": s.format,
                    "has_bounding_boxes": s.has_bounding_boxes,
                    "bounding_box_count": s.bounding_box_count
                }
                for s in snapshots
            ],
            "total": len(snapshots),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error retrieving snapshots: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving snapshots")


@router.post("/snapshots")
async def capture_snapshot():
    """Capture a snapshot from the current camera feed"""
    if not camera_service or not detection_service:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    try:
        # Get current frame from camera
        frame = camera_service.get_current_frame()
        if frame is None:
            raise HTTPException(status_code=400, detail="No camera feed available")
        
        # Run object detection on the frame
        detections, processing_time = detection_service.detect_objects(frame)
        
        # Get camera settings
        camera_settings = camera_service.get_camera_settings()
        
        # Save detection to database
        detection_id = await detection_service.save_detection_to_db(
            detections, processing_time, "live_capture", camera_settings
        )
        
        # Save snapshot with bounding boxes
        snapshot_path = await detection_service.save_snapshot(
            frame, detections, detection_id
        )
        
        if snapshot_path:
            return {
                "message": "Snapshot captured successfully",
                "snapshot_path": snapshot_path,
                "detection_id": detection_id,
                "objects_detected": len(detections),
                "processing_time_ms": processing_time
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to save snapshot")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error capturing snapshot: {e}")
        raise HTTPException(status_code=500, detail=f"Error capturing snapshot: {str(e)}")


@router.get("/camera/settings")
async def get_camera_settings():
    """Get current camera settings"""
    if not camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")
    
    return camera_service.get_camera_settings()


@router.post("/camera/settings")
async def update_camera_setting(
    property_name: str = Form(...),
    value: float = Form(...)
):
    """Update camera setting"""
    if not camera_service:
        raise HTTPException(status_code=503, detail="Camera service not initialized")
    
    success = await camera_service.update_camera_setting(property_name, value)
    
    if success:
        return {"message": f"Updated {property_name} to {value}", "success": True}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to update {property_name}")


@router.get("/model/info")
async def get_model_info():
    """Get model information"""
    if not detection_service:
        raise HTTPException(status_code=503, detail="Detection service not initialized")
    
    return detection_service.get_model_info()


@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get system statistics"""
    try:
        total_detections = db.query(Detection).count()
        total_snapshots = db.query(Snapshot).count()
        
        # Get recent activity (last 24 hours)
        from datetime import datetime, timedelta
        yesterday = datetime.utcnow() - timedelta(days=1)
        
        recent_detections = db.query(Detection).filter(Detection.timestamp >= yesterday).count()
        recent_snapshots = db.query(Snapshot).filter(Snapshot.timestamp >= yesterday).count()
        
        return {
            "total_detections": total_detections,
            "total_snapshots": total_snapshots,
            "recent_detections_24h": recent_detections,
            "recent_snapshots_24h": recent_snapshots,
            "camera_status": "active" if camera_service and camera_service.is_running else "inactive",
            "detection_service_status": "active" if detection_service else "inactive",
            "model_info": detection_service.get_model_info() if detection_service else None
        }
        
    except Exception as e:
        logger.error(f"Error retrieving stats: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving stats")


@router.post("/mode/switch")
async def switch_detection_mode(use_sagemaker: str = Form(...)):
    """Switch between local YOLO and SageMaker detection modes"""
    try:
        # Update the .env file
        env_file = ".env"
        env_content = []
        
        # Read existing .env file
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                env_content = f.readlines()
        
        # Update or add USE_SAGEMAKER setting
        updated = False
        for i, line in enumerate(env_content):
            if line.startswith('USE_SAGEMAKER='):
                env_content[i] = f'USE_SAGEMAKER={use_sagemaker.lower()}\n'
                updated = True
                break
        
        if not updated:
            env_content.append(f'USE_SAGEMAKER={use_sagemaker.lower()}\n')
        
        # Write back to .env file
        with open(env_file, 'w') as f:
            f.writelines(env_content)
        
        # Update the settings object
        settings.use_sagemaker = use_sagemaker.lower() == 'true'
        
        # Update detection service configuration
        if detection_service:
            detection_service.use_sagemaker = settings.use_sagemaker
            # Clear the existing model to force reinitialization
            detection_service.model = None
            logger.info("Detection service configuration updated, model cleared for reinitialization")
        
        mode_name = "Cloud SageMaker" if settings.use_sagemaker else "Local YOLO"
        logger.info(f"Switched to {mode_name} mode")
        
        return {
            "success": True,
            "message": f"Successfully switched to {mode_name} mode. Please refresh the page to ensure proper initialization.",
            "current_mode": mode_name,
            "use_sagemaker": settings.use_sagemaker
        }
        
    except Exception as e:
        logger.error(f"Error switching detection mode: {e}")
        return {
            "success": False,
            "message": f"Failed to switch mode: {str(e)}"
        }


@router.post("/restart")
async def restart_services():
    """Restart detection and camera services"""
    try:
        if detection_service:
            # Reinitialize detection service
            await detection_service.initialize()
            logger.info("Detection service restarted successfully")
        
        if camera_service:
            # Restart camera service if needed
            if not camera_service.is_running:
                await camera_service.start()
                logger.info("Camera service restarted successfully")
        
        return {
            "success": True,
            "message": "Services restarted successfully"
        }
        
    except Exception as e:
        logger.error(f"Error restarting services: {e}")
        return {
            "success": False,
            "message": f"Failed to restart services: {str(e)}"
        }
