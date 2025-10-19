"""
Detection model for storing object detection metadata
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.sql import func
from app.core.database import Base


class Detection(Base):
    """Model for storing detection metadata"""
    
    __tablename__ = "detections"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    image_path = Column(String(500), nullable=True)
    model_name = Column(String(100), nullable=False)
    confidence_threshold = Column(Float, nullable=False)
    
    # Detection results
    objects_detected = Column(Integer, nullable=False, default=0)
    detection_data = Column(JSON, nullable=True)  # Store full detection results
    
    # Camera settings at time of detection
    camera_settings = Column(JSON, nullable=True)
    
    # Processing metadata
    processing_time_ms = Column(Float, nullable=True)
    frame_width = Column(Integer, nullable=True)
    frame_height = Column(Integer, nullable=True)
    
    def __repr__(self):
        return f"<Detection(id={self.id}, timestamp={self.timestamp}, objects={self.objects_detected})>"
