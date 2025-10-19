"""
Snapshot model for storing saved image information
"""

from sqlalchemy import Column, Integer, String, DateTime, Text, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from app.core.database import Base


class Snapshot(Base):
    """Model for storing snapshot information"""
    
    __tablename__ = "snapshots"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    filename = Column(String(255), nullable=False, unique=True)
    file_path = Column(String(500), nullable=False)
    file_size_bytes = Column(Integer, nullable=True)
    
    # Associated detection
    detection_id = Column(Integer, ForeignKey("detections.id"), nullable=True)
    detection = relationship("Detection", backref="snapshots")
    
    # Image metadata
    image_width = Column(Integer, nullable=True)
    image_height = Column(Integer, nullable=True)
    format = Column(String(10), nullable=True)  # jpg, png, etc.
    
    # Processing info
    has_bounding_boxes = Column(String(10), nullable=False, default="true")
    bounding_box_count = Column(Integer, nullable=True)
    
    def __repr__(self):
        return f"<Snapshot(id={self.id}, filename={self.filename}, timestamp={self.timestamp})>"
