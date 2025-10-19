"""
Camera service for handling camera input and settings control
"""

import cv2
import asyncio
import logging
from typing import Optional, Dict, Any, Tuple
import numpy as np
from datetime import datetime

from app.core.config import settings

logger = logging.getLogger(__name__)


class CameraService:
    """Service for camera operations and settings control"""
    
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.current_frame: Optional[np.ndarray] = None
        self.camera_settings: Dict[str, Any] = {}
        
    async def start(self):
        """Start camera service"""
        try:
            # Check if camera is disabled (for Docker testing)
            if settings.camera_source == -1:
                logger.info("🎥 Camera service disabled (Docker mode)")
                self.is_running = True
                return
            
            self.cap = cv2.VideoCapture(settings.camera_source)
            
            if not self.cap.isOpened():
                raise Exception(f"Could not open camera source {settings.camera_source}")
            
            # Set initial camera properties
            await self._configure_camera()
            
            # Get initial camera settings
            self.camera_settings = await self._get_camera_settings()
            
            self.is_running = True
            logger.info("🎥 Camera service started successfully")
            
            # Start frame capture loop
            asyncio.create_task(self._capture_loop())
            
        except Exception as e:
            logger.error(f"Failed to start camera service: {e}")
            raise
    
    async def stop(self):
        """Stop camera service"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        logger.info("🛑 Camera service stopped")
    
    async def _configure_camera(self):
        """Configure camera settings"""
        if not self.cap:
            return
        
        try:
            # Set frame size
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)
            
            # Set FPS
            self.cap.set(cv2.CAP_PROP_FPS, settings.fps)
            
            # Try to set exposure (may not work on all cameras)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
            self.cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Lower exposure value
            
            # Try to set focus (may not work on all cameras)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # Disable autofocus
            self.cap.set(cv2.CAP_PROP_FOCUS, 50)  # Set focus value
            
            # Set brightness (0-64 range, use 32 as middle)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 32)
            
            # Set contrast (0-64 range, use 32 as middle)
            self.cap.set(cv2.CAP_PROP_CONTRAST, 32)
            
            logger.info("📷 Camera settings configured")
            
        except Exception as e:
            logger.warning(f"Some camera settings could not be applied: {e}")
    
    async def _get_camera_settings(self) -> Dict[str, Any]:
        """Get current camera settings"""
        if not self.cap:
            return {}
        
        settings_dict = {}
        
        try:
            # Get actual frame dimensions
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Get current property values
            exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
            brightness = self.cap.get(cv2.CAP_PROP_BRIGHTNESS)
            contrast = self.cap.get(cv2.CAP_PROP_CONTRAST)
            focus = self.cap.get(cv2.CAP_PROP_FOCUS)
            
            # Get property ranges for proper slider scaling
            exposure_min = self.cap.get(cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_MIN)
            exposure_max = self.cap.get(cv2.CAP_PROP_EXPOSURE, cv2.CAP_PROP_MAX)
            brightness_min = self.cap.get(cv2.CAP_PROP_BRIGHTNESS, cv2.CAP_PROP_MIN)
            brightness_max = self.cap.get(cv2.CAP_PROP_BRIGHTNESS, cv2.CAP_PROP_MAX)
            contrast_min = self.cap.get(cv2.CAP_PROP_CONTRAST, cv2.CAP_PROP_MIN)
            contrast_max = self.cap.get(cv2.CAP_PROP_CONTRAST, cv2.CAP_PROP_MAX)
            focus_min = self.cap.get(cv2.CAP_PROP_FOCUS, cv2.CAP_PROP_MIN)
            focus_max = self.cap.get(cv2.CAP_PROP_FOCUS, cv2.CAP_PROP_MAX)
            
            settings_dict.update({
                "frame_width": actual_width,
                "frame_height": actual_height,
                "fps": actual_fps,
                "exposure": exposure,
                "brightness": brightness,
                "contrast": contrast,
                "focus": focus,
                "auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
                "autofocus": self.cap.get(cv2.CAP_PROP_AUTOFOCUS),
                "timestamp": datetime.utcnow().isoformat(),
                # Property ranges for frontend sliders
                "exposure_range": {"min": exposure_min, "max": exposure_max},
                "brightness_range": {"min": brightness_min, "max": brightness_max},
                "contrast_range": {"min": contrast_min, "max": contrast_max},
                "focus_range": {"min": focus_min, "max": focus_max}
            })
            
            logger.info(f"📊 Camera settings retrieved: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            logger.info(f"📊 Property ranges - Exposure: {exposure_min:.2f} to {exposure_max:.2f}, Brightness: {brightness_min:.2f} to {brightness_max:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to get camera settings: {e}")
        
        return settings_dict
    
    async def _capture_loop(self):
        """Continuous frame capture loop"""
        while self.is_running:
            try:
                if self.cap and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret:
                        self.current_frame = frame.copy()
                    else:
                        logger.warning("Failed to read frame from camera")
                else:
                    logger.error("Camera not available")
                    break
                
                # Small delay to prevent excessive CPU usage
                await asyncio.sleep(1.0 / settings.fps)
                
            except Exception as e:
                logger.error(f"Error in capture loop: {e}")
                await asyncio.sleep(0.1)
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get the current frame"""
        # Return None if camera is disabled
        if settings.camera_source == -1:
            return None
        return self.current_frame.copy() if self.current_frame is not None else None
    
    def get_camera_settings(self) -> Dict[str, Any]:
        """Get current camera settings"""
        return self.camera_settings.copy()
    
    async def update_camera_setting(self, property_name: str, value: float) -> bool:
        """Update a camera setting"""
        if not self.cap:
            return False
        
        try:
            # Map property names to OpenCV constants
            property_map = {
                "exposure": cv2.CAP_PROP_EXPOSURE,
                "brightness": cv2.CAP_PROP_BRIGHTNESS,
                "contrast": cv2.CAP_PROP_CONTRAST,
                "focus": cv2.CAP_PROP_FOCUS,
                "frame_width": cv2.CAP_PROP_FRAME_WIDTH,
                "frame_height": cv2.CAP_PROP_FRAME_HEIGHT,
                "fps": cv2.CAP_PROP_FPS
            }
            
            if property_name in property_map:
                success = self.cap.set(property_map[property_name], value)
                if success:
                    # Update our settings cache
                    self.camera_settings[property_name] = value
                    self.camera_settings["timestamp"] = datetime.utcnow().isoformat()
                    logger.info(f"📷 Updated {property_name} to {value}")
                    return True
                else:
                    logger.warning(f"Failed to set {property_name} to {value}")
                    return False
            else:
                logger.error(f"Unknown camera property: {property_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating camera setting {property_name}: {e}")
            return False
