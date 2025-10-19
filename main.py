"""
Main application entry point for the Computer Vision Backend Service
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import uvicorn
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

from app.core.config import settings
from app.core.database import init_db
from app.api.routes import router
from app.services.camera_service import CameraService
from app.services.detection_service import DetectionService

# Load environment variables
load_dotenv()

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    global camera_service, detection_service
    
    # Initialize database
    await init_db()
    
    # Initialize services
    camera_service = CameraService()
    detection_service = DetectionService()
    
    # Initialize detection service
    await detection_service.initialize()
    
    # Start camera service (only if camera is enabled)
    if settings.camera_source != -1:
        await camera_service.start()
    else:
        logger.info("🎥 Camera service disabled (Docker mode)")
        camera_service.is_running = True
    
    # Set services in router
    from app.api.routes import set_services
    set_services(camera_service, detection_service)
    
    print("🚀 Computer Vision Backend Service started successfully!")
    
    yield
    
    # Shutdown
    if camera_service:
        await camera_service.stop()
    
    print("🛑 Computer Vision Backend Service stopped")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Computer Vision Backend Service",
    description="Real-time object detection with camera input and live streaming",
    version="1.0.0",
    lifespan=lifespan
)

# Include API routes
app.include_router(router, prefix="/api/v1")

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Templates
templates = Jinja2Templates(directory="app/templates")

# Global services
camera_service = None
detection_service = None

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the main web interface"""
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Computer Vision Backend Service"
    })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Computer Vision Backend Service",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
