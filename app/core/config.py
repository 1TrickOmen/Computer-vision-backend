"""
Core configuration settings for the Computer Vision Backend Service
"""

import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    # Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"  # Ignore unknown env vars like use_cloud_model, lambda_function_name
    )
    
    # Database Configuration
    database_url: str = Field(default="sqlite:///./detections.db", env="DATABASE_URL")
    
    # Camera/Stream Configuration
    camera_source: int = Field(default=0, env="CAMERA_SOURCE")
    frame_width: int = Field(default=640, env="FRAME_WIDTH")
    frame_height: int = Field(default=480, env="FRAME_HEIGHT")
    fps: int = Field(default=30, env="FPS")
    
    # Object Detection Configuration
    detector_choice: str = Field(default="yolov8n", env="DETECTOR_CHOICE")
    confidence_threshold: float = Field(default=0.5, env="CONFIDENCE_THRESHOLD")
    iou_threshold: float = Field(default=0.45, env="IOU_THRESHOLD")
    
    # Snapshot Configuration
    enable_snapshots: bool = Field(default=True, env="ENABLE_SNAPSHOTS")
    snapshot_folder: str = Field(default="./snapshots", env="SNAPSHOT_FOLDER")
    max_snapshots: int = Field(default=100, env="MAX_SNAPSHOTS")
    
    # Security Configuration
    allowed_file_types: List[str] = Field(default=["jpg", "jpeg", "png", "mp4", "avi", "mov"], env="ALLOWED_FILE_TYPES")
    max_file_size_mb: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    
    # AWS Configuration
    aws_region: str = Field(default="us-east-1", env="AWS_REGION")
    aws_access_key_id: str = Field(default="", env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str = Field(default="", env="AWS_SECRET_ACCESS_KEY")
    aws_s3_bucket: str = Field(default="", env="AWS_S3_BUCKET")
    
    # AWS SageMaker Configuration
    use_sagemaker: bool = Field(default=True, env="USE_SAGEMAKER")  # Default to cloud mode
    aws_sagemaker_endpoint: str = Field(default="yolov8n-endpoint", env="AWS_SAGEMAKER_ENDPOINT")
    
    # Note: Cloud SageMaker mode is the default for better scalability
    # To use local YOLO, set USE_SAGEMAKER=false in .env file
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: str = Field(default="./logs/app.log", env="LOG_FILE")
    
    # Application Configuration
    app_name: str = "Computer Vision Backend Service"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    
    # Note: legacy Config class is replaced by model_config above for pydantic v2


# Global settings instance
settings = Settings()

# Create necessary directories
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        settings.snapshot_folder,
        os.path.dirname(settings.log_file),
        "app/static/uploads"
    ]
    
    for directory in directories:
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

# Initialize directories on import
create_directories()

def ensure_cloud_default():
    """Ensure cloud mode is the default on startup"""
    try:
        env_file = ".env"
        env_content = []
        
        # Read existing .env file
        if os.path.exists(env_file):
            with open(env_file, 'r') as f:
                env_content = f.readlines()
        
        # Update or add USE_SAGEMAKER setting to default to cloud
        updated = False
        for i, line in enumerate(env_content):
            if line.startswith('USE_SAGEMAKER='):
                env_content[i] = 'USE_SAGEMAKER=true\n'
                updated = True
                break
        
        if not updated:
            env_content.append('USE_SAGEMAKER=true\n')
        
        # Also ensure AWS_SAGEMAKER_ENDPOINT is set
        endpoint_updated = False
        for i, line in enumerate(env_content):
            if line.startswith('AWS_SAGEMAKER_ENDPOINT='):
                env_content[i] = 'AWS_SAGEMAKER_ENDPOINT=yolov8n-endpoint\n'
                endpoint_updated = True
                break
        
        if not endpoint_updated:
            env_content.append('AWS_SAGEMAKER_ENDPOINT=yolov8n-endpoint\n')
        
        # Write back to .env file
        with open(env_file, 'w') as f:
            f.writelines(env_content)
            
        print("✅ Defaulted to cloud SageMaker mode on startup")
        
    except Exception as e:
        print(f"⚠️ Could not update .env file: {e}")

# Ensure cloud mode is default on startup
ensure_cloud_default()

