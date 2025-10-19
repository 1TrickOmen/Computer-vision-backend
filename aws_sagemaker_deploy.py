#!/usr/bin/env python3
"""
AWS SageMaker Deployment Script for YOLO Model
Deploys YOLO model to AWS SageMaker for cloud inference
"""

import argparse
import boto3
import os
import tarfile
import tempfile
import logging
from pathlib import Path
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model_package(model_name="yolov8n", temp_dir=None):
    """Create model package for SageMaker"""
    logger.info(f"Creating model package for {model_name}")
    
    if temp_dir is None:
        temp_dir = Path(tempfile.mkdtemp())
    
    # Create model.tar.gz using tar (following AWS blog structure)
    model_archive = temp_dir / "model.tar.gz"
    with tarfile.open(model_archive, 'w:gz') as tar:
        # Add model file to root
        tar.add(f"{model_name}.pt", arcname=f"{model_name}.pt")
        # Create code directory and add inference files
        tar.add("inference.py", arcname="code/inference.py")
        
        # Create requirements.txt for SageMaker
        requirements_content = """opencv-python==4.8.1.78
torchvision==0.15.2
seaborn==0.13.2
ultralytics==8.0.196
omegaconf==2.3.0
"""
        requirements_file = temp_dir / "requirements.txt"
        with open(requirements_file, "w") as f:
            f.write(requirements_content)
        tar.add(requirements_file, arcname="code/requirements.txt")
    
    logger.info(f"Model package created: {model_archive}")
    return model_archive

def upload_to_s3(model_archive, bucket_name, s3_key):
    """Upload model package to S3"""
    logger.info(f"Uploading model to s3://{bucket_name}/{s3_key}")
    
    s3_client = boto3.client('s3')
    try:
        s3_client.upload_file(str(model_archive), bucket_name, s3_key)
        logger.info("✅ Model uploaded to S3 successfully")
        return f"s3://{bucket_name}/{s3_key}"
    except ClientError as e:
        logger.error(f"Failed to upload to S3: {e}")
        raise

def create_sagemaker_model(model_name, model_data_uri, execution_role_arn, region):
    """Create SageMaker model"""
    logger.info(f"Creating SageMaker model: {model_name}")
    
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    container = {
        "Image": f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.0.1-gpu-py310-cu118-ubuntu20.04",
        "ModelDataUrl": model_data_uri,
        "Mode": "SingleModel",
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference.py",
            "SAGEMAKER_REGION": region,
            "MMS_MODEL_STORE": "/opt/ml/model",
            "SAGEMAKER_MODEL_SERVER_WORKERS": "1"
        }
    }
    
    try:
        response = sagemaker_client.create_model(
            ModelName=model_name,
            PrimaryContainer=container,
            ExecutionRoleArn=execution_role_arn
        )
        logger.info(f"✅ Model created: {response['ModelArn']}")
        return response['ModelArn']
    except ClientError as e:
        if "Cannot create already existing model" in str(e):
            logger.warning(f"Model {model_name} already exists. Skipping creation.")
            return f"arn:aws:sagemaker:{region}:{boto3.client('sts').get_caller_identity()['Account']}:model/{model_name}"
        raise e

def create_endpoint_config(model_name, region):
    """Create endpoint configuration"""
    logger.info(f"Creating endpoint configuration: {model_name}-config")
    
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    endpoint_config_name = f"{model_name}-config"
    
    try:
        response = sagemaker_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "AllTraffic",
                    "ModelName": model_name,
                    "InitialInstanceCount": 1,
                    "InstanceType": "ml.t2.medium",
                    "InitialVariantWeight": 1
                }
            ]
        )
        logger.info(f"✅ Endpoint configuration created: {response['EndpointConfigArn']}")
        return response['EndpointConfigArn']
    except ClientError as e:
        if "Cannot create already existing endpoint configuration" in str(e):
            logger.warning(f"Endpoint configuration {endpoint_config_name} already exists. Skipping creation.")
            return f"arn:aws:sagemaker:{region}:{boto3.client('sts').get_caller_identity()['Account']}:endpoint-config/{endpoint_config_name}"
        raise e

def create_endpoint(model_name, region):
    """Create SageMaker endpoint"""
    logger.info(f"Creating endpoint: {model_name}-endpoint")
    
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    endpoint_name = f"{model_name}-endpoint"
    endpoint_config_name = f"{model_name}-config"
    
    try:
        response = sagemaker_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
        logger.info(f"✅ Endpoint created: {response['EndpointArn']}")
        return response['EndpointArn']
    except ClientError as e:
        if "Cannot create already existing endpoint" in str(e):
            logger.warning(f"Endpoint {endpoint_name} already exists. Skipping creation.")
            return f"arn:aws:sagemaker:{region}:{boto3.client('sts').get_caller_identity()['Account']}:endpoint/{endpoint_name}"
        raise e

def wait_for_endpoint(endpoint_name, region):
    """Wait for endpoint to be ready"""
    logger.info(f"Waiting for endpoint {endpoint_name} to be ready...")
    
    sagemaker_client = boto3.client('sagemaker', region_name=region)
    
    try:
        waiter = sagemaker_client.get_waiter('endpoint_in_service')
        waiter.wait(EndpointName=endpoint_name)
        logger.info("✅ Endpoint is ready!")
    except Exception as e:
        logger.warning(f"Endpoint may not be ready yet: {e}")

def main():
    parser = argparse.ArgumentParser(description='Deploy YOLO model to AWS SageMaker')
    parser.add_argument('--model-name', default='yolov8n', help='YOLO model name')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--execution-role-arn', required=True, help='SageMaker execution role ARN')
    parser.add_argument('--instance-type', default='ml.t2.medium', help='SageMaker instance type')
    parser.add_argument('--bucket-name', help='S3 bucket name (optional)')
    
    args = parser.parse_args()
    
    try:
        # Get AWS account ID
        sts_client = boto3.client('sts')
        account_id = sts_client.get_caller_identity()['Account']
        
        # Set default bucket name
        if not args.bucket_name:
            args.bucket_name = f"yolo-sagemaker-{args.region}-{account_id}"
        
        logger.info("🚀 Starting SageMaker deployment...")
        
        # 1. Create model package
        model_archive = create_model_package(args.model_name)
        
        # 2. Upload to S3
        s3_key = f"models/{args.model_name}/model.tar.gz"
        model_data_uri = upload_to_s3(model_archive, args.bucket_name, s3_key)
        
        # 3. Create SageMaker model
        model_arn = create_sagemaker_model(args.model_name, model_data_uri, args.execution_role_arn, args.region)
        
        # 4. Create endpoint configuration
        endpoint_config_arn = create_endpoint_config(args.model_name, args.region)
        
        # 5. Create endpoint
        endpoint_arn = create_endpoint(args.model_name, args.region)
        
        # 6. Wait for endpoint to be ready
        endpoint_name = f"{args.model_name}-endpoint"
        wait_for_endpoint(endpoint_name, args.region)
        
        print("\n" + "="*50)
        print("DEPLOYMENT COMPLETED!")
        print("="*50)
        print(f"Endpoint Name: {endpoint_name}")
        print(f"Endpoint ARN: {endpoint_arn}")
        print(f"S3 Model Path: {model_data_uri}")
        print(f"\nAdd these to your .env file:")
        print(f"AWS_SAGEMAKER_ENDPOINT={endpoint_name}")
        print(f"AWS_REGION={args.region}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        raise

if __name__ == "__main__":
    main()

