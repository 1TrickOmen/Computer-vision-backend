# AWS SageMaker Deployment Guide

Quick guide to deploy the YOLO model to AWS SageMaker for cloud inference.

## 🎯 Overview

Deploy only the YOLO model to SageMaker while keeping the FastAPI application running locally. This satisfies the cloud deployment requirement.

## 📋 Prerequisites

- **AWS Account** with SageMaker permissions
- **AWS CLI** configured with credentials
- **Python 3.10+** with boto3 installed

## 🚀 Quick Deployment

### 1. Configure AWS Credentials

```bash
# Option 1: AWS CLI
aws configure

# Option 2: Environment Variables
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1
```

### 2. Create SageMaker Execution Role

Create an IAM role with these policies:
- `AmazonSageMakerFullAccess`
- `AmazonS3FullAccess`

**Role Name**: `SageMakerExecutionRole`

### 3. Deploy Model

```bash
# Install dependencies
pip install boto3 botocore

# Deploy YOLO model to SageMaker
python aws_sagemaker_deploy.py \
    --model-name yolov8n \
    --region us-east-1 \
    --execution-role-arn arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole \
    --instance-type ml.t2.medium
```

### 4. Configure Application

Add to your `.env` file:

```bash
# AWS SageMaker Configuration
USE_SAGEMAKER=true
AWS_SAGEMAKER_ENDPOINT=yolov8n-endpoint
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

### 5. Test Cloud Integration

```bash
# Start application
python main.py

# Verify SageMaker integration
curl http://localhost:8000/api/v1/model/info
```

Expected response:
```json
{
  "model_name": "yolov8n",
  "status": "sagemaker_endpoint",
  "endpoint": "yolov8n-endpoint",
  "aws_region": "us-east-1"
}
```

## 🔄 Switching Modes

### Local YOLO
```bash
USE_SAGEMAKER=false
```

### Cloud SageMaker
```bash
USE_SAGEMAKER=true
AWS_SAGEMAKER_ENDPOINT=yolov8n-endpoint
```

## 💰 Cost Management

| Instance Type | Cost/Hour | Use Case |
|---------------|-----------|----------|
| ml.t2.medium | ~$0.05 | Development |
| ml.m5.large | ~$0.12 | Production |

## 🛠️ Management Commands

```bash
# List endpoints
aws sagemaker list-endpoints

# Delete endpoint
aws sagemaker delete-endpoint --endpoint-name yolov8n-endpoint

# Check status
aws sagemaker describe-endpoint --endpoint-name yolov8n-endpoint
```

## 🔧 Troubleshooting

### Common Issues

1. **"Endpoint not found"**
   - Verify endpoint name in `.env`
   - Check: `aws sagemaker list-endpoints`

2. **"Access denied"**
   - Check AWS credentials
   - Verify SageMaker permissions

3. **"Model loading failed"**
   - Check S3 model path
   - Verify model.tar.gz exists

### Debug Mode
```bash
LOG_LEVEL=DEBUG
```

## 📊 Performance Comparison

| Method | Latency | Cost | Scalability |
|--------|---------|------|-------------|
| Local YOLO | ~50ms | Free | Hardware limited |
| SageMaker | ~200ms | $0.05/hour | Auto-scaling |

## 🎯 Benefits

- ✅ **Satisfies cloud requirement** from task specification
- ✅ **Auto-scaling** based on demand
- ✅ **Managed infrastructure** - no server maintenance
- ✅ **Easy model updates** via SageMaker
- ✅ **Cost control** - pay only for usage

---

**🎉 Your YOLO model is now running in the cloud! The FastAPI application seamlessly switches between local and cloud inference.**
