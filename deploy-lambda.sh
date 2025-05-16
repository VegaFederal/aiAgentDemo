#!/bin/bash

# Script to package and deploy Lambda functions

# Configuration
LAMBDA_CODE_BUCKET="aiad-lambda-code"
REGION="us-east-1"
ENVIRONMENT="dev"

# Create deployment bucket if it doesn't exist
echo "Creating Lambda code bucket if it doesn't exist..."
aws s3api head-bucket --bucket $LAMBDA_CODE_BUCKET 2>/dev/null || aws s3 mb s3://$LAMBDA_CODE_BUCKET --region $REGION

# Package and upload document processor Lambda
echo "Packaging document processor Lambda..."
cd src/document-processor
pip install -r requirements.txt -t .
zip -r document-processor.zip .
aws s3 cp document-processor.zip s3://$LAMBDA_CODE_BUCKET/lambda/document-processor.zip
rm document-processor.zip
cd ../..

# Package and upload LLM agent Lambda
echo "Packaging LLM agent Lambda..."
cd src/llm-agent
pip install -r requirements.txt -t .
zip -r llm-agent.zip .
aws s3 cp llm-agent.zip s3://$LAMBDA_CODE_BUCKET/lambda/llm-agent.zip
rm llm-agent.zip
cd ../..

echo "Lambda functions packaged and uploaded successfully!"
echo "Now you can deploy the CloudFormation stack with:"
echo "aws cloudformation deploy --template-file infrastructure/cloudformation/main-template.yaml --stack-name aiad-$ENVIRONMENT --parameter-overrides Environment=$ENVIRONMENT LambdaCodeBucket=$LAMBDA_CODE_BUCKET --capabilities CAPABILITY_NAMED_IAM"