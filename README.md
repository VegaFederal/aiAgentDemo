# AI Agent Demo

A serverless web application that demonstrates using AI agents with LLMs. The application uses Amazon Bedrock for Titan Embeddings and Claude 3.5 Sonnet V2 to provide context-aware responses based on document embeddings.

## Architecture

This application uses the following AWS services:

- **Amazon Bedrock**: For accessing Claude 3.5 Sonnet V2 and Amazon Titan Embeddings models
- **AWS Lambda**: For serverless backend logic
- **Amazon API Gateway**: To create RESTful endpoints for the web application
- **Amazon S3**: To host static web assets and store documents
- **Amazon DynamoDB**: To store document embeddings
- **Amazon CloudFront**: For content delivery
- **AWS IAM**: For managing permissions between services

## Components

### Backend

1. **Document Processor Lambda**:
   - Processes documents uploaded to S3
   - Extracts text from documents (currently supports TXT and HTML)
   - Chunks text into manageable segments
   - Generates embeddings using Amazon Titan Embeddings
   - Stores embeddings in DynamoDB

2. **LLM Agent Lambda**:
   - Handles user queries
   - Generates embeddings for user queries
   - Retrieves relevant context from DynamoDB
   - Calls Claude 3.5 Sonnet V2 with context and query
   - Supports multiple question types (regular, multiple choice, yes/no, true/false)

### Frontend

A simple web interface that:
- Provides a chat-like interface
- Allows users to select question types
- Sends queries to the API Gateway endpoint
- Displays responses from the LLM agent

## Deployment

The application is deployed using AWS CloudFormation with nested stacks:

1. **Main Template**: Orchestrates the deployment of all resources
2. **Security Groups**: Defines security groups for the application
3. **Static Website**: Sets up S3 and CloudFront for hosting the web interface
4. **Bedrock Lambda**: Configures Lambda functions, DynamoDB, and API Gateway

## Getting Started

### Prerequisites

- AWS Account with access to Amazon Bedrock
- Access to Claude 3.5 Sonnet V2 and Amazon Titan Embeddings models
- AWS CLI configured with appropriate permissions

### Deployment Steps

1. Clone this repository
2. Deploy the CloudFormation stack:

```bash
aws cloudformation deploy \
  --template-file infrastructure/cloudformation/main-template.yaml \
  --stack-name ai-agent-demo \
  --parameter-overrides Environment=dev \
  --capabilities CAPABILITY_IAM
```

3. Upload your documents to the created S3 bucket
4. Update the API endpoint in the web interface
5. Access the application through the CloudFront URL

## Document Format Support

The application currently supports:

- **HTML**: Best option for preserving text structure
- **TXT**: Simple plain text documents

## Question Types

The application supports different question types:

- **Regular Questions**: General queries about the document content
- **Multiple Choice**: Questions with specific options
- **Yes/No Questions**: Questions requiring a yes or no answer
- **True/False Questions**: Questions requiring a true or false determination