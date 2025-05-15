import json
import os
import boto3
import uuid
import re

# Initialize AWS clients
s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')

# Environment variables
EMBEDDINGS_TABLE = os.environ.get('EMBEDDINGS_TABLE')
EMBEDDING_MODEL_ID = os.environ.get('EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v1')
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '200'))

# Get table reference
embeddings_table = dynamodb.Table(EMBEDDINGS_TABLE)

def extract_text(bucket, key):
    """Extract text from a document in S3 based on file extension"""
    file_obj = s3.get_object(Bucket=bucket, Key=key)
    content = file_obj['Body'].read()
    
    # Determine file type from extension
    file_extension = key.split('.')[-1].lower()
    
    if file_extension == 'txt':
        return content.decode('utf-8')
    
    elif file_extension == 'html' or file_extension == 'htm':
        # Simple HTML text extraction
        text = content.decode('utf-8')
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of specified size"""
    chunks = []
    if len(text) <= chunk_size:
        chunks.append(text)
    else:
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Try to find a natural break point
            if end < len(text):
                for break_char in ['\n\n', '\n', '. ', '? ', '! ']:
                    break_pos = text.rfind(break_char, start, end)
                    if break_pos > start:
                        end = break_pos + len(break_char)
                        break
            
            chunks.append(text[start:end].strip())
            start = end - chunk_overlap
    
    return chunks

def generate_embedding(text):
    """Generate embeddings for the given text using Amazon Titan"""
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({
            "inputText": text
        })
    )
    response_body = json.loads(response['body'].read())
    return response_body['embedding']

def store_embedding(document_id, chunk_id, text_chunk, embedding, metadata):
    """Store embedding and metadata in DynamoDB"""
    item = {
        'id': f"{document_id}_{chunk_id}",
        'document_id': document_id,
        'chunk_id': chunk_id,
        'content': text_chunk,
        'embedding': embedding,
        'metadata': metadata
    }
    
    embeddings_table.put_item(Item=item)

def lambda_handler(event, context):
    """Process documents from S3 and generate embeddings"""
    try:
        # Get S3 bucket and key from event
        bucket = event['Records'][0]['s3']['bucket']['name']
        key = event['Records'][0]['s3']['object']['key']
        
        # Extract metadata if provided
        metadata = {}
        # Try to get some basic metadata from S3
        s3_metadata = s3.head_object(Bucket=bucket, Key=key).get('Metadata', {})
        metadata = {
            'filename': key.split('/')[-1],
            **s3_metadata
        }
        
        # Generate a document ID if not provided
        document_id = metadata.get('document_id', str(uuid.uuid4()))
        
        # Extract text from document
        text = extract_text(bucket, key)
        
        # Split text into chunks
        chunks = chunk_text(text)
        
        # Process each chunk
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = generate_embedding(chunk)
            
            # Store in DynamoDB
            store_embedding(document_id, i, chunk, embedding, metadata)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Successfully processed document {key}',
                'document_id': document_id,
                'chunks_processed': len(chunks)
            })
        }
    
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Error processing document: {str(e)}'
            })
        }