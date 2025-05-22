import json
import os
import boto3
import uuid
from index import process_zip_file, chunk_text, generate_embedding, store_embedding

# Initialize AWS clients
s3 = boto3.client('s3')
lambda_client = boto3.client('lambda')

def lambda_handler(event, context):
    """Process all documents in a bucket or specific prefix"""
    try:
        # Get bucket and optional prefix from event
        bucket = event.get('bucket')
        prefix = event.get('prefix', '')
        
        if not bucket:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'message': 'Bucket name is required'
                })
            }
        
        # Check if we should process a ZIP file directly
        zip_key = event.get('zipKey')
        if zip_key:
            return process_single_zip(bucket, zip_key)
        
        # List all objects in the bucket with the given prefix
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        
        if 'Contents' not in response:
            return {
                'statusCode': 404,
                'body': json.dumps({
                    'message': f'No files found in bucket {bucket} with prefix {prefix}'
                })
            }
        
        # Find ZIP files
        zip_files = [obj['Key'] for obj in response['Contents'] 
                    if obj['Key'].lower().endswith('.zip')]
        
        if not zip_files:
            return {
                'statusCode': 404,
                'body': json.dumps({
                    'message': f'No ZIP files found in bucket {bucket} with prefix {prefix}'
                })
            }
        
        # Process the first ZIP file (or we could process all of them)
        return process_single_zip(bucket, zip_files[0])
        
    except Exception as e:
        print(f"Error processing documents: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Error processing documents: {str(e)}'
            })
        }

def process_single_zip(bucket, key):
    """Process a single ZIP file containing all documents"""
    try:
        print(f"Processing ZIP file: {key} from bucket: {bucket}")
        
        # Get the ZIP file
        file_obj = s3.get_object(Bucket=bucket, Key=key)
        content = file_obj['Body'].read()
        
        # Generate a document ID
        document_id = str(uuid.uuid4())
        
        # Extract metadata if provided
        metadata = {}
        s3_metadata = file_obj.get('Metadata', {})
        metadata = {
            'filename': key.split('/')[-1],
            'source_bucket': bucket,
            'source_key': key,
            **s3_metadata
        }
        
        # Process the ZIP file to extract text with hierarchical structure
        text = process_zip_file(content, key)
        
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
                'message': f'Successfully processed ZIP file {key}',
                'document_id': document_id,
                'chunks_processed': len(chunks)
            })
        }
    
    except Exception as e:
        print(f"Error processing ZIP file: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Error processing ZIP file: {str(e)}'
            })
        }