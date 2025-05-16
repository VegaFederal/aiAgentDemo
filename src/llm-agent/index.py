import json
import os
import boto3
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
bedrock = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')

# Get environment variables
EMBEDDINGS_TABLE = os.environ.get('EMBEDDINGS_TABLE')
EMBEDDING_MODEL_ID = os.environ.get('EMBEDDING_MODEL_ID')
LLM_MODEL_ID = os.environ.get('LLM_MODEL_ID')
SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', '0.7'))
MAX_CONTEXT_DOCS = int(os.environ.get('MAX_CONTEXT_DOCS', '5'))

# Log configuration
logger.info(f"Configuration: EMBEDDINGS_TABLE={EMBEDDINGS_TABLE}, EMBEDDING_MODEL_ID={EMBEDDING_MODEL_ID}, LLM_MODEL_ID={LLM_MODEL_ID}")

# Get table reference
embeddings_table = dynamodb.Table(EMBEDDINGS_TABLE)

def generate_embedding(text):
    """Generate embeddings for the given text using Amazon Titan"""
    logger.info(f"Generating embedding for text of length {len(text)}")
    response = bedrock.invoke_model(
        modelId=EMBEDDING_MODEL_ID,
        body=json.dumps({
            "inputText": text
        })
    )
    response_body = json.loads(response['body'].read())
    logger.info(f"Generated embedding of dimension {len(response_body['embedding'])}")
    return response_body['embedding']

def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve_relevant_context(query_embedding):
    """Retrieve relevant documents based on embedding similarity"""
    logger.info("Retrieving relevant context from DynamoDB")
    # In a real implementation, you would use a vector database
    # This is a simplified version that scans the DynamoDB table
    response = embeddings_table.scan()
    items = response.get('Items', [])
    logger.info(f"Found {len(items)} items in DynamoDB")
    
    # Calculate similarity for each document
    similarities = []
    for item in items:
        doc_embedding = item['embedding']
        similarity = cosine_similarity(query_embedding, doc_embedding)
        if similarity >= SIMILARITY_THRESHOLD:
            similarities.append({
                'document_id': item['document_id'],
                'content': item['content'],
                'similarity': similarity
            })
    
    # Sort by similarity (highest first) and take top N
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    selected_docs = similarities[:MAX_CONTEXT_DOCS]
    logger.info(f"Selected {len(selected_docs)} relevant documents with similarities: {[round(doc['similarity'], 3) for doc in selected_docs]}")
    return selected_docs

def call_claude(query, context_docs, question_type=None):
    """Call Claude 3.5 Sonnet with the query and context"""
    # Prepare context from retrieved documents
    context_text = "\n\n".join([doc['content'] for doc in context_docs])
    logger.info(f"Calling Claude with context length {len(context_text)} and question type {question_type}")
    
    # Prepare the prompt for Claude
    system_prompt = """You are a helpful AI assistant answering questions based on the provided context.
    Only use information from the context to answer the question. If the context doesn't contain the answer, say you don't know."""
    
    # Add instructions based on question type
    if question_type == 'multiple_choice':
        system_prompt += "\nThis is a multiple choice question. Provide the letter of the correct answer."
    elif question_type == 'yes_no':
        system_prompt += "\nThis is a yes/no question. Answer with 'Yes' or 'No' followed by an explanation."
    elif question_type == 'true_false':
        system_prompt += "\nThis is a true/false question. Answer with 'True' or 'False' followed by an explanation."
    
    # Create the message for Claude
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    ]
    
    # Call Claude with inference profile
    try:
        logger.info(f"Invoking Bedrock model: {LLM_MODEL_ID}")
        logger.info(f"Sending {len(messages)} messages to Claude")
        logger.info(f"Sending {{messages}} to Claude")
        response = bedrock.invoke_model(
            modelId=LLM_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": messages
            })
        )
        
        response_body = json.loads(response['body'].read())
        response_text = response_body['content'][0]['text']
        logger.info(f"Received response of length {len(response_text)}")
        return response_text
    except Exception as e:
        logger.error(f"Error calling Claude: {str(e)}")
        raise

def lambda_handler(event, context):
    """Handle API Gateway requests for the LLM agent"""
    logger.info(f"Received event: {json.dumps(event)}")
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))
        query = body.get('query')
        question_type = body.get('question_type')  # multiple_choice, yes_no, true_false, or null
        
        logger.info(f"Processing query: '{query}' with question_type: {question_type}")
        
        if not query:
            logger.warning("Missing query parameter")
            return {
                'statusCode': 400,
                'headers': {'Content-Type': 'application/json'},
                'body': json.dumps({'error': 'Missing query parameter'})
            }
        
        # Generate embedding for the query
        query_embedding = generate_embedding(query)
        
        # Retrieve relevant context
        context_docs = retrieve_relevant_context(query_embedding)
        
        # Call Claude with the query and context
        response = call_claude(query, context_docs, question_type)
        
        logger.info("Successfully processed request")
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'response': response,
                'context_used': len(context_docs)
            })
        }
    
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }
