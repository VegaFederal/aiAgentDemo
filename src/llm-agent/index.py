import json
import os
import boto3

# Initialize AWS clients
bedrock = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')

# Get environment variables
EMBEDDINGS_TABLE = os.environ.get('EMBEDDINGS_TABLE')
EMBEDDING_MODEL_ID = os.environ.get('EMBEDDING_MODEL_ID')
LLM_MODEL_ID = os.environ.get('LLM_MODEL_ID')
SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', '0.7'))
MAX_CONTEXT_DOCS = int(os.environ.get('MAX_CONTEXT_DOCS', '5'))

# Get table reference
embeddings_table = dynamodb.Table(EMBEDDINGS_TABLE)

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

def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve_relevant_context(query_embedding):
    """Retrieve relevant documents based on embedding similarity"""
    # In a real implementation, you would use a vector database
    # This is a simplified version that scans the DynamoDB table
    response = embeddings_table.scan()
    items = response.get('Items', [])
    
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
    return similarities[:MAX_CONTEXT_DOCS]

def call_claude(query, context_docs, question_type=None):
    """Call Claude 3.5 Sonnet with the query and context"""
    # Prepare context from retrieved documents
    context_text = "\n\n".join([doc['content'] for doc in context_docs])
    
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
    
    # Call Claude
    response = bedrock.invoke_model(
        modelId=LLM_MODEL_ID,
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1000,
            "messages": messages
        })
    )
    
    response_body = json.loads(response['body'].read())
    return response_body['content'][0]['text']

def lambda_handler(event, context):
    """Handle API Gateway requests for the LLM agent"""
    try:
        # Parse request body
        body = json.loads(event.get('body', '{}'))
        query = body.get('query')
        question_type = body.get('question_type')  # multiple_choice, yes_no, true_false, or null
        
        if not query:
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
        print(f"Error processing request: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': str(e)})
        }