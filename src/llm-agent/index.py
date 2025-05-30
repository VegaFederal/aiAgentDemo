import json
import os
import boto3
import logging

# Configure logging for python
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
bedrock = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')

# Get environment variables
EMBEDDINGS_TABLE = os.environ.get('EMBEDDINGS_TABLE')
EMBEDDING_MODEL_ID = os.environ.get('EMBEDDING_MODEL_ID', 'amazon.titan-embed-text-v2:0')
LLM_MODEL_ID = os.environ.get('LLM_MODEL_ID', 'anthropic.claude-3-5-sonnet-20241022-v2:0')
SIMILARITY_THRESHOLD = float(os.environ.get('SIMILARITY_THRESHOLD', '0.7'))
MAX_CONTEXT_DOCS = int(os.environ.get('MAX_CONTEXT_DOCS', '5'))

# Log configuration
logger.info(f"Configuration: EMBEDDINGS_TABLE={EMBEDDINGS_TABLE}, EMBEDDING_MODEL_ID={EMBEDDING_MODEL_ID}, LLM_MODEL_ID={LLM_MODEL_ID}")

# Validate required configuration
if not EMBEDDINGS_TABLE:
    logger.error("Missing required environment variable: EMBEDDINGS_TABLE")
    
if not EMBEDDING_MODEL_ID:
    logger.error("Missing required environment variable: EMBEDDING_MODEL_ID")
    
if not LLM_MODEL_ID:
    logger.error("Missing required environment variable: LLM_MODEL_ID")

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
    embedding = response_body['embedding']
    logger.info(f"Generated embedding of dimension {len(embedding)}")
    return embedding

def cosine_similarity(vec_a, vec_b):
    """Calculate cosine similarity between two vectors"""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    return dot_product / (norm_a * norm_b)

def retrieve_relevant_context(query_embedding):
    """Retrieve relevant documents based on embedding similarity"""
    logger.info("Using optimized retrieval method")
    
    # Since vector search isn't available, use the optimized fallback method directly
    return retrieve_relevant_context_fallback(query_embedding)

def retrieve_relevant_context_fallback(query_embedding):
    """Optimized fallback method for large datasets"""
    logger.info("Using optimized fallback method with aggressive batching")
    
    # Increase batch size for fewer network calls
    batch_size = 500
    
    # Only retrieve necessary fields
    projection_expression = "id, document_id, content, embedding_json"
    
    # Use early stopping with a lower threshold for initial filtering
    initial_threshold = SIMILARITY_THRESHOLD * 0.8
    similarities = []
    last_evaluated_key = None
    processed_count = 0
    max_processed = 15600  # Safety limit to avoid processing too many items
    
    while processed_count < max_processed:
        # Prepare scan parameters for this batch
        scan_params = {
            'TableName': EMBEDDINGS_TABLE,
            'Limit': batch_size,
            'ProjectionExpression': projection_expression
        }
        
        if last_evaluated_key:
            scan_params['ExclusiveStartKey'] = last_evaluated_key
        
        # Execute the scan for this batch
        response = dynamodb.meta.client.scan(**scan_params)
        batch_items = response.get('Items', [])
        processed_count += len(batch_items)
        
        logger.info(f"Processing batch of {len(batch_items)} items, total processed: {processed_count}")
        
        # Process this batch
        batch_similarities = []
        for item in batch_items:
            try:
                doc_embedding = json.loads(item.get('embedding_json', '[]'))
                similarity = cosine_similarity(query_embedding, doc_embedding)
                
                # Use initial lower threshold for faster filtering
                if similarity >= initial_threshold:
                    batch_similarities.append({
                        'document_id': item.get('document_id', ''),
                        'content': item.get('content', ''),
                        'similarity': similarity
                    })
            except Exception as e:
                logger.error(f"Error processing item: {str(e)}")
        
        # Add batch results to overall results
        similarities.extend(batch_similarities)
        logger.info(f"Found {len(batch_similarities)} potential matches in this batch, total: {len(similarities)}")
        
        # Update pagination key
        last_evaluated_key = response.get('LastEvaluatedKey')
        
        # Aggressive early stopping if we have enough good matches
        if len(similarities) >= MAX_CONTEXT_DOCS * 5:
            logger.info(f"Early stopping after finding {len(similarities)} potential matches")
            break
            
        # Break if no more pages
        if not last_evaluated_key:
            logger.info("Reached end of table")
            break
    
    # Final filtering with actual threshold and sorting
    logger.info(f"Filtering {len(similarities)} potential matches with threshold {SIMILARITY_THRESHOLD}")
    final_similarities = [doc for doc in similarities if doc['similarity'] >= SIMILARITY_THRESHOLD]
    final_similarities.sort(key=lambda x: x['similarity'], reverse=True)
    selected_docs = final_similarities[:MAX_CONTEXT_DOCS]
    
    logger.info(f"Selected {len(selected_docs)} most relevant documents from {processed_count} total records")
    return selected_docs


def call_claude(query, context_docs, question_type=None):
    """Call Claude 3.5 Sonnet with the query and context"""
    # Prepare context from retrieved documents
    context_text = "\n\n".join([doc['content'] for doc in context_docs])
    logger.info(f"Calling Claude with context length {len(context_text)} and question type {question_type}")
    
    # Verify model ID is available
    if not LLM_MODEL_ID:
        error_msg = "Cannot call LLM: Model ID is not configured"
        logger.error(error_msg)
        return f"Error: {error_msg}. Please check the Lambda configuration."
    
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
        {"role": "user", "content": system_prompt},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
    ]
    
    # Call Claude with inference profile
    try:
        logger.info(f"Invoking Bedrock model: {LLM_MODEL_ID}")
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
        error_message = f"Error calling LLM: {str(e)}"
        return error_message

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
        
        # Check for required environment variables
        if not EMBEDDINGS_TABLE:
            error_msg = "Missing required environment variable: EMBEDDINGS_TABLE"
            logger.error(error_msg)
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': error_msg})
            }
        
        # Generate embedding for the query
        try:
            query_embedding = generate_embedding(query)
        except Exception as e:
            error_msg = f"Error generating embedding: {str(e)}"
            logger.error(error_msg)
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': error_msg})
            }
        
        # Retrieve relevant context
        try:
            context_docs = retrieve_relevant_context(query_embedding)
        except Exception as e:
            error_msg = f"Error retrieving context: {str(e)}"
            logger.error(error_msg)
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': error_msg})
            }
        
        # Call Claude with the query and context
        response = call_claude(query, context_docs, question_type)
        
        # Check if response contains error message
        if response and response.startswith("Error:"):
            logger.error(f"LLM error: {response}")
            return {
                'statusCode': 500,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({'error': response})
            }
        
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
