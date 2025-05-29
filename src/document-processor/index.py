import json
import os
import boto3
import logging
import re
import time

# Initialize AWS clients
s3 = boto3.client('s3')
bedrock = boto3.client('bedrock-runtime')
dynamodb = boto3.resource('dynamodb')

# Initialize logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Environment variables
EMBEDDINGS_TABLE = os.environ.get('EMBEDDINGS_TABLE')
EMBEDDING_MODEL_ID = os.environ.get('EMBEDDING_MODEL_ID')
CHUNK_SIZE = int(os.environ.get('CHUNK_SIZE', '1000'))
CHUNK_OVERLAP = int(os.environ.get('CHUNK_OVERLAP', '200'))

# Get table reference
embeddings_table = dynamodb.Table(EMBEDDINGS_TABLE)

def extract_links_from_html(html_content):
    """
    Extract and normalize all hyperlinks from HTML content.
    
    This function parses HTML content to find all anchor tags (<a>) with href attributes,
    extracts both the URL and the link text, normalizes the URLs by removing fragments,
    and filters out external links and special protocols.
    
    Args:
        html_content (str): The HTML content to parse for links.
        
    Returns:
        list: A list of dictionaries, each containing:
            - 'url' (str): The normalized URL from the href attribute.
            - 'text' (str): The cleaned text content of the anchor tag.
            
    Example:
        >>> html = '<a href="page1.html">Go to Page 1</a><a href="http://external.com">External</a>'
        >>> extract_links_from_html(html)
        [{'url': 'page1.html', 'text': 'Go to Page 1'}]
        
    Notes:
        - Only internal links are returned (external links like http://, https://, mailto: are filtered out)
        - URL fragments (e.g., #section) are removed from the URLs
        - HTML tags within the link text are removed
        - Empty links are filtered out
        - The function uses a regex pattern to find links, which handles various HTML formatting styles
    """
    links = []
    # Find all href attributes in anchor tags
    href_pattern = re.compile(r'<a\s+[^>]*href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    matches = href_pattern.findall(html_content)
    
    for match in matches:
        link_url = match[0]
        link_text = re.sub(r'<[^>]+>', '', match[1]).strip()  # Remove any HTML tags in link text
        
        # Normalize the link (remove fragments, etc.)
        link_url = link_url.split('#')[0]  # Remove fragment
        
        # Only include internal links (exclude external links and special protocols)
        if link_url and not link_url.startswith(('http://', 'https://', 'mailto:')):
            links.append({
                'url': link_url,
                'text': link_text
            })
    
    return links


def extract_document_id(filename, html_content):
    """Extract document ID from filename or HTML content"""
    # Try to extract from filename patterns
    if re.match(r'FAR_\d+\.html?', filename):
        return re.search(r'FAR_(\d+)\.html?', filename).group(1), 'FAR'
    elif re.match(r'Part_\d+\.html?', filename):
        return re.search(r'Part_(\d+)\.html?', filename).group(1), 'Part'
    elif re.match(r'Subpart_\d+\.\d+\.html?', filename):
        return re.search(r'Subpart_(\d+\.\d+)\.html?', filename).group(1), 'Subpart'
    elif re.match(r'\d+\.\d+-\d+\.html?', filename):
        return re.search(r'(\d+\.\d+-\d+)\.html?', filename).group(1), 'Section'
    elif re.match(r'\d+\.\d+\.html?', filename):
        return re.search(r'(\d+\.\d+)\.html?', filename).group(1), 'Section'
    elif re.match(r'Corrections+\.html?', filename):
        return re.match(r'(Corrections)+\.html?', filename).group(1), 'Extra'
    
    # Try to extract from HTML content
    if html_content:
        # Look for ID in title or h1 elements
        title_match = re.search(r'<title>(.*?)</title>', html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            title = title_match.group(1)
            # Extract patterns like "27.203" or "Part 27" from title
            id_match = re.search(r'(\d+\.\d+(?:-\d+)?)', title)
            if id_match:
                return id_match.group(1), 'Section'
            id_match = re.search(r'Part\s+(\d+)', title, re.IGNORECASE)
            if id_match:
                return id_match.group(1), 'Part'
            id_match = re.search(r'Subpart\s+(\d+\.\d+)', title, re.IGNORECASE)
            if id_match:
                return id_match.group(1), 'Subpart'
        
        # Look for ID in h1 elements
        h1_match = re.search(r'<h1[^>]*>(.*?)</h1>', html_content, re.IGNORECASE | re.DOTALL)
        if h1_match:
            h1_text = h1_match.group(1)
            # Extract numerical patterns
            id_match = re.search(r'<span[^>]*>(\d+\.\d+(?:-\d+)?)</span>', h1_text)
            if id_match:
                return id_match.group(1), 'Section'
            id_match = re.search(r'<span[^>]*>Part\s+(\d+)</span>', h1_text, re.IGNORECASE)
            if id_match:
                return id_match.group(1), 'Part'
            id_match = re.search(r'<span[^>]*>Subpart\s+(\d+\.\d+)</span>', h1_text, re.IGNORECASE)
            if id_match:
                return id_match.group(1), 'Subpart'
    
    return None, None

def parse_document_structure(file_contents):
    """Parse document structure using HTML links to identify hierarchical relationships"""
    # Dictionary to store document hierarchy
    document_structure = {}
    link_map = {}  # Maps files to the files that link to them
    document_ids = {}  # Store document IDs and types
    
    # First pass: extract document IDs and build link map
    for filename, data in file_contents.items():
        if filename.endswith(('.html', '.htm')):
            # Get the original HTML content before tag removal
            html_content = data.get('original_html', '')
            if html_content:
                # Extract document ID and type
                doc_id, doc_type = extract_document_id(filename, html_content)

                if doc_id:
                    document_ids[filename] = {
                        'id': doc_id,
                        'type': doc_type
                    }
                
                # Extract links
                links = extract_links_from_html(html_content)

                for link in links:
                    # Normalize link to match filenames
                    link_url = link['url']
                    link_basename = os.path.basename(link_url)
                    if link_basename in file_contents:
                        # Add to link map with link text for context
                        if link_basename not in link_map:
                            link_map[link_basename] = []
                        link_map[link_basename].append({
                            'source': filename,
                            'text': link['text']
                        })
    
    # Identify top-level files (typically FAR files or those not linked from others)
    all_linked_files = set()
    for links in link_map.values():
        all_linked_files.update([link['source'] for link in links])

    all_files = set(file_contents.keys())
    unlinked_files = [f for f in all_files - all_linked_files if f.endswith(('.html', '.htm'))]
    far_files = [f for f in all_files if re.match(r'FAR_\d+\.html?', f)]

    # Combine both approaches to get all top-level files
    top_level_files = list(set(unlinked_files + far_files))
    logger.info(f"Unlinked files: {unlinked_files}")
    logger.info(f"FAR files: {far_files}")
    logger.info(f"Combined top-level files: {top_level_files}")

    # If still no top-level files found, use all HTML files
    if not top_level_files:
        top_level_files = [f for f in all_files if f.endswith(('.html', '.htm'))]
        logger.warning(f"No top-level files identified, using all HTML files: {top_level_files}")

    # Track visited files to prevent circular references
    visited_files = set()
    
    # Build hierarchy starting from top-level files
    def build_hierarchy(filename, level=1, parent_id=None, parent_path=None, path_so_far=None):
        if path_so_far is None:
            path_so_far = []
            
        logger.info(f"Building hierarchy for {filename} at level {level} with parent ID {parent_id}")
        
        # Check for circular references
        if filename in path_so_far:
            logger.warning(f"Circular reference detected: {' -> '.join(path_so_far)} -> {filename}")
            return None
            
        # Add current file to path
        current_path = path_so_far + [filename]
        
        if filename not in file_contents:
            logger.warning(f"File not found: {filename}")
            return None
        
        # Mark as visited
        visited_files.add(filename)
        
        # Get document ID and type
        doc_info = document_ids.get(filename, {})
        doc_id = doc_info.get('id')
        doc_type = doc_info.get('type', 'Unknown')
        
        if not doc_id:
            doc_id = f"unknown_{level}_{len(document_structure)}"
        
        # Create node structure
        node = {
            'type': doc_type,
            'level': level,
            'id': doc_id,
            'filename': filename,
            'children': []
        }
        
        if parent_id:
            node['parent_id'] = parent_id
        
        if parent_path:
            node['path'] = f"{parent_path} > {doc_id}"
        else:
            node['path'] = doc_id
        
        # Find children (files that this file links to)
        html_content = file_contents[filename].get('original_html', '')
        if html_content:
            links = extract_links_from_html(html_content)
            for link in links:
                link_url = link['url']
                link_basename = os.path.basename(link_url)
                
                # Skip if it's the same file or already processed as a child of this node
                if (link_basename == filename or 
                    any(child['filename'] == link_basename for child in node['children'])):
                    continue
                    
                if link_basename in file_contents:
                    # Check if this is a child document based on ID patterns
                    is_child = False
                    
                    # Check document type hierarchy
                    if doc_type == 'FAR' and document_ids.get(link_basename, {}).get('type') == 'Part':
                        is_child = True
                    elif doc_type == 'Part' and document_ids.get(link_basename, {}).get('type') == 'Subpart':
                        is_child = True
                    elif doc_type == 'Subpart' and document_ids.get(link_basename, {}).get('type') == 'Section':
                        is_child = True
                    # Check ID patterns (e.g., Part 27 links to 27.xxx)
                    elif doc_id and document_ids.get(link_basename, {}).get('id', '').startswith(doc_id):
                        is_child = True
                    
                    if is_child:
                        # Recursively build hierarchy for child, passing the current path
                        child = build_hierarchy(link_basename, level + 1, doc_id, node['path'], current_path)
                        if child:
                            node['children'].append(child)
        
        return node
    
    # Build the full hierarchy
    for top_file in sorted(top_level_files):
        if top_file not in visited_files:  # Only process files that haven't been visited yet
            hierarchy = build_hierarchy(top_file)
            if hierarchy:
                document_structure[top_file] = hierarchy
    
    return document_structure


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks with respect to semantic boundaries"""
    logger.info(f"Chunking text of length {len(text)} with semantic awareness")
    
    # Ensure chunk_overlap is less than chunk_size
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 2
        logger.warning(f"Chunk overlap was too large. Adjusted to {chunk_overlap}")
    
    # For very short texts, just return the text as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    # Define semantic boundary markers in order of preference
    section_boundaries = [
        # Section and subsection patterns
        r'\n\s*\d+\.\d+\s+[A-Z]',  # Section numbers like "1.1 Title"
        r'\n\s*\([a-z]\)\s+',      # Subsection markers like "(a) Text"
        
        # Paragraph boundaries
        r'\n\n+',                  # Multiple newlines
        
        # Sentence boundaries
        r'\.(?=\s+[A-Z])',         # Period followed by space and capital letter
        r'\?(?=\s+[A-Z])',         # Question mark followed by space and capital letter
        r'!(?=\s+[A-Z])',          # Exclamation mark followed by space and capital letter
        
        # Clause boundaries
        r';(?=\s+[a-z])',          # Semicolon followed by space and lowercase letter
        r':(?=\s+[a-z])',          # Colon followed by space and lowercase letter
        
        # Phrase boundaries
        r',(?=\s+[a-z])',          # Comma followed by space and lowercase letter
        
        # Word boundaries (last resort)
        r'\s+'                      # Any whitespace
    ]
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Determine the end of the current chunk
        end = min(start + chunk_size, len(text))
        
        # If we're not at the end of the text, look for a semantic boundary
        if end < len(text):
            # Try each boundary type in order of preference
            found_boundary = False
            
            for boundary_pattern in section_boundaries:
                # Look for the last occurrence of the boundary pattern before the end
                matches = list(re.finditer(boundary_pattern, text[start:end]))
                if matches:
                    # Get the last match position
                    last_match = matches[-1]
                    boundary_pos = start + last_match.start()
                    
                    # Only use the boundary if it's not too close to the start
                    if boundary_pos > start + (chunk_size // 4):  # At least 25% of chunk size
                        end = boundary_pos
                        found_boundary = True
                        logger.debug(f"Found boundary at position {end} using pattern {boundary_pattern}")
                        break
            
            if not found_boundary:
                logger.debug("No suitable semantic boundary found, using maximum chunk size")
        
        # Extract the chunk and clean it
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        
        # Move the start position for the next chunk
        start = end - chunk_overlap
        
        # Ensure we're making progress
        if start <= end - chunk_size:
            start = end - chunk_overlap
            
        # Safety check to prevent infinite loops
        if start >= len(text):
            break
    
    logger.info(f"Created {len(chunks)} semantically meaningful chunks")
    return chunks

def generate_embedding(text, max_retries=3, fallback_model_ids=None):
    """
    Generate embeddings for the given text using Amazon Titan with retry logic and fallbacks
    
    Args:
        text: The text to generate embeddings for
        max_retries: Maximum number of retry attempts
        fallback_model_ids: List of fallback model IDs to try if the primary model fails
    
    Returns:
        List of embedding values
    """
    if fallback_model_ids is None:
        fallback_model_ids = [
            "amazon.titan-embed-text-v1",  # Fallback to older version
            "cohere.embed-english-v3"      # Fallback to alternative model
        ]
    
    # Ensure text is not too long for the model
    if len(text) > 8000:  # Most embedding models have limits around 8K tokens
        logger.warning(f"Text too long ({len(text)} chars), truncating to 8000 chars")
        text = text[:8000]
    
    # Try primary model with retries
    retry_count = 0
    last_exception = None
    
    while retry_count < max_retries:
        try:
            logger.info(f"Generating embedding using {EMBEDDING_MODEL_ID} (attempt {retry_count + 1})")
            response = bedrock.invoke_model(
                modelId=EMBEDDING_MODEL_ID,
                body=json.dumps({
                    "inputText": text
                })
            )
            response_body = json.loads(response['body'].read())
            
            # Validate the response
            if 'embedding' in response_body and response_body['embedding']:
                logger.info(f"Successfully generated embedding with {EMBEDDING_MODEL_ID}")
                return response_body['embedding']
            else:
                logger.warning(f"Invalid response from {EMBEDDING_MODEL_ID}: {response_body}")
                retry_count += 1
                time.sleep(1)  # Add a small delay before retrying
        
        except Exception as e:
            last_exception = e
            logger.warning(f"Error generating embedding with {EMBEDDING_MODEL_ID} (attempt {retry_count + 1}): {str(e)}")
            retry_count += 1
            time.sleep(2 ** retry_count)  # Exponential backoff
    
    # If primary model failed, try fallback models
    logger.warning(f"Primary model {EMBEDDING_MODEL_ID} failed after {max_retries} attempts. Trying fallbacks.")
    
    for fallback_id in fallback_model_ids:
        try:
            logger.info(f"Trying fallback model {fallback_id}")
            response = bedrock.invoke_model(
                modelId=fallback_id,
                body=json.dumps({
                    "inputText": text
                })
            )
            response_body = json.loads(response['body'].read())
            
            if 'embedding' in response_body and response_body['embedding']:
                logger.info(f"Successfully generated embedding with fallback model {fallback_id}")
                return response_body['embedding']
            else:
                logger.warning(f"Invalid response from fallback model {fallback_id}: {response_body}")
        
        except Exception as e:
            logger.warning(f"Error with fallback model {fallback_id}: {str(e)}")
    
    # If all models fail, generate a simple hash-based embedding as last resort
    logger.error("All embedding models failed. Generating simple hash-based embedding.")
    try:
        # Create a simple embedding based on hash of text
        import hashlib
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # Convert hash to a list of 1536 float values (typical embedding dimension)
        simple_embedding = []
        for i in range(1536):
            # Use modulo to cycle through the hash bytes
            byte_val = hash_bytes[i % len(hash_bytes)]
            # Convert to a float between -1 and 1
            float_val = (byte_val / 128.0) - 1.0
            simple_embedding.append(float_val)
        
        logger.warning("Using hash-based fallback embedding (low quality)")
        return simple_embedding
    
    except Exception as e:
        logger.error(f"Failed to generate even a hash-based embedding: {str(e)}")
        # If all else fails, return a zero vector
        logger.critical("Returning zero vector as last resort")
        return [0.0] * 1536  # Standard embedding size


def extract_hierarchy_metadata(text_chunk):
    """Extract hierarchical metadata from the chunk header"""
    hierarchy = {}
    
    # Look for document type
    type_match = re.search(r'Type: ([^\|]+)', text_chunk)
    logger.info(f"Type match: {type_match}")
    if type_match:
        hierarchy['doc_type'] = type_match.group(1).strip()
    
    # Look for document ID
    id_match = re.search(r'ID: ([^\|]+)', text_chunk)
    logger.info(f"ID match: {id_match}")
    if id_match:
        hierarchy['doc_id'] = id_match.group(1).strip()
    
    # Look for parent ID
    parent_match = re.search(r'Parent: ([^\|]+)', text_chunk)
    logger.info(f"Parent match: {parent_match}")
    if parent_match:
        hierarchy['parent_id'] = parent_match.group(1).strip()
    
    # Look for level information
    level_match = re.search(r'Level: ([^\|]+)', text_chunk)
    logger.info(f"Level match: {level_match}")
    if level_match:
        hierarchy['level'] = level_match.group(1).strip()
    
    # Look for path information
    path_match = re.search(r'Path: ([^\|]+)', text_chunk)
    logger.info(f"Path match: {path_match}")
    if path_match:
        hierarchy['path'] = path_match.group(1).strip()
        
        # Parse path components for easier querying
        path_components = [p.strip() for p in hierarchy['path'].split('>')]
        if len(path_components) > 0:
            hierarchy['path_components'] = path_components
    
    # Look for title
    title_match = re.search(r'Title: ([^\|]+)', text_chunk)
    logger.info(f"Title match: {title_match}")
    if title_match:
        hierarchy['title'] = title_match.group(1).strip()
    
    # Look for filename
    file_match = re.search(r'File: ([^\n]+)', text_chunk)
    logger.info(f"File match: {file_match}")
    if file_match:
        hierarchy['filename'] = file_match.group(1).strip()

    logger.info(f"Hierarchy: {hierarchy}")
    return hierarchy

def store_embedding(document_id, chunk_id, text_chunk, embedding, metadata):
    """Store embedding and metadata in DynamoDB with hierarchical information"""
    # Extract hierarchy information from the chunk
    hierarchy_metadata = extract_hierarchy_metadata(text_chunk)
    logger.info(f"Extracted hierarchy metadata: {hierarchy_metadata}")
    logger.info(f"Metadata: {metadata}")
    
    # Merge with existing metadata
    enhanced_metadata = {
        **metadata,
        'hierarchy': hierarchy_metadata
    }
    
    # Extract content without the header
    content_parts = text_chunk.split('\n\n', 1)
    content = content_parts[1] if len(content_parts) > 1 else text_chunk
    
    item = {
        'id': f"{document_id}_{chunk_id}",
        'document_id': document_id,
        'chunk_id': chunk_id,
        'content': content,  # Store content without header
        'full_chunk': text_chunk,  # Store full chunk with header
        'embedding_json': json.dumps(embedding),  # Store as JSON string
        'metadata': enhanced_metadata
    }
    logger.info(f"Storing item: {item}")
    
    # Add searchable attributes for hierarchy
    if 'doc_id' in hierarchy_metadata:
        item['doc_id'] = hierarchy_metadata['doc_id']
    
    if 'parent_id' in hierarchy_metadata:
        item['parent_id'] = hierarchy_metadata['parent_id']
        
    if 'path' in hierarchy_metadata:
        item['path'] = hierarchy_metadata['path']
    
    embeddings_table.put_item(Item=item)

def process_files(bucket, s3_objects):
    logger.info(f"Processing {len(s3_objects)} files in bucket: {bucket}")
    # Get list of all HTML/TXT files
    file_contents = {}
    for s3_object in s3_objects:
        file_name = s3_object['Key']
        logger.info(f"Processing file: {file_name}")
        if file_name.endswith(('.html', '.htm', '.txt')):
            try:
              # Get file content from S3
                file_obj = s3.get_object(Bucket=bucket, Key=file_name)
                content = file_obj['Body'].read().decode('utf-8', errors='ignore')
                if file_name.endswith(('.html', '.htm')):
                    # Store original HTML for link extraction
                    original_html = content
                    
                    # Extract title for metadata
                    title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                    title = title_match.group(1) if title_match else file_name
                    
                    # Extract document ID from content
                    doc_id, doc_type = extract_document_id(file_name, content)
                    logger.info(f"B1 Document ID: {doc_id}, Document Type: {doc_type}")
                    
                    # Remove HTML tags for text content
                    text_content = re.sub(r'<[^>]+>', ' ', content)
                    # Remove extra whitespace
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
                    
                    file_contents[file_name] = {
                        'content': text_content,
                        'original_html': original_html,
                        'title': title,
                        'doc_id': doc_id,
                        'doc_type': doc_type,
                        's3_metadata': s3_object.get('Metadata', {})
                    }
                else:
                    file_contents[file_name] = {
                        'content': content,
                        'title': file_name,
                        's3_metadata':s3_object.get('Metadata',{})
                    }
            except Exception as e:
                print(f"Error processing file {file_name}: {str(e)}")
        
    # Parse document structure using HTML links
    document_structure = parse_document_structure(file_contents)

    # Process files with hierarchical context
    structured_content = []
            
    # Helper function to process nodes recursively
    def process_node(node):
        logger.info(f"B2 Processing recursive node: {node}")
        filename = node.get('filename')
        if filename in file_contents:
            content = file_contents[filename]['content']
            title = file_contents[filename].get('title', filename)
            doc_type = node.get('type', 'Unknown')
            doc_id = node.get('id', 'unknown')
            
            # Use the path from the node if available
            current_path = node.get('path', doc_id)
            
            # Create hierarchical header
            part = {
                'type': doc_type,
                'doc_id': doc_id,
                'level': node.get('level', 0),
                'path': current_path,
                'title': title,
                'filename': filename,
                'content': content
           }
            
            if 'parent_id' in node:
                part['parent'] = node['parent_id']
            logger.info(f"B2 Part: {part}")
            structured_content.append(part)
            
            # Process children recursively
            for child in node.get('children', []):
                logger.info(f"B2 Child node: {child}")
                process_node(child)

    # Process top-level nodes
    for filename, node in document_structure.items():
        process_node(node)
    
    # Add any remaining files that weren't part of the hierarchy
    processed_files = set()
    for structure in document_structure.values():
        def collect_filenames(node):
            if 'filename' in node:
                processed_files.add(node['filename'])
            for child in node.get('children', []):
                collect_filenames(child)
        collect_filenames(structure)
    
    for file, data in file_contents.items():
        if file not in processed_files:
            title = data.get('title', file)
            doc_id = data.get('doc_id')
            doc_type = data.get('doc_type', 'Unknown')
            
            content = {
                'type': doc_type if doc_type else "Standalone",
                'doc_id': doc_id if doc_id else "unknown",
                'title': title,
                'filename': file,
                'content': data['content']
            }
            logger.info(f"B2 Content: {content}")
            structured_content.append(content)

    # Combine all text with separators
    return structured_content

def process_chunks(chunks, document_id, metadata):
    logger.info(f"Chunking text into {len(chunks)} chunks for document {document_id}")
    # Process each chunk
    for i, chunk in enumerate(chunks):
        # Generate embedding
        embedding = generate_embedding(chunk)
        logger.info(f"Chunk {i} embedding generated")
        
        # Store in DynamoDB
        store_embedding(document_id, i, chunk, embedding, metadata)
        

def process_bucket(bucket):
    """Process all documents in a bucket or specific prefix"""
    try:
        logger.info(f"Processing documents in bucket: {bucket}")

        # List all objects in the bucket with the given prefix
        response = s3.list_objects_v2(Bucket=bucket)

        if 'Contents' not in response:
            logger.info(f"No files found in bucket {bucket}")
            return {
                'statusCode': 404,
                'body': json.dumps({
                    'message': f'No files found in bucket {bucket}'
                })
            }
        structured_content = process_files(bucket, response['Contents'])
        total_chunks = 0
        processed_docs = []

        for content_item in structured_content:
            text = content_item['content']
            document_id = content_item['doc_id']

            # Split text into chunks
            chunks = chunk_text(text)
            total_chunks += len(chunks)

            # Create metadata
            metadata = {
                'filename': content_item['filename'],
                'source_bucket': bucket,
                'source_key': content_item['filename'],
                **content_item.get('s3_metadata', {})
            }
            process_chunks(chunks, document_id, metadata)
            processed_docs.append(document_id)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Successfully processed bucket {bucket}',
                'documents_processed': len(processed_docs),
                'chunks_processed': total_chunks
            })
        }
    except Exception as e:
        print(f"Error processing bucket: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Error processing bucket: {str(e)}'
            })
        }
def lambda_handler(event, context):
    """Process all documents in a bucket or specific prefix"""
    logger.info(f"configuration info: table: {EMBEDDINGS_TABLE}, model: {EMBEDDING_MODEL_ID}")
    logger.info(f"Environment Vars: {os.environ}")
    try:
        # Get bucket and optional prefix from event
        bucket = event.get('bucket')
        prefix = event.get('prefix', '')
        logger.info(f"Processing documents in bucket: {bucket} with prefix: {prefix}")
        
        if not bucket:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'message': 'Bucket name is required'
                })
            }
        return process_bucket(bucket)        
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Error processing documents: {str(e)}'
            })
        }