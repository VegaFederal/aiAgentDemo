import json
import os
import boto3
import logging
import re

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
    """Extract all links from HTML content"""
    links = []
    # Find all href attributes in anchor tags
    href_pattern = re.compile(r'<a\s+[^>]*href=[\'"]([^\'"]+)[\'"][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
    matches = href_pattern.findall(html_content)
    for match in matches:
        link_url = match[0]
        link_text = re.sub(r'<[^>]+>', '', match[1]).strip()  # Remove any HTML tags in link text
        
        # Normalize the link (remove fragments, etc.)
        link_url = link_url.split('#')[0]  # Remove fragment
        if link_url and not link_url.startswith(('http://', 'https://', 'mailto:')):
            links.append({
                'url': link_url,
                'text': link_text
            })
    return links

def extract_document_id(filename, html_content):
    """Extract document ID from filename or HTML content"""
    logger.info(f"Extracting document ID from filename: {filename}")
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
                logger.info(f"A1 Extracted document ID: {doc_id} for file: {filename}")

                if doc_id:
                    document_ids[filename] = {
                        'id': doc_id,
                        'type': doc_type
                    }
                
                # Extract links
                links = extract_links_from_html(html_content)
                logger.info(f"A1 Extracted links: {links} from {filename}")

                for link in links:
                    # Normalize link to match filenames
                    link_url = link['url']
                    link_basename = os.path.basename(link_url)
                    logger.info(f"A1 Normalized link basename: {link_basename}")
                    #logger.info(f"A1 File Contents: {file_contents}")
                    if html_content.find(link_basename):
                        logger.info(f"A1 Found {link_basename} in html")
                        # Add to link map with link text for context
                        if link_basename not in link_map:
                            link_map[link_basename] = []
                        link_map[link_basename].append({
                            'source': filename,
                            'text': link['text']
                        })
    
    logger.info(f"A1 Link Map: {link_map}")
    # Identify top-level files (typically FAR files or those not linked from others)
    all_linked_files = set()
    for links in link_map.values():
        all_linked_files.update([link['source'] for link in links])
    
    all_files = set(file_contents.keys())
    top_level_files = [f for f in all_files - all_linked_files if f.endswith(('.html', '.htm'))]
    
    # If no top-level files found, use FAR files as top level
    if not top_level_files:
        top_level_files = [f for f in all_files if re.match(r'FAR_\d+\.html?', f)]
    
    # Build hierarchy starting from top-level files
    def build_hierarchy(filename, level=1, parent_id=None, parent_path=None):
        if filename not in file_contents:
            logger.info(f"A2 File not found: {filename}")
            return None
        
        # Get document ID and type
        doc_info = document_ids.get(filename, {})
        doc_id = doc_info.get('id')
        doc_type = doc_info.get('type', 'Unknown')
        logger.info(f"A2 Document ID: {doc_id} for file: {filename} of type {doc_type}")
        
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
            logger.info(f"A2 Extracted links: {links}")
            for link in links:
                link_url = link['url']
                logger.info(f"A2 Link URL: {link_url}")
                link_basename = os.path.basename(link_url)
                logger.info(f"A2 Link basename: {link_basename} in {file_contents}")
                if link_basename in file_contents and link_basename != filename:
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
                    logger.info(f"A2 Is child: {is_child}")
                    
                    if is_child:
                        child = build_hierarchy(link_basename, level + 1, doc_id, node['path'])
                        if child:
                            node['children'].append(child)
    
        
        logger.info(f"A2 Node Returned = {node}")
        return node
    
    # Build the full hierarchy
    for top_file in sorted(top_level_files):
        hierarchy = build_hierarchy(top_file)
        logger.info(f"A1 Hierarchy = {hierarchy}")
        if hierarchy:
            document_structure[top_file] = hierarchy
    
    logger.info(f"A1 Document Structure = {document_structure}")
    return document_structure        

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks of specified size"""
    logger.info(f"Chunking text of length {len(text)}")
    logger.info(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")
    logger.info(f"Text: {text}")
    chunks = []
    
    # Ensure chunk_overlap is less than chunk_size to prevent infinite loops
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 2
        logger.warning(f"Chunk overlap was too large. Adjusted to {chunk_overlap}")
    
    if len(text) <= chunk_size:
        logger.info(f"Text length <= chunk size. No chunking needed.")
        chunks.append(text)
    else:
        logger.info(f"Text length > chunk size. Chunking needed.")
        start = 0
        end = min(start + chunk_size, len(text))
        while start < len(text) and end < len(text):
            end = min(start + chunk_size, len(text))
            logger.info(f"Start: {start}, End: {end}")
            
            # Try to find a natural break point
            if end < len(text):
                found_break = False
                for break_char in ['\n\n', '\n', '. ', '? ', '! ']:
                    break_pos = text.rfind(break_char, start, end)
                    logger.info(f"Break char: {break_char}, Break pos: {break_pos}")
                    if break_pos > start:
                        end = break_pos + len(break_char)
                        found_break = True
                        break
                
                # If no natural break found, just use the maximum chunk size
                if not found_break:
                    logger.info("No natural break found, using maximum chunk size")
            
            chunk = text[start:end].strip()
            if chunk:  # Only add non-empty chunks
                chunks.append(chunk)
            
            # Advance start position, ensuring it moves forward
            start = end - chunk_overlap            
            logger.info(f"New start: {start}")
    
    logger.info(f"Chunked text into {len(chunks)} chunks")
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
    logger.info(f"B1 Processing {len(s3_objects)} files in bucket: {bucket}")
    # Get list of all HTML/TXT files
    file_contents = {}
    for s3_object in s3_objects:
        file_name = s3_object['Key']
        logger.info(f"B1 Processing file: {file_name}")
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
    logger.info(f"B1 Document Structure: {document_structure}")

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
        logger.info(f"B1 Top-level node: {node}")
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
    logger.info(f"D1 Processing {len(chunks)} chunks for document {document_id}")
    # Process each chunk
    for i, chunk in enumerate(chunks):
        # Generate embedding
        embedding = generate_embedding(chunk)
        logger.info(f"D1 Chunk {i} embedding generated")
        
        # Store in DynamoDB
        store_embedding(document_id, i, chunk, embedding, metadata)
        

def process_bucket(bucket):
    """Process all documents in a bucket or specific prefix"""
    try:
        logger.info(f"C1 Processing documents in bucket: {bucket}")

        # List all objects in the bucket with the given prefix
        response = s3.list_objects_v2(Bucket=bucket)
        logger.info(f"C1 List of all objects in Bucket: {response}")

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

        logger.info(f"C1 Processing Structured Content for {len(structured_content)} content")
        for content_item in structured_content:
            logger.info(f"C1 Content Item: {content_item}")
            text = content_item['content']
            logger.info(f"C1 Content: {text}")
            document_id = content_item['doc_id']
            logger.info(f"C1 Document ID: {document_id}")

            logger.info(f"C1 Processing document {document_id}")
            # Split text into chunks
            chunks = chunk_text(text)
            logger.info(f"C1 Document {document_id} split into {len(chunks)} chunks")
            total_chunks += len(chunks)

            # Create metadata
            metadata = {
                'filename': content_item['filename'],
                'source_bucket': bucket,
                'source_key': content_item['filename'],
                **content_item.get('s3_metadata', {})
            }
            logger.info(f"C1 Metadata: {metadata}")
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