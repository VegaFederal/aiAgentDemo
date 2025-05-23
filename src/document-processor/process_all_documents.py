import json
import os
import boto3
import logging
import uuid
import re
import zipfile
import tempfile

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
    logging.info("Extracting links from HTML content")
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
    top_level_files = [f for f in all_files - all_linked_files if f.endswith(('.html', '.htm'))]
    
    # If no top-level files found, use FAR files as top level
    if not top_level_files:
        top_level_files = [f for f in all_files if re.match(r'FAR_\d+\.html?', f)]
    
    # Build hierarchy starting from top-level files
    def build_hierarchy(filename, level=1, parent_id=None, parent_path=None):
        if filename not in file_contents:
            return None
        
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
                    
                    if is_child:
                        child = build_hierarchy(link_basename, level + 1, doc_id, node['path'])
                        if child:
                            node['children'].append(child)
    
        
        logger.info(f"Node Returned = {node}")
        return node
    
    # Build the full hierarchy
    for top_file in sorted(top_level_files):
        hierarchy = build_hierarchy(top_file)
        if hierarchy:
            document_structure[top_file] = hierarchy
    
    return document_structure

def process_zip_file(zip_content, key):
    """Process a ZIP file and extract text from all HTML files with hierarchical structure based on HTML links"""
    print(f"Processing ZIP file: {key}")
    logger.info(f"Processing ZIP file: {key}")
    
    # Create a temporary directory to extract files
    with tempfile.TemporaryDirectory() as temp_dir:
        zip_path = os.path.join(temp_dir, "archive.zip")
        
        # Write the ZIP content to a file
        with open(zip_path, 'wb') as f:
            f.write(zip_content)
        
        # Extract all files
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        
        # Get list of all HTML/TXT files
        file_contents = {}
        
        for root, dirs, files in os.walk(temp_dir):
            for file in files:
                if file.endswith(('.html', '.htm', '.txt')):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        try:
                            content = f.read()
                            if file.endswith(('.html', '.htm')):
                                # Store original HTML for link extraction
                                original_html = content
                                
                                # Extract title for metadata
                                title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                                title = title_match.group(1) if title_match else file
                                
                                # Extract document ID from content
                                doc_id, doc_type = extract_document_id(file, content)
                                
                                # Remove HTML tags for text content
                                text_content = re.sub(r'<[^>]+>', ' ', content)
                                # Remove extra whitespace
                                text_content = re.sub(r'\s+', ' ', text_content).strip()
                                
                                file_contents[file] = {
                                    'content': text_content,
                                    'original_html': original_html,
                                    'title': title,
                                    'doc_id': doc_id,
                                    'doc_type': doc_type
                                }
                            else:
                                file_contents[file] = {
                                    'content': content,
                                    'title': file
                                }
                        except Exception as e:
                            print(f"Error processing file {file}: {str(e)}")
        
        # Parse document structure using HTML links
        document_structure = parse_document_structure(file_contents)
        
        # Process files with hierarchical context
        structured_content = []
        
        # Helper function to process nodes recursively
        def process_node(node, parent_path=None):
            filename = node.get('filename')
            if filename in file_contents:
                content = file_contents[filename]['content']
                title = file_contents[filename].get('title', filename)
                doc_type = node.get('type', 'Unknown')
                doc_id = node.get('id', 'unknown')
                
                # Use the path from the node if available
                current_path = node.get('path', doc_id)
                
                # Create hierarchical header
                header_parts = [
                    f"Type: {doc_type}",
                    f"ID: {doc_id}",
                    f"Level: {node.get('level', 0)}",
                    f"Path: {current_path}",
                    f"Title: {title}",
                    f"File: {filename}"
                ]
                
                if 'parent_id' in node:
                    header_parts.insert(2, f"Parent: {node['parent_id']}")
                    
                header = " | ".join(header_parts)
                structured_content.append(f"{header}\n\n{content}")
                
                # Process children recursively
                for child in node.get('children', []):
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
                
                header_parts = [
                    f"Type: {doc_type}" if doc_type else "Type: Standalone",
                    f"ID: {doc_id}" if doc_id else "ID: unknown",
                    f"Title: {title}",
                    f"File: {file}"
                ]
                
                header = " | ".join(header_parts)
                structured_content.append(f"{header}\n\n{data['content']}")
    
    # Combine all text with separators
    return "\n\n---\n\n".join(structured_content)

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
    print(f"Hierarchy metadata: {hierarchy_metadata}")
    logger.info(f"Hierarchy metadata: {hierarchy_metadata}")
    
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
    print(f"Item: {item}")
    
    # Add searchable attributes for hierarchy
    if 'doc_id' in hierarchy_metadata:
        item['doc_id'] = hierarchy_metadata['doc_id']
    
    if 'parent_id' in hierarchy_metadata:
        item['parent_id'] = hierarchy_metadata['parent_id']
        
    if 'path' in hierarchy_metadata:
        item['path'] = hierarchy_metadata['path']
    
    embeddings_table.put_item(Item=item)

def lambda_handler(event, context):
    """Process all documents in a bucket or specific prefix"""
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
        
        # Check if we should process a ZIP file directly
        zip_key = event.get('zipKey')
        logger.info(f"ZIP key: {zip_key}")
        if zip_key:
            return process_single_zip(bucket, zip_key)
        
        # List all objects in the bucket with the given prefix
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        logger.info(f"Response: {response}")
        if 'Contents' not in response:
            logger.info(f"No files found in bucket {bucket} with prefix {prefix}")
            return {
                'statusCode': 404,
                'body': json.dumps({
                    'message': f'No files found in bucket {bucket} with prefix {prefix}'
                })
            }
        
        # Find ZIP files
        zip_files = [obj['Key'] for obj in response['Contents'] 
                    if obj['Key'].lower().endswith('.zip')]
        logger.info(f"ZIP files: {zip_files}")
        
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
        logger.info(f"Processing ZIP file: {key} from bucket: {bucket}")
        
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