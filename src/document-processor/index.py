import datetime
import json
import os
import boto3
import logging
import re
import time
import uuid

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
    """
    Extract the document ID from the filename or HTML content.

    This function attempts to extract a document ID from the filename or HTML content.
    The document ID is expected to be in one of the following formats:
    - FAR_XXXX.html
    - Part_XXXX.html
    - Subpart_XXXX.XX.html
    - XXXX.XX-XX.html
    - XXXX.XX.html
    - Corrections.html

    Args:
        filename (str): The filename of the document.
        html_content (str): The HTML content of the document.

    Returns:
        tuple: A tuple containing two elements:
            - The document ID (str) if found, otherwise None.
            - The document type (str) if found, otherwise None.

    Example:
        >>> extract_document_id('FAR_1234.html', '')
        ('1234', 'FAR')
        >>> extract_document_id('Corrections.html', '')
        ('Corrections', 'Extra')
        >>> extract_document_id('unknown.html', '<title>Part 27</title>')
        ('27', 'Part')
    """
    # Remove file extension
    filename = os.path.splitext(filename)[0]
    logger.debug('Extracting document ID from filename: %s', filename)
    # Try to extract from filename patterns
# Try to extract from filename patterns (without .html extension)
    if re.match(r'FAR_\d+$', filename):
        logger.debug(f"Found FAR")
        return re.search(r'FAR_(\d+)$', filename).group(1), 'FAR'
    elif re.match(r'Part_\d+$', filename):
        logger.debug(f"Found Part")
        return re.search(r'Part_(\d+)$', filename).group(1), 'Part'
    elif re.match(r'Subpart_\d+\.\d+$', filename):
        logger.debug(f"Found Subpart")
        return re.search(r'Subpart_(\d+\.\d+)$', filename).group(1), 'Subpart'
    elif re.match(r'\d+\.\d+-\d+$', filename):
        logger.debug(f"Found Section")
        return re.search(r'(\d+\.\d+-\d+)$', filename).group(1), 'Section'
    elif re.match(r'\d+\.\d+$', filename):
        logger.debug(f"Found Section")
        return re.search(r'(\d+\.\d+)$', filename).group(1), 'Section'
    elif re.match(r'Corrections$', filename):
        logger.debug(f"Found Corrections")
        return re.match(r'(Corrections)$', filename).group(1), 'Extra'
    logger.debug("Didn't find anything in the file name.")
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
    """
    Parse the document structure from the given file contents.

    This function processes the file contents to identify the top-level document
    and its hierarchy, including links to other documents.

    Args:
        file_contents (dict): A dictionary where keys are filenames and values are
                               dictionaries containing 'original_html' and 'cleaned_html'.

    Returns:
        tuple: A tuple containing two elements:
            - A dictionary representing the document structure, where keys are filenames
              and values are lists of linked filenames.
            - A dictionary containing document IDs and types.

    Example:
        >>> file_contents = {
        ...     'FAR_1234.html': {'original_html': '<a href="Part_27.html">Link</a>', 'cleaned_html': 'Link'},
        ...     'Part_27.html': {'original_html': '', 'cleaned_html': ''}
        ... }
        >>> parse_document_structure(file_contents)
        ({'FAR_1234.html': ['Part_27.html']}, {'1234': 'FAR'})
    """
    logger.info("Parsing document structure")
    # Dictionary to store document hierarchy
    document_structure = {}
    link_map = {}  # Maps files to the files that link to them
    document_ids = {}  # Store document IDs and types
    
    # First pass: extract document IDs and build link map
    for filename, data in file_contents.items():
        logger.debug(f"Processing file: {filename}")
        if filename.endswith(('.html', '.htm')):
            # Get the original HTML content before tag removal
            html_content = data.get('original_html', '')
            if html_content:
                # Extract document ID and type
                doc_id, doc_type = extract_document_id(filename, html_content)
                logger.debug(f"Extracted document ID: {doc_id} and type: {doc_type}")

                if doc_id:
                    document_ids[filename] = {
                        'id': doc_id,
                        'type': doc_type
                    }
                
                # Extract links
                links = extract_links_from_html(html_content)
                logger.debug(f"Extracted links: {links}")

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
    logger.debug(f"All files: {all_files}")
    logger.debug(f"Unlinked files: {unlinked_files}")
    logger.debug(f"FAR files: {far_files}")
    logger.debug(f"Combined top-level files: {top_level_files}")

    # If still no top-level files found, use all HTML files
    if not top_level_files:
        top_level_files = [f for f in all_files if f.endswith(('.html', '.htm'))]
        logger.warning(f"No top-level files identified, using all HTML files: {top_level_files}")

    # Track visited files to prevent circular references
    visited_files = set()
    
    # Build hierarchy starting from top-level files
    def build_hierarchy(filename, level=1, parent_id=None, parent_path=None, path_so_far=None):
        """
        Recursively build the document structure by finding children of each file.

        Args:
            filename (string): name of the file
            level (int, optional): depth level. Defaults to 1.
            parent_id (string, optional): ID of the parent file. Defaults to None.
            parent_path (string, optional): path pulled from the url. Defaults to None.
            path_so_far (string, optional): relative path. Defaults to None.

        Returns:
            node: dictionary 
        """
        if path_so_far is None:
            path_so_far = []
            
        logger.debug(f"Building hierarchy for {filename} at level {level} with parent ID {parent_id}")
        
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
        logger.debug(f"Node = {node}")
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
    """
    Chunk the given text into smaller parts.

    This function splits the text into chunks based on the specified chunk size and overlap.
    It aims to preserve semantic boundaries in the text.

    Args:
        text (str): The text to be chunked.
        chunk_size (int, optional): The maximum size of each chunk. Defaults to CHUNK_SIZE.
        chunk_overlap (int, optional): The number of overlapping characters between chunks. Defaults to CHUNK_OVERLAP.

    Returns:
        list: A list of text chunks.
    """
    logger.debug(f"Chunking text of length {len(text)}")
    
    # Ensure chunk_overlap is less than chunk_size
    if chunk_overlap >= chunk_size:
        chunk_overlap = chunk_size // 2
        logger.warning(f"Chunk overlap was too large. Adjusted to {chunk_overlap}")

    logger.debug(f"Chunk size: {chunk_size}, Chunk overlap: {chunk_overlap}")
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
        logger.debug(f"Processing chunk from {start} to {end}")
        
        # If we're not at the end of the text, look for a semantic boundary
        if end < len(text):
            # Try each boundary type in order of preference
            found_boundary = False
            
            for boundary_pattern in section_boundaries:
                # Look for the last occurrence of the boundary pattern before the end
                matches = list(re.finditer(boundary_pattern, text[start:end]))
                logger.debug(f"Found {len(matches)} matches for pattern {boundary_pattern}")
                if matches:
                    # Get the last match position
                    last_match = matches[-1]
                    boundary_pos = start + last_match.start()
                    logger.debug(f"Last match position: {boundary_pos}")
                    
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
            logger.debug(f"Added chunk: {chunk[:50]}...")
        
        # Move the start position for the next chunk
        new_start = end - chunk_overlap
        
        # Ensure we're making progress
        if new_start <= end - chunk_size:
            new_start = end - chunk_overlap
        logger.debug(f"Next chunk starts at {start}")
        # Safety check to prevent infinite loops
        if start >= len(text) or new_start == start:
            break
        start = new_start
    logger.debug(f"Created {len(chunks)} semantically meaningful chunks")
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
            logger.debug(f"Generating embedding using {EMBEDDING_MODEL_ID} (attempt {retry_count + 1})")
            response = bedrock.invoke_model(
                modelId=EMBEDDING_MODEL_ID,
                body=json.dumps({
                    "inputText": text
                })
            )
            response_body = json.loads(response['body'].read())
            
            # Validate the response
            if 'embedding' in response_body and response_body['embedding']:
                logger.debug(f"Successfully generated embedding with {EMBEDDING_MODEL_ID}")
                embedding = response_body['embedding']
                
                # Store the embedding in the vector-compatible format for DynamoDB vector search
                return embedding
            else:
                logger.warning(f"Invalid response from {EMBEDDING_MODEL_ID}: {response_body}")
                retry_count += 1
                time.sleep(1)  # Add a small delay before retrying
        
        except Exception as e:
            last_exception = e
            logger.error(f"Error generating embedding with {EMBEDDING_MODEL_ID} (attempt {retry_count + 1}): {str(e)}")
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
            logger.error(f"Error with fallback model {fallback_id}: {str(e)}")
    
    # If all models fail, generate a simple hash-based embedding as last resort
    logger.warning("All embedding models failed. Generating simple hash-based embedding.")
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
    logger.debug(f"Type match: {type_match}")
    if type_match:
        hierarchy['doc_type'] = type_match.group(1).strip()
    
    # Look for document ID
    id_match = re.search(r'ID: ([^\|]+)', text_chunk)
    logger.debug(f"ID match: {id_match}")
    if id_match:
        hierarchy['doc_id'] = id_match.group(1).strip()
    
    # Look for parent ID
    parent_match = re.search(r'Parent: ([^\|]+)', text_chunk)
    logger.debug(f"Parent match: {parent_match}")
    if parent_match:
        hierarchy['parent_id'] = parent_match.group(1).strip()
    
    # Look for level information
    level_match = re.search(r'Level: ([^\|]+)', text_chunk)
    logger.debug(f"Level match: {level_match}")
    if level_match:
        hierarchy['level'] = level_match.group(1).strip()
    
    # Look for path information
    path_match = re.search(r'Path: ([^\|]+)', text_chunk)
    logger.debug(f"Path match: {path_match}")
    if path_match:
        hierarchy['path'] = path_match.group(1).strip()
        
        # Parse path components for easier querying
        path_components = [p.strip() for p in hierarchy['path'].split('>')]
        if len(path_components) > 0:
            hierarchy['path_components'] = path_components
    
    # Look for title
    title_match = re.search(r'Title: ([^\|]+)', text_chunk)
    logger.debug(f"Title match: {title_match}")
    if title_match:
        hierarchy['title'] = title_match.group(1).strip()
    
    # Look for filename
    file_match = re.search(r'File: ([^\n]+)', text_chunk)
    logger.debug(f"File match: {file_match}")
    if file_match:
        hierarchy['filename'] = file_match.group(1).strip()

    logger.debug(f"Hierarchy: {hierarchy}")
    return hierarchy

def store_embedding(document_id, chunk_id, text_chunk, embedding, metadata):
    """
    Store document embedding and metadata in DynamoDB.
    
    This function extracts hierarchical metadata from the text chunk header,
    merges it with the provided metadata, and stores the embedding along with
    the content in DynamoDB. It also adds top-level attributes for efficient
    querying by document ID, parent ID, and path.
    
    Args:
        document_id (str): Unique identifier for the document
        chunk_id (int): Identifier for this specific chunk within the document
        text_chunk (str): Text content with hierarchical header information
        embedding (list): Vector embedding generated from the text
        metadata (dict): Additional metadata to store with the embedding
        
    Returns:
        None
        
    Note:
        The text_chunk should have a header in the format:
        "Type: Part | ID: 27 | Level: 1 | Path: 27 | Title: Part 27 | File: Part_27.html"
        
        The function extracts hierarchical information from this header and
        stores it both within the metadata and as top-level attributes for
        efficient querying.
    """
    # Extract hierarchy information from the chunk
    hierarchy_metadata = extract_hierarchy_metadata(text_chunk)
    logger.debug(f"Extracted hierarchy metadata: {hierarchy_metadata}")
    logger.debug(f"Metadata: {metadata}")
    
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
        'embedding_json': json.dumps(embedding),  # Store as JSON string for backward compatibility
        'embedding_vector': embedding,  # Store as native list for vector search
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

def process_node(node, file_contents, structured_content):
    """
    Process a document node and add it to structured_content
    
    Args:
        node: The document node to process
        file_contents: Dictionary of file contents
        structured_content: List to append structured content to
    """
    logger.debug(f"Processing node: {node}")
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
        
        # Create structured content with header and content
        structured_content.append({
            'content': f"{header}\n\n{content}",
            'doc_id': doc_id,
            'file': filename,
            's3_key': file_contents[filename].get('s3_key'),
            's3_metadata': file_contents[filename].get('s3_metadata', {})
        })
        
        # Process children recursively
        for child in node.get('children', []):
            logger.debug(f"Child node: {child}")
            process_node(child, file_contents, structured_content)

def collect_filenames(node, processed_files):
    """Collect all filenames in a document structure"""
    if 'filename' in node:
        processed_files.add(node['filename'])
    for child in node.get('children', []):
        collect_filenames(child, processed_files)

def process_files(bucket, s3_objects):
    """
    Process files from an S3 bucket, extract content, and build document hierarchy.
    
    This function processes HTML and TXT files from an S3 bucket, extracts their content,
    builds a hierarchical document structure based on links between HTML files, and
    creates structured content with metadata headers for embedding generation.
    
    Args:
        bucket (str): Name of the S3 bucket containing the files
        s3_objects (list): List of S3 object dictionaries from list_objects_v2
        
    Returns:
        list: A list of structured content items, each containing:
            - 'content' (str): Text content with hierarchical header
            - 'doc_id' (str): Document identifier
            - 'file' (str): Original filename
            - 's3_key' (str): S3 object key
            - 's3_metadata' (dict): Metadata from the S3 object
            
    Note:
        The function builds a hierarchical document structure by:
        1. Extracting document IDs and types from filenames and HTML content
        2. Analyzing links between HTML files to determine parent-child relationships
        3. Building a tree structure representing the document hierarchy
        4. Creating structured content with headers containing hierarchical metadata
        
        The structured content format includes a header like:
        "Type: Part | ID: 27 | Level: 1 | Path: 27 | Title: Part 27 | File: Part_27.html"
        
        This header is later parsed by extract_hierarchy_metadata to maintain
        the hierarchical relationships when storing embeddings.
    """
    logger.info(f"Processing {len(s3_objects)} files in bucket: {bucket}")
    # Get list of all HTML/TXT files
    file_contents = {}
    for s3_object in s3_objects:
        file_name = s3_object['Key']
        logger.debug(f"Processing file: {file_name}")
        if file_name.endswith(('.html', '.htm', '.txt')):
            try:
                # Get file content from S3
                file_obj = s3.get_object(Bucket=bucket, Key=file_name)
                content = file_obj['Body'].read().decode('utf-8', errors='ignore')
                
                # Get metadata from file_obj, not s3_object
                metadata = file_obj.get('Metadata', {})
                
                if file_name.endswith(('.html', '.htm')):
                    # Store original HTML for link extraction
                    original_html = content
                    
                    # Extract title for metadata
                    title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                    title = title_match.group(1) if title_match else file_name
                    
                    # Extract document ID from content
                    doc_id, doc_type = extract_document_id(file_name, content)
                    logger.debug(f"Document ID: {doc_id}, Document Type: {doc_type}")
                    
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
                        's3_key': file_name,
                        's3_metadata': metadata  # Use metadata from file_obj
                    }
                    logger.debug(f"File contents: {file_contents[file_name]}")
                else:
                    file_contents[file_name] = {
                        'content': content,
                        'title': file_name,
                        's3_key': file_name,
                        's3_metadata': metadata  # Use metadata from file_obj
                    }
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {str(e)}")

    # Parse document structure using HTML links
    document_structure = parse_document_structure(file_contents)

    # Process files with hierarchical context
    structured_content = []
            
    # Process top-level nodes
    for filename, node in document_structure.items():
        process_node(node, file_contents, structured_content)
    
    # Add any remaining files that weren't part of the hierarchy
    processed_files = set()
    for structure in document_structure.values():
        collect_filenames(structure, processed_files)
    
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
            structured_content.append({
                'content': f"{header}\n\n{data['content']}",
                'doc_id': doc_id,
                'file': file,
                's3_key': data.get('s3_key'),
                's3_metadata': data.get('s3_metadata', {})
            })

    # Return the structured content
    return structured_content


def process_chunks(chunks, document_id, metadata):
    """
    Process chunks of text, generate embeddings, and store them in DynamoDB.

    This function processes a list of text chunks, generates embeddings for each chunk,
    and stores the embeddings along with the chunk content and metadata in DynamoDB.

    Args:
        chunks (list): List of text chunks to process
        document_id (str): Unique identifier for the document
        metadata (dict): Metadata to associate with each embedding

    Returns:
        None
    """
    logger.debug(f"Processing {len(chunks)} chunks for document {document_id}")
    # Process each chunk
    for i, chunk in enumerate(chunks):
        # Generate embedding
        embedding = generate_embedding(chunk)
        logger.debug(f"Chunk {i+1} embedding generated")
        
        # Store in DynamoDB
        store_embedding(document_id, i, chunk, embedding, metadata)
        

def process_bucket(bucket, prefix, checkpoint, time_limit_ms):
    """
    Process documents in an S3 bucket with checkpointing and time limits.
    
    This function processes documents in an S3 bucket in stages, with checkpointing
    to handle Lambda timeouts. It maintains the hierarchical structure of documents
    while processing them in batches, and can resume processing from where it left off.
    
    The processing is divided into stages:
    1. Collecting files: List all objects in the bucket
    2. Collecting contents: Get the content of each file
    3. Building structure: Parse the document structure based on links
    4. Creating structured content: Create content with hierarchical headers
    5. Processing structure: Generate embeddings and store in DynamoDB
    
    Args:
        bucket (str): Name of the S3 bucket containing the documents
        prefix (str): Prefix to filter objects in the bucket
        checkpoint (dict): Checkpoint data from a previous invocation, or a new checkpoint
        time_limit_ms (int): Time limit in milliseconds before checkpointing
        
    Returns:
        dict: A dictionary containing:
            - 'is_complete' (bool): Whether processing is complete
            - 'checkpoint' (dict): Updated checkpoint data for the next invocation
                - 'file_list' (list): List of files in the bucket
                - 'file_contents' (dict): Contents of processed files
                - 'document_structure' (dict): Hierarchical document structure
                - 'structured_content' (list): Content with hierarchical headers
                - 'processed_files' (list): Files that have been fully processed
                - 'processed_count' (int): Number of documents processed
                - 'chunk_count' (int): Number of chunks processed
                - 'stage' (str): Current processing stage
                
    Note:
        This function is designed to be called repeatedly until processing is complete.
        It uses the checkpoint to track progress and resume processing from where it
        left off, allowing it to handle large document sets that exceed Lambda timeouts.
        
        The function converts datetime objects to ISO format strings to ensure they
        can be serialized to JSON for checkpointing.
    """


    start_time = int(time.time() * 1000)
    result = {
        'is_complete': False,
        'checkpoint': checkpoint
    }
    
    # Stage 1: Collect all files
    if checkpoint['stage'] == 'collecting_files':
        logger.info("Stage 1: Collecting files")
        
        # List all objects in the bucket
        all_objects = []
        continuation_token = None
        
        while True:
            # Check time limit
            current_time = int(time.time() * 1000)
            if current_time - start_time > time_limit_ms:
                logger.warning(f"Time limit reached during file collection. Checkpointing.")
                # Convert datetime objects to strings before saving
                serializable_objects = []
                for obj in all_objects:
                    serializable_obj = {k: (v.isoformat() if isinstance(v, datetime.datetime) else v) 
                                       for k, v in obj.items()}
                    serializable_objects.append(serializable_obj)
                checkpoint['file_list'] = serializable_objects
                return result
            
            # List objects
            params = {'Bucket': bucket, 'Prefix': prefix}
            if continuation_token:
                params['ContinuationToken'] = continuation_token
                
            response = s3.list_objects_v2(**params)
            
            if 'Contents' in response:
                # Convert datetime objects to strings before adding to list
                for obj in response['Contents']:
                    serializable_obj = {k: (v.isoformat() if isinstance(v, datetime.datetime) else v) 
                                       for k, v in obj.items()}
                    all_objects.append(serializable_obj)
                
            if not response.get('IsTruncated'):
                break
                
            continuation_token = response.get('NextContinuationToken')
        
        logger.info(f"Found {len(all_objects)} objects")
        checkpoint['file_list'] = all_objects
        checkpoint['stage'] = 'collecting_contents'
    
    # Stage 2: Collect file contents
    if checkpoint['stage'] == 'collecting_contents':
        logger.info("Stage 2: Collecting file contents")
        
        # Process files that haven't been processed yet
        for s3_object in checkpoint['file_list']:
            # Check time limit
            current_time = int(time.time() * 1000)
            if current_time - start_time > time_limit_ms:
                logger.warning(f"Time limit reached during content collection. Checkpointing.")
                return result
            
            file_name = s3_object['Key']
            
            # Skip if already processed or not HTML/TXT
            if (file_name in checkpoint['file_contents'] or 
                not file_name.endswith(('.html', '.htm', '.txt'))):
                continue
                
            try:
                # Get file content
                file_obj = s3.get_object(Bucket=bucket, Key=file_name)
                content = file_obj['Body'].read().decode('utf-8', errors='ignore')
                
                if file_name.endswith(('.html', '.htm')):
                    # Process HTML file
                    original_html = content
                    title_match = re.search(r'<title>(.*?)</title>', content, re.IGNORECASE)
                    title = title_match.group(1) if title_match else file_name
                    doc_id, doc_type = extract_document_id(file_name, original_html)
                    text_content = re.sub(r'<[^>]+>', ' ', content)
                    text_content = re.sub(r'\s+', ' ', text_content).strip()
                    
                    checkpoint['file_contents'][file_name] = {
                        'content': text_content,
                        'original_html': original_html,
                        'title': title,
                        'doc_id': doc_id,
                        'doc_type': doc_type,
                        's3_key': file_name
                    }
                else:
                    # Process TXT file
                    checkpoint['file_contents'][file_name] = {
                        'content': content,
                        'title': file_name,
                        's3_key': file_name
                    }
                    
            except Exception as e:
                logger.error(f"Error processing file {file_name}: {str(e)}")
        
        logger.info(f"Collected contents for {len(checkpoint['file_contents'])} files")
        checkpoint['stage'] = 'building_structure'
    
    # Stage 3: Build document structure
    if checkpoint['stage'] == 'building_structure':
        logger.info("Stage 3: Building document structure")
        
        # Check time limit
        current_time = int(time.time() * 1000)
        if current_time - start_time > time_limit_ms:
            logger.warning(f"Time limit reached before building structure. Checkpointing.")
            return result
            
        # Build document structure
        checkpoint['document_structure'] = parse_document_structure(checkpoint['file_contents'])
        checkpoint['stage'] = 'creating_structured_content'
    
    # Stage 4: Create structured content
    if checkpoint['stage'] == 'creating_structured_content':
        logger.info("Stage 4: Creating structured content")
        
        # Check time limit
        current_time = int(time.time() * 1000)
        if current_time - start_time > time_limit_ms:
            logger.warning(f"Time limit reached before creating structured content. Checkpointing.")
            return result
            
        # Create structured content
        structured_content = []
        
        # Process top-level nodes
        for filename, node in checkpoint['document_structure'].items():
            process_node(node, checkpoint['file_contents'], structured_content)
        
        # Add any remaining files
        processed_files = set()
        for structure in checkpoint['document_structure'].values():
            collect_filenames(structure, processed_files)
        
        for file, data in checkpoint['file_contents'].items():
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
                structured_content.append({
                    'content': f"{header}\n\n{data['content']}",
                    'doc_id': doc_id or str(uuid.uuid4()),
                    'file': file,
                    's3_key': data.get('s3_key')
                })
        
        checkpoint['structured_content'] = structured_content
        checkpoint['stage'] = 'processing_structure'
    
    # Stage 5: Process structured content
    if checkpoint['stage'] == 'processing_structure':
        logger.info("Stage 5: Processing structured content")
        
        # Initialize processed_items if it doesn't exist
        if 'processed_items' not in checkpoint:
            checkpoint['processed_items'] = []
        
        # Process items that haven't been processed yet
        for i, content_item in enumerate(checkpoint['structured_content']):
            # Generate a unique ID for this content item
            item_id = f"{content_item.get('file')}_{i}"
            
            # Skip if already processed
            if item_id in checkpoint['processed_items']:
                logger.info(f"Already processed item {item_id}")
                continue
                
            # Check time limit
            current_time = int(time.time() * 1000)
            if current_time - start_time > time_limit_ms:
                logger.warning(f"Time limit reached during structure processing. Checkpointing.")
                return result
                
            try:
                # Process this content item
                text = content_item['content']
                document_id = content_item.get('doc_id') or str(uuid.uuid4())
                
                # Split text into chunks
                chunks = chunk_text(text)
                
                # Create metadata
                metadata = {
                    'filename': content_item['file'],
                    'source_bucket': bucket,
                    'source_key': content_item.get('s3_key')
                }
                
                # Process chunks
                process_chunks(chunks, document_id, metadata)
                
                # Update checkpoint
                if content_item['file'] not in checkpoint['processed_files']:
                    checkpoint['processed_files'].append(content_item['file'])
                
                # Mark this specific item as processed
                checkpoint['processed_items'].append(item_id)
                
                checkpoint['processed_count'] += 1
                checkpoint['chunk_count'] += len(chunks)
                
            except Exception as e:
                logger.error(f"Error processing content item {i}: {str(e)}")
        
        logger.info("Done with Stage 5")
        # Check if all items have been processed
        logger.info(f"processed_items vs structured_content {len(checkpoint['processed_items'])} ?? {len(checkpoint['structured_content'])} ")
        
        if len(checkpoint['processed_items']) >= len(checkpoint['structured_content']):
            logger.info("All items processed. Processing complete.")
            result['is_complete'] = True

    
    return result

def lambda_handler(event, context):
    """Process all documents in a bucket or specific prefix with checkpointing"""
    try:
        # Get parameters from event
        bucket = event.get('bucket')
        prefix = event.get('prefix', '')
        checkpoint_key = event.get('checkpoint_key')
        
        logger.info(f"Processing documents in bucket: {bucket} with prefix: {prefix}")
        logger.info(f"Checkpoint key: {checkpoint_key}")
        
        if not bucket:
            return {
                'statusCode': 400,
                'body': json.dumps({
                    'message': 'Bucket name is required'
                })
            }
        
        # Initialize or load checkpoint
        checkpoint = load_checkpoint(bucket, checkpoint_key)
        
        # Check if processing is already complete
        if checkpoint.get('is_complete', False):
            logger.info("Processing is already complete. Nothing to do.")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Processing already complete',
                    'documents_processed': checkpoint.get('processed_count', 0),
                    'chunks_processed': checkpoint.get('chunk_count', 0)
                })
            }
        
        # Calculate remaining time
        remaining_time = context.get_remaining_time_in_millis() if context else 300000  # Default to 5 minutes
        logger.debug(f"Remaining time: {remaining_time} ms")
        
        # Reserve 30 seconds for cleanup and checkpointing
        processing_time_limit = remaining_time - 30000
        
        # Process with time limit
        result = process_bucket(bucket, prefix, checkpoint, processing_time_limit)
        
        # Save checkpoint if not complete
        if not result['is_complete']:
            checkpoint_key = save_checkpoint(bucket, result['checkpoint'])
            
            # Create a new event for the next invocation
            next_event = {
                'bucket': bucket,
                'prefix': prefix,
                'checkpoint_key': checkpoint_key
            }
            
            # Invoke Lambda function asynchronously to continue processing
            lambda_client = boto3.client('lambda')
            lambda_client.invoke(
                FunctionName=context.function_name,
                InvocationType='Event',  # Asynchronous invocation
                Payload=json.dumps(next_event)
            )
            
            return {
                'statusCode': 202,
                'body': json.dumps({
                    'message': 'Processing continued in new invocation',
                    'checkpoint_key': checkpoint_key,
                    'documents_processed_so_far': result['checkpoint']['processed_count'],
                    'chunks_processed_so_far': result['checkpoint']['chunk_count']
                })
            }
        else:
            # Mark checkpoint as complete
            result['checkpoint']['is_complete'] = True
            final_checkpoint_key = save_checkpoint(bucket, result['checkpoint'])
            
            logger.info(f"Processing complete! Final checkpoint: {final_checkpoint_key}")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Processing complete',
                    'final_checkpoint': final_checkpoint_key,
                    'documents_processed': result['checkpoint']['processed_count'],
                    'chunks_processed': result['checkpoint']['chunk_count']
                })
            }
            
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'message': f'Error processing documents: {str(e)}'
            })
        }


def load_checkpoint(bucket, checkpoint_key):
    """Load checkpoint from S3 or initialize a new one"""
    if checkpoint_key:
        try:
            response = s3.get_object(Bucket=bucket, Key=checkpoint_key)
            checkpoint = json.loads(response['Body'].read().decode('utf-8'))
            logger.debug(f"Loaded checkpoint: {checkpoint}")
            
            # Check if checkpoint is marked as complete
            if checkpoint.get('is_complete', False):
                logger.info("Loaded checkpoint is marked as complete. No further processing needed.")
                # Return checkpoint with is_complete flag
                return checkpoint
                
            return checkpoint
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
    
    # Initialize new checkpoint
    return {
        'file_list': [],
        'file_contents': {},
        'document_structure': {},
        'structured_content': [],
        'processed_files': [],
        'processed_count': 0,
        'chunk_count': 0,
        'stage': 'collecting_files',
        'is_complete': False
    }


def save_checkpoint(bucket, checkpoint):
    """Save checkpoint to S3"""
    checkpoint_key = f"checkpoints/document_processor_{int(time.time())}.json"
    
    # Create a lightweight checkpoint by removing large content
    lightweight_checkpoint = {
        'file_list': checkpoint['file_list'],
        'processed_files': checkpoint['processed_files'],
        'processed_items': checkpoint.get('processed_items', []),  # Add this line
        'processed_count': checkpoint['processed_count'],
        'chunk_count': checkpoint['chunk_count'],
        'stage': checkpoint['stage'],
        'is_complete': checkpoint.get('is_complete', False)  # Add this line
    }
    
    # If we're in the processing stage, include document structure
    if checkpoint['stage'] == 'processing_structure':
        lightweight_checkpoint['structured_content'] = checkpoint['structured_content']
    
    # Save to S3
    s3.put_object(
        Bucket=bucket,
        Key=checkpoint_key,
        Body=json.dumps(lightweight_checkpoint),
        ContentType='application/json'
    )
    
    logger.info(f"Saved checkpoint to {checkpoint_key}")
    return checkpoint_key
