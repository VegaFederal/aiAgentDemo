// API client for the LLM Agent

// Base URL for the API
const API_BASE_URL = 'https://your-api-gateway-url.execute-api.us-east-1.amazonaws.com/dev/api';

// Status elements
const statusElement = document.getElementById('status');
const resultsElement = document.getElementById('results');

/**
 * Process a query with streaming results
 * @param {string} query - The user's query
 * @param {string} questionType - Optional question type (multiple_choice, yes_no, true_false)
 */
async function processQueryWithStreaming(query, questionType = null) {
  try {
    // Clear previous results
    resultsElement.innerHTML = '';
    statusElement.textContent = 'Processing...';
    
    // Initial request parameters
    let requestParams = {
      query: query,
      question_type: questionType
    };
    
    let batchNumber = 1;
    let totalMatches = 0;
    let embedding = null;
    let accumulatedResults = [];
    
    // Process batches until complete
    while (true) {
      // If we have an embedding from a previous batch, use it
      if (embedding) {
        requestParams.embedding = embedding;
      }
      
      // If we have accumulated results, include them
      if (accumulatedResults.length > 0) {
        requestParams.accumulated_results = accumulatedResults;
      }
      
      // Update status
      statusElement.textContent = `Processing batch ${batchNumber}...`;
      
      // Make API request
      const response = await fetch(API_BASE_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestParams)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(`API error: ${errorData.error || response.statusText}`);
      }
      
      // Process batch results
      const data = await response.json();
      
      // Save embedding for subsequent requests
      if (data.embedding) {
        embedding = data.embedding;
      }
      
      // Save accumulated results for next request
      if (data.accumulated_results) {
        accumulatedResults = data.accumulated_results;
      }
      
      // Display batch results
      const batchResults = data.batch_results || [];
      totalMatches = batchResults.length;
      
      // Update the UI with batch results
      updateBatchResults(batchResults, batchNumber, data.items_processed || 0);
      
      // If this is the final batch, show the LLM response
      if (data.is_final) {
        if (data.llm_response) {
          displayFinalResponse(data.llm_response);
        }
        statusElement.textContent = `Complete! Found ${totalMatches} relevant matches across ${batchNumber} batches.`;
        break;
      }
      
      // Prepare for next batch
      requestParams.last_key = data.next_key;
      batchNumber++;
    }
  } catch (error) {
    console.error('Error processing query:', error);
    statusElement.textContent = `Error: ${error.message}`;
  }
}

/**
 * Update the UI with batch results
 * @param {Array} batchResults - Array of matching documents
 * @param {number} batchNumber - Current batch number
 * @param {number} itemsProcessed - Number of items processed in this batch
 */
function updateBatchResults(batchResults, batchNumber, itemsProcessed) {
  if (batchResults.length === 0) {
    return;
  }
  
  const batchElement = document.createElement('div');
  batchElement.className = 'batch-results';
  batchElement.innerHTML = `<h3>Batch ${batchNumber} Results (${batchResults.length} matches from ${itemsProcessed} items)</h3>`;
  
  const matchesList = document.createElement('ul');
  batchResults.forEach(match => {
    const matchItem = document.createElement('li');
    matchItem.innerHTML = `
      <strong>Document ID:</strong> ${match.document_id}<br>
      <strong>Similarity:</strong> ${(match.similarity * 100).toFixed(2)}%<br>
      <div class="match-content">${match.content.substring(0, 200)}${match.content.length > 200 ? '...' : ''}</div>
    `;
    matchesList.appendChild(matchItem);
  });
  
  batchElement.appendChild(matchesList);
  resultsElement.appendChild(batchElement);
  
  // Scroll to the bottom to show latest results
  window.scrollTo(0, document.body.scrollHeight);
}

/**
 * Display the final LLM response
 * @param {string} response - The LLM's response text
 */
function displayFinalResponse(response) {
  const responseElement = document.createElement('div');
  responseElement.className = 'llm-response';
  responseElement.innerHTML = `
    <h2>Final Answer</h2>
    <div class="response-content">${response}</div>
  `;
  resultsElement.appendChild(responseElement);
}