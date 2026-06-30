"""
Vector search tool implementations for GDPR Agent.
Provides search functionality across historical fines, GDPR legislation, and internal policies.
"""

import logging
from typing import Optional, Dict, Any, List
from . import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PRODUCTION TOOL 1: Historical Fines
# ============================================================================
def tool_search_historical_fines(
    query_text: str, 
    query_vector: Optional[List[float]] = None, 
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Search historical GDPR enforcement data and fines precedents.
    
    Searches historical data on GDPR breaches, legal precedents, enforcement actions,
    and monetary fine amounts levied against retail corporations.
    
    Args:
        query_text: The search query text describing compliance concerns or enforcement topics
        query_vector: Optional pre-computed embedding vector. If None, will be generated from query_text
        top_k: Maximum number of results to return (default: 3)
        
    Returns:
        Dictionary containing search results with columns: source_file_name, translated_text
        
    Raises:
        Exception: If OpenAI embedding generation fails (re-raised after logging)
        Exception: If vector search index query fails (re-raised after logging)
    """
    logger.info("Searching historical fines index")
    logger.debug("Query text: %s", query_text[:100] + "..." if len(query_text) > 100 else query_text)
    logger.debug("Top K: %d", top_k)
    
    # Generate embedding if not provided
    if query_vector is None:
        try:
            logger.debug("Generating embedding for query text")
            query_vector = config.openai_client.embeddings.create(  
                input=[query_text], model="text-embedding-3-small"
            ).data[0].embedding
            logger.debug("Embedding generated successfully (%d dimensions)", len(query_vector))
        except Exception as e:
            logger.exception("Failed to generate embedding for historical fines search: %s", e)
            raise

    # Query the vector search index
    try:
        logger.debug("Querying historical fines index: %s", config.CASE_LAW_INDEX)
        index = config.vsc.get_index(
            endpoint_name=config.VS_ENDPOINT_NAME, 
            index_name=config.CASE_LAW_INDEX
        )
        results = index.similarity_search(
            query_vector=query_vector,
            query_text=query_text,
            columns=["source_file_name", "translated_text"],
            num_results=top_k,
            query_type="HYBRID"
        )
        logger.info("Historical fines search completed successfully")
        logger.debug("Retrieved %d results", len(results.get('result', {}).get('data_array', [])))
        return results
    except Exception as e:
        logger.exception("Failed to query historical fines index: %s", e)
        raise


# ============================================================================
# PRODUCTION TOOL 2: GDPR Legislation
# ============================================================================
def tool_search_gdpr_legislation(
    query_text: str, 
    query_vector: Optional[List[float]] = None, 
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Search official GDPR legislation and statutory text.
    
    Searches official legal articles, statutory text, and clauses of the GDPR
    (specifically Chapter 3 regarding data subject rights).
    
    Args:
        query_text: The search query text describing legal requirements or articles
        query_vector: Optional pre-computed embedding vector. If None, will be generated from query_text
        top_k: Maximum number of results to return (default: 3)
        
    Returns:
        Dictionary containing search results with columns: article_title, text_content
        
    Raises:
        Exception: If OpenAI embedding generation fails (re-raised after logging)
        Exception: If vector search index query fails (re-raised after logging)
    """
    logger.info("Searching GDPR legislation index")
    logger.debug("Query text: %s", query_text[:100] + "..." if len(query_text) > 100 else query_text)
    logger.debug("Top K: %d", top_k)
    
    # Generate embedding if not provided
    if query_vector is None:
        try:
            logger.debug("Generating embedding for query text")
            query_vector = config.openai_client.embeddings.create( 
                input=[query_text], model="text-embedding-3-small"
            ).data[0].embedding
            logger.debug("Embedding generated successfully (%d dimensions)", len(query_vector))
        except Exception as e:
            logger.exception("Failed to generate embedding for GDPR legislation search: %s", e)
            raise
    
    # Query the vector search index
    try:
        logger.debug("Querying GDPR legislation index: %s", config.GDPR_POLICIES_INDEX)
        index = config.vsc.get_index(
            endpoint_name=config.VS_ENDPOINT_NAME, 
            index_name=config.GDPR_POLICIES_INDEX
        )
        results = index.similarity_search(
            query_vector=query_vector,
            columns=["article_title", "text_content"],
            num_results=top_k
        )
        logger.info("GDPR legislation search completed successfully")
        logger.debug("Retrieved %d results", len(results.get('result', {}).get('data_array', [])))
        return results
    except Exception as e:
        logger.exception("Failed to query GDPR legislation index: %s", e)
        raise


# ============================================================================
# PRODUCTION TOOL 3: Internal Corporate Policy
# ============================================================================
def tool_search_retail_policy(
    query_text: str, 
    query_vector: Optional[List[float]] = None, 
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Search internal company policies and operational procedures.
    
    Searches internal company guidelines, retail store operational privacy policies,
    customer data consent scripts, and standard operating procedures.
    
    Args:
        query_text: The search query text describing policy topics or procedures
        query_vector: Optional pre-computed embedding vector. If None, will be generated from query_text
        top_k: Maximum number of results to return (default: 3)
        
    Returns:
        Dictionary containing search results with columns: section_title, text_content
        
    Raises:
        Exception: If OpenAI embedding generation fails (re-raised after logging)
        Exception: If vector search index query fails (re-raised after logging)
    """
    logger.info("Searching internal retail policy index")
    logger.debug("Query text: %s", query_text[:100] + "..." if len(query_text) > 100 else query_text)
    logger.debug("Top K: %d", top_k)
    
    # Generate embedding if not provided
    if query_vector is None:
        try:
            logger.debug("Generating embedding for query text")
            query_vector = config.openai_client.embeddings.create(  
                input=[query_text], model="text-embedding-3-small"
            ).data[0].embedding
            logger.debug("Embedding generated successfully (%d dimensions)", len(query_vector))
        except Exception as e:
            logger.exception("Failed to generate embedding for retail policy search: %s", e)
            raise
    
    # Query the vector search index
    try:
        logger.debug("Querying retail policy index: %s", config.PROCEDURES_INDEX)
        index = config.vsc.get_index(
            endpoint_name=config.VS_ENDPOINT_NAME, 
            index_name=config.PROCEDURES_INDEX
        )
        results = index.similarity_search(
            query_vector=query_vector,
            columns=["section_title", "text_content"],
            num_results=top_k
        )
        logger.info("Retail policy search completed successfully")
        logger.debug("Retrieved %d results", len(results.get('result', {}).get('data_array', [])))
        return results
    except Exception as e:
        logger.exception("Failed to query retail policy index: %s", e)
        raise
