"""
Configuration and client initialization for GDPR Agent.
Manages Vector Search and OpenAI client setup with global state.
"""

import logging
from typing import Optional
from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

# Vector Search
VS_ENDPOINT_NAME = "gdpr_rag_endpoint"

# Index paths (corrected to actual indexes in main.default)
GDPR_POLICIES_INDEX = "main.default.gdpr_law_vector_index"
CASE_LAW_INDEX = "main.default.gdpr_fines_vector_index"
PROCEDURES_INDEX = "main.default.privacy_policy_vector_index"

# Search parameters
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.7

# LLM parameters
TEMPERATURE = 0.1
MAX_TOKENS = 2000

# ============================================================
# CLIENT INITIALIZATION
# ============================================================

# Global clients (will be initialized by setup function)
openai_client: Optional[OpenAI] = None
vsc: Optional[VectorSearchClient] = None


def setup(openai_api_key: str) -> None:
    """
    Initialize global clients - called from notebook with API key.
    
    Args:
        openai_api_key: OpenAI API key for client authentication
        
    Raises:
        ValueError: If openai_api_key is empty or None
        Exception: If client initialization fails
    """
    global openai_client, vsc
    
    if not openai_api_key:
        error_msg = "OpenAI API key cannot be empty or None"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info("Initializing GDPR Agent configuration")
    
    try:
        # Initialize OpenAI client
        logger.debug("Creating OpenAI client")
        openai_client = OpenAI(api_key=openai_api_key)
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize OpenAI client: %s", e)
        raise
    
    try:
        # Initialize Vector Search client
        logger.debug("Creating Vector Search client")
        vsc = VectorSearchClient(disable_notice=True)
        logger.info("Vector Search client initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize Vector Search client: %s", e)
        raise
    
    # Log configuration summary
    logger.info("Configuration loaded successfully")
    logger.info("Vector Search Endpoint: %s", VS_ENDPOINT_NAME)
    logger.info("Policies Index: %s", GDPR_POLICIES_INDEX)
    logger.info("Case Law Index: %s", CASE_LAW_INDEX)
    logger.info("Procedures Index: %s", PROCEDURES_INDEX)
    logger.debug("LLM parameters - Temperature: %s, Max Tokens: %d", TEMPERATURE, MAX_TOKENS)
    logger.debug("Search parameters - Top K: %d, Similarity Threshold: %s", TOP_K_RESULTS, SIMILARITY_THRESHOLD)


def is_configured() -> bool:
    """
    Check if clients have been initialized.
    
    Returns:
        True if both clients are initialized, False otherwise
    """
    configured = openai_client is not None and vsc is not None
    logger.debug("Configuration check: %s", "configured" if configured else "not configured")
    return configured
