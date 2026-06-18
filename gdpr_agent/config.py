"""Configuration and client initialization"""
from databricks.vector_search.client import VectorSearchClient
from openai import OpenAI

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
openai_client = None
vsc = None

def setup(openai_api_key: str):
    """Initialize global clients - called from notebook with API key"""
    global openai_client, vsc
    openai_client = OpenAI(api_key=openai_api_key)
    vsc = VectorSearchClient(disable_notice=True)
    print("✓ Configuration loaded")
    print(f"✓ Vector Search Endpoint: {VS_ENDPOINT_NAME}")
    print(f"✓ Policies Index: {GDPR_POLICIES_INDEX}")
    print(f"✓ Case Law Index: {CASE_LAW_INDEX}")
    print(f"✓ Procedures Index: {PROCEDURES_INDEX}")
    print("✓ Clients initialized")