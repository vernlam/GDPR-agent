"""
Configuration for GDPR compliance document ingestion pipeline.
Single source of truth for all catalog, schema, volume, and table references.
"""

# Unity Catalog Configuration
CATALOG = "main"
SCHEMA = "default"

# Volume paths
COMPLIANCE_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/compliance_sources"
ENFORCEMENT_VOLUME = f"/Volumes/{CATALOG}/{SCHEMA}/enforcement_tracker"

# Vector Search Endpoint
VECTOR_ENDPOINT = "gdpr_rag_endpoint"

# Source configurations
SOURCES = {
    "gdpr": {
        "file": f"{COMPLIANCE_VOLUME}/GDPR_Chapter_3.md",
        "table": f"{CATALOG}.{SCHEMA}.gdpr_statutory_legislation",
        "embeddings_table": f"{CATALOG}.{SCHEMA}.gdpr_statutory_legislation_embeddings",
        "vector_index": f"{CATALOG}.{SCHEMA}.gdpr_law_vector_index"
    },
    
    "policy": {
        "file": f"{COMPLIANCE_VOLUME}/retail_privacy_policy.md",
        "table": f"{CATALOG}.{SCHEMA}.retail_corporate_policy",
        "embeddings_table": f"{CATALOG}.{SCHEMA}.retail_corporate_policy_embeddings",
        "vector_index": f"{CATALOG}.{SCHEMA}.privacy_policy_vector_index"
    },
    
    "enforcement": {
        "volume_path": ENFORCEMENT_VOLUME,
        "raw_table": f"{CATALOG}.{SCHEMA}.gdpr_raw_multilingual_precedents",
        "translated_table": f"{CATALOG}.{SCHEMA}.gdpr_translated_precedents",
        "embeddings_table": f"{CATALOG}.{SCHEMA}.gdpr_precedents_embeddings",
        "vector_index": f"{CATALOG}.{SCHEMA}.enforcement_tracker_vector_index"
    }
}

# Embedding model configuration
EMBEDDING_MODEL = "databricks-gte-large-en"

# Translation configuration
TRANSLATION_TARGET_LANGUAGE = "English"
TRANSLATION_CHUNK_SIZE = 3000