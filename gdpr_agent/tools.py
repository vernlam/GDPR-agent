from . import config
# ============================================================================
# PRODUCTION TOOL 1: Historical Fines
# ============================================================================
def tool_search_historical_fines(query_text: str, query_vector: list = None, top_k: int = 3):
    """
    Searches historical data on GDPR breaches, legal precedents, enforcement actions, 
    and monetary fine amounts levied against retail corporations.
    """
    # 1. Only generate embedding if not provided
    if query_vector is None:
        query_vector = config.openai_client.embeddings.create(  
            input=[query_text], model="text-embedding-3-small"
        ).data[0].embedding

    # 2. Query the specific index natively
    index = config.vsc.get_index(endpoint_name=config.VS_ENDPOINT_NAME, index_name=config.CASE_LAW_INDEX)  
    results = index.similarity_search(
        query_vector=query_vector,
        query_text=query_text,
        columns=["source_file_name", "translated_text"],
        num_results=top_k,
        query_type="HYBRID"
    )
    return results

# ============================================================================
# PRODUCTION TOOL 2: GDPR Legislation
# ============================================================================
def tool_search_gdpr_legislation(query_text: str, query_vector: list = None, top_k: int = 3):
    """
    Searches official legal articles, statutory text, and clauses of the GDPR 
    (specifically Chapter 3 regarding data subject rights).
    """
    if query_vector is None:
        query_vector = config.openai_client.embeddings.create( 
            input=[query_text], model="text-embedding-3-small"
        ).data[0].embedding
        
    index = config.vsc.get_index(endpoint_name=config.VS_ENDPOINT_NAME, index_name=config.GDPR_POLICIES_INDEX)  
    
    return index.similarity_search(
        query_vector=query_vector,
        columns=["article_title", "text_content"],
        num_results=top_k
    )

# ============================================================================
# PRODUCTION TOOL 3: Internal Corporate Policy
# ============================================================================
def tool_search_retail_policy(query_text: str, query_vector: list = None, top_k: int = 3):
    """
    Searches internal company guidelines, retail store operational privacy policies, 
    customer data consent scripts, and standard operating procedures.
    """
    if query_vector is None:
        query_vector = config.openai_client.embeddings.create(  
            input=[query_text], model="text-embedding-3-small"
        ).data[0].embedding
        
    index = config.vsc.get_index(endpoint_name=config.VS_ENDPOINT_NAME, index_name=config.PROCEDURES_INDEX)  
    
    return index.similarity_search(
        query_vector=query_vector,
        columns=["section_title", "text_content"],
        num_results=top_k
    )

