"""
One-time setup for Vector Search indices.
Only run this when creating indices for the first time or after deletion.
"""

from databricks.vector_search.client import VectorSearchClient
from config import SOURCES, VECTOR_ENDPOINT
from utils.spark_helpers import get_or_create_spark


def create_vector_endpoint(endpoint_name: str = VECTOR_ENDPOINT):
    """Create Vector Search endpoint if it doesn't exist."""
    vsc = VectorSearchClient()
    
    try:
        endpoint = vsc.get_endpoint(endpoint_name)
        print(f"✓ Endpoint already exists: {endpoint_name}")
    except:
        print(f"Creating endpoint: {endpoint_name}")
        vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
        print(f"✅ Endpoint created: {endpoint_name}")


def enable_change_data_feed():
    """Enable Change Data Feed on all embeddings tables."""
    spark = get_or_create_spark()
    
    embeddings_tables = [
        SOURCES["gdpr"]["embeddings_table"],
        SOURCES["policy"]["embeddings_table"],
        SOURCES["enforcement"]["embeddings_table"]
    ]
    
    for table in embeddings_tables:
        spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
        print(f"✅ CDF enabled: {table}")


def create_vector_index(
    index_name: str,
    source_table: str,
    embedding_col: str,
    text_col: str,
    primary_key: str,
    endpoint_name: str = VECTOR_ENDPOINT
):
    """Create a vector search index."""
    vsc = VectorSearchClient()
    
    # Check if exists
    try:
        existing = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        print(f"⚠️  Index already exists: {index_name}")
        return
    except:
        pass
    
    print(f"Creating index: {index_name}")
    
    # Create delta sync index with hybrid search
    vsc.create_delta_sync_index(
        endpoint_name=endpoint_name,
        index_name=index_name,
        source_table_name=source_table,
        pipeline_type="TRIGGERED",
        primary_key=primary_key,
        embedding_dimension=1024,  # databricks-gte-large-en
        embedding_vector_column=embedding_col,
        columns_to_sync=[text_col, primary_key]
    )
    
    print(f"✅ Index created: {index_name}")


def setup_all_indices():
    """Full setup: endpoint, CDF, and all indices."""
    print("=" * 70)
    print("🚀 VECTOR SEARCH INFRASTRUCTURE SETUP")
    print("=" * 70 + "\n")
    
    # Step 1: Endpoint
    create_vector_endpoint()
    
    # Step 2: Enable CDF
    print("\n📊 Enabling Change Data Feed...")
    enable_change_data_feed()
    
    # Step 3: Create indices
    print("\n🔨 Creating Vector Search Indices...")
    
    # GDPR legislation index
    create_vector_index(
        index_name=SOURCES["gdpr"]["vector_index"],
        source_table=SOURCES["gdpr"]["embeddings_table"],
        embedding_col="embedding",
        text_col="text_content",
        primary_key="chunk_id"
    )
    
    # Corporate policy index
    create_vector_index(
        index_name=SOURCES["policy"]["vector_index"],
        source_table=SOURCES["policy"]["embeddings_table"],
        embedding_col="embedding",
        text_col="text_content",
        primary_key="chunk_id"
    )
    
    # Enforcement tracker index
    create_vector_index(
        index_name=SOURCES["enforcement"]["vector_index"],
        source_table=SOURCES["enforcement"]["embeddings_table"],
        embedding_col="embedding",
        text_col="full_document_text_translated",
        primary_key="source_file_name"
    )
    
    print("\n" + "=" * 70)
    print("✅ SETUP COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    setup_all_indices()