"""
One-time setup for Vector Search indices.
Only run this when creating indices for the first time or after deletion.
"""

import logging
from typing import Optional, List
from databricks.vector_search.client import VectorSearchClient
from config import SOURCES, VECTOR_ENDPOINT
from utils.spark_helpers import get_or_create_spark

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def create_vector_endpoint(endpoint_name: str = VECTOR_ENDPOINT) -> None:
    """
    Create Vector Search endpoint if it doesn't exist.
    
    Args:
        endpoint_name: Name of the vector search endpoint to create
    """
    vsc = VectorSearchClient()
    
    try:
        endpoint = vsc.get_endpoint(endpoint_name)
        logger.info("Vector search endpoint already exists: %s", endpoint_name)
    except Exception:
        logger.info("Creating vector search endpoint: %s", endpoint_name)
        try:
            vsc.create_endpoint(name=endpoint_name, endpoint_type="STANDARD")
            logger.info("Successfully created vector search endpoint: %s", endpoint_name)
        except Exception as e:
            logger.exception("Failed to create vector search endpoint %s: %s", endpoint_name, e)
            raise


def enable_change_data_feed() -> None:
    """
    Enable Change Data Feed on all embeddings tables.
    Required for delta sync vector search indices.
    """
    spark = get_or_create_spark()
    
    embeddings_tables = [
        SOURCES["gdpr"]["embeddings_table"],
        SOURCES["policy"]["embeddings_table"],
        SOURCES["enforcement"]["embeddings_table"]
    ]
    
    logger.info("Enabling Change Data Feed on %d embeddings tables", len(embeddings_tables))
    
    for table in embeddings_tables:
        try:
            spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
            logger.info("Change Data Feed enabled for table: %s", table)
        except Exception as e:
            logger.exception("Failed to enable Change Data Feed for table %s: %s", table, e)
            raise


def create_vector_index(
    index_name: str,
    source_table: str,
    embedding_col: str,
    text_col: str,
    primary_key: str,
    endpoint_name: str = VECTOR_ENDPOINT
) -> None:
    """
    Create a vector search index.
    
    Args:
        index_name: Fully qualified name for the vector search index
        source_table: Source Delta table with embeddings
        embedding_col: Column name containing embedding vectors
        text_col: Column name containing text content
        primary_key: Primary key column name
        endpoint_name: Vector search endpoint name
    """
    vsc = VectorSearchClient()
    
    # Check if index already exists
    try:
        existing = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        logger.warning("Vector search index already exists: %s", index_name)
        return
    except Exception:
        pass
    
    logger.info("Creating vector search index: %s", index_name)
    logger.info("Source table: %s, Primary key: %s, Embedding column: %s", 
                source_table, primary_key, embedding_col)
    
    try:
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
        logger.info("Successfully created vector search index: %s", index_name)
    except Exception as e:
        logger.exception("Failed to create vector search index %s: %s", index_name, e)
        raise


def setup_all_indices() -> None:
    """
    Full setup: endpoint, CDF, and all vector search indices.
    
    Creates:
    - Vector search endpoint
    - Enables Change Data Feed on embeddings tables
    - Creates indices for GDPR legislation, corporate policy, and enforcement tracker
    """
    logger.info("=" * 70)
    logger.info("VECTOR SEARCH INFRASTRUCTURE SETUP")
    logger.info("=" * 70)
    
    # Step 1: Create endpoint
    logger.info("Step 1: Creating vector search endpoint")
    create_vector_endpoint()
    
    # Step 2: Enable Change Data Feed
    logger.info("Step 2: Enabling Change Data Feed on embeddings tables")
    enable_change_data_feed()
    
    # Step 3: Create vector search indices
    logger.info("Step 3: Creating vector search indices")
    
    # GDPR legislation index
    logger.info("Creating GDPR statutory legislation vector index")
    create_vector_index(
        index_name=SOURCES["gdpr"]["vector_index"],
        source_table=SOURCES["gdpr"]["embeddings_table"],
        embedding_col="embedding",
        text_col="text_content",
        primary_key="chunk_id"
    )
    
    # Corporate policy index
    logger.info("Creating corporate policy vector index")
    create_vector_index(
        index_name=SOURCES["policy"]["vector_index"],
        source_table=SOURCES["policy"]["embeddings_table"],
        embedding_col="embedding",
        text_col="text_content",
        primary_key="chunk_id"
    )
    
    # Enforcement tracker index
    logger.info("Creating enforcement tracker vector index")
    create_vector_index(
        index_name=SOURCES["enforcement"]["vector_index"],
        source_table=SOURCES["enforcement"]["embeddings_table"],
        embedding_col="embedding",
        text_col="full_document_text_translated",
        primary_key="source_file_name"
    )
    
    logger.info("=" * 70)
    logger.info("VECTOR SEARCH SETUP COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    setup_all_indices()
