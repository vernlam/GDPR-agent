"""
Sync vector search indices for compliance documents.
Triggers sync operations to update vector indices with latest embeddings.
"""

import argparse
import logging
from typing import List, Optional, Dict, Any
from databricks.vector_search.client import VectorSearchClient

from config import SOURCES, VECTOR_ENDPOINT
from utils.spark_helpers import get_or_create_spark, table_exists

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def sync_vector_index(
    index_name: str,
    source_table: str,
    endpoint_name: str = VECTOR_ENDPOINT
) -> None:
    """
    Sync a vector search index with its source embeddings table.
    
    Args:
        index_name: Fully qualified vector index name
        source_table: Source embeddings table
        endpoint_name: Vector search endpoint name
    """
    logger.info("Initiating vector index sync: %s", index_name)
    logger.info("Source embeddings table: %s", source_table)
    
    # Verify source table exists
    spark = get_or_create_spark()
    if not table_exists(source_table, spark):
        error_msg = "Source embeddings table does not exist: %s"
        logger.error(error_msg, source_table)
        raise ValueError(f"Source table {source_table} does not exist")
    
    # Initialize Vector Search client
    vsc = VectorSearchClient()
    
    try:
        # Get the index
        index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        
        # Trigger sync
        index.sync()
        
        logger.info("Vector index synced successfully: %s", index_name)
        
    except Exception as e:
        logger.exception("Failed to sync vector index %s: %s", index_name, e)
        raise


def sync_gdpr_index() -> str:
    """
    Sync GDPR statutory legislation vector index.
    
    Returns:
        Index name
    """
    logger.info("Syncing GDPR statutory legislation vector index")
    
    source_config = SOURCES["gdpr"]
    index_name = source_config["vector_index"]
    embeddings_table = source_config["embeddings_table"]
    
    sync_vector_index(
        index_name=index_name,
        source_table=embeddings_table
    )
    
    return index_name


def sync_policy_index() -> str:
    """
    Sync corporate privacy policy vector index.
    
    Returns:
        Index name
    """
    logger.info("Syncing corporate privacy policy vector index")
    
    source_config = SOURCES["policy"]
    index_name = source_config["vector_index"]
    embeddings_table = source_config["embeddings_table"]
    
    sync_vector_index(
        index_name=index_name,
        source_table=embeddings_table
    )
    
    return index_name


def sync_enforcement_index() -> str:
    """
    Sync enforcement tracker vector index.
    
    Returns:
        Index name
    """
    logger.info("Syncing enforcement tracker vector index")
    
    source_config = SOURCES["enforcement"]
    index_name = source_config["vector_index"]
    embeddings_table = source_config["embeddings_table"]
    
    sync_vector_index(
        index_name=index_name,
        source_table=embeddings_table
    )
    
    return index_name


def sync_all_indices(sources: Optional[List[str]] = None) -> Dict[str, str]:
    """
    Sync all or specified vector search indices.
    
    Args:
        sources: List of source names to sync (gdpr, policy, enforcement)
                 If None, syncs all indices
                 
    Returns:
        Dictionary mapping source name to index name
    """
    # Default to all sources
    if sources is None:
        sources = ["gdpr", "policy", "enforcement"]
    
    results = {}
    
    logger.info("=" * 70)
    logger.info("STARTING VECTOR INDEX SYNC")
    logger.info("=" * 70)
    logger.info("Vector Search Endpoint: %s", VECTOR_ENDPOINT)
    logger.info("Sources to sync: %s", ', '.join(sources))
    
    # Sync each index
    if "gdpr" in sources:
        try:
            results["gdpr"] = sync_gdpr_index()
        except Exception as e:
            logger.exception("Failed to sync GDPR vector index: %s", e)
    
    if "policy" in sources:
        try:
            results["policy"] = sync_policy_index()
        except Exception as e:
            logger.exception("Failed to sync corporate policy vector index: %s", e)
    
    if "enforcement" in sources:
        try:
            results["enforcement"] = sync_enforcement_index()
        except Exception as e:
            logger.exception("Failed to sync enforcement tracker vector index: %s", e)
    
    logger.info("=" * 70)
    logger.info("VECTOR INDEX SYNC COMPLETE - %d/%d indices successful", len(results), len(sources))
    logger.info("=" * 70)
    
    return results


def check_index_status(index_name: str, endpoint_name: str = VECTOR_ENDPOINT) -> Dict[str, Any]:
    """
    Check the status of a vector search index.
    
    Args:
        index_name: Fully qualified index name
        endpoint_name: Vector search endpoint name
        
    Returns:
        Dictionary with index status information
    """
    vsc = VectorSearchClient()
    
    try:
        index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        
        # Get index description/status
        status = index.describe()
        
        return {
            "name": index_name,
            "status": status.get("status", {}).get("state", "UNKNOWN"),
            "indexed_rows": status.get("status", {}).get("indexed_row_count", 0),
            "ready": status.get("status", {}).get("ready", False)
        }
        
    except Exception as e:
        logger.exception("Failed to retrieve status for vector index %s: %s", index_name, e)
        return {
            "name": index_name,
            "status": "ERROR",
            "error": str(e)
        }


def check_all_indices_status(sources: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """
    Check status of all vector search indices.
    
    Args:
        sources: List of source names to check (gdpr, policy, enforcement)
                 If None, checks all indices
                 
    Returns:
        Dictionary mapping source name to status info
    """
    if sources is None:
        sources = ["gdpr", "policy", "enforcement"]
    
    statuses = {}
    
    logger.info("=" * 70)
    logger.info("VECTOR INDEX STATUS CHECK")
    logger.info("=" * 70)
    
    for source in sources:
        if source in SOURCES:
            index_name = SOURCES[source]["vector_index"]
            status = check_index_status(index_name)
            statuses[source] = status
            
            logger.info("%s Index: %s", source.upper(), index_name)
            logger.info("  Status: %s", status.get('status', 'UNKNOWN'))
            logger.info("  Ready: %s", status.get('ready', False))
            logger.info("  Indexed Rows: %d", status.get('indexed_rows', 0))
            if "error" in status:
                logger.error("  Error: %s", status['error'])
    
    logger.info("=" * 70)
    logger.info("Status check complete for %d indices", len(statuses))
    logger.info("=" * 70)
    
    return statuses


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync vector search indices")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["gdpr", "policy", "enforcement"],
        help="Specific indices to sync (default: all)"
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Check index status without syncing"
    )
    
    args = parser.parse_args()
    
    if args.status_only:
        check_all_indices_status(sources=args.sources)
    else:
        sync_all_indices(sources=args.sources)
