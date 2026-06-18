"""
Sync vector search indices for compliance documents.
Triggers sync operations to update vector indices with latest embeddings.
"""

import argparse
from typing import List, Optional
from databricks.vector_search.client import VectorSearchClient

from config import SOURCES, VECTOR_ENDPOINT
from utils.spark_helpers import get_or_create_spark, table_exists


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
    print(f"🔄 Syncing vector index: {index_name}")
    print(f"   Source: {source_table}")
    
    # Verify source table exists
    spark = get_or_create_spark()
    if not table_exists(source_table, spark):
        raise ValueError(f"Source table {source_table} does not exist")
    
    # Initialize Vector Search client
    vsc = VectorSearchClient()
    
    try:
        # Get the index
        index = vsc.get_index(endpoint_name=endpoint_name, index_name=index_name)
        
        # Trigger sync
        index.sync()
        
        print(f"✅ Vector index synced successfully: {index_name}\n")
        
    except Exception as e:
        print(f"❌ Failed to sync {index_name}: {e}\n")
        raise


def sync_gdpr_index() -> str:
    """
    Sync GDPR statutory legislation vector index.
    
    Returns:
        Index name
    """
    print("📚 Syncing GDPR Vector Index...")
    
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
    print("📚 Syncing Corporate Policy Vector Index...")
    
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
    print("📚 Syncing Enforcement Tracker Vector Index...")
    
    source_config = SOURCES["enforcement"]
    index_name = source_config["vector_index"]
    embeddings_table = source_config["embeddings_table"]
    
    sync_vector_index(
        index_name=index_name,
        source_table=embeddings_table
    )
    
    return index_name


def sync_all_indices(sources: Optional[List[str]] = None) -> dict:
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
    
    print("=" * 70)
    print("🚀 STARTING VECTOR INDEX SYNC")
    print("=" * 70 + "\n")
    print(f"Vector Search Endpoint: {VECTOR_ENDPOINT}\n")
    
    # Sync each index
    if "gdpr" in sources:
        try:
            results["gdpr"] = sync_gdpr_index()
        except Exception as e:
            print(f"❌ Failed to sync GDPR index: {e}\n")
    
    if "policy" in sources:
        try:
            results["policy"] = sync_policy_index()
        except Exception as e:
            print(f"❌ Failed to sync policy index: {e}\n")
    
    if "enforcement" in sources:
        try:
            results["enforcement"] = sync_enforcement_index()
        except Exception as e:
            print(f"❌ Failed to sync enforcement index: {e}\n")
    
    print("=" * 70)
    print(f"🎉 INDEX SYNC COMPLETE - {len(results)}/{len(sources)} indices successful")
    print("=" * 70)
    
    return results


def check_index_status(index_name: str, endpoint_name: str = VECTOR_ENDPOINT) -> dict:
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
        return {
            "name": index_name,
            "status": "ERROR",
            "error": str(e)
        }


def check_all_indices_status(sources: Optional[List[str]] = None) -> dict:
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
    
    print("=" * 70)
    print("📊 VECTOR INDEX STATUS CHECK")
    print("=" * 70 + "\n")
    
    for source in sources:
        if source in SOURCES:
            index_name = SOURCES[source]["vector_index"]
            status = check_index_status(index_name)
            statuses[source] = status
            
            print(f"{source.upper()} Index: {index_name}")
            print(f"   Status: {status.get('status', 'UNKNOWN')}")
            print(f"   Ready: {status.get('ready', False)}")
            print(f"   Indexed Rows: {status.get('indexed_rows', 0)}")
            if "error" in status:
                print(f"   Error: {status['error']}")
            print()
    
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