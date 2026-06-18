"""
Document ingestion pipeline for GDPR compliance sources.
Handles GDPR legislation, corporate policy, and enforcement tracker documents.
"""

import argparse
from typing import List, Optional
from pyspark.sql import SparkSession

from config import SOURCES, CATALOG, SCHEMA
from utils.file_parser import parse_gdpr_markdown, parse_policy_markdown, parse_enforcement_pdfs
from utils.spark_helpers import (
    get_or_create_spark,
    list_to_dataframe,
    write_to_delta,
    table_exists,
    get_table_row_count,
    add_metadata_columns
)


def ingest_gdpr_legislation(spark: SparkSession) -> str:
    """
    Ingest GDPR Chapter 3 markdown file.
    
    Returns:
        Target table name
    """
    print("📖 Ingesting GDPR Statutory Legislation...")
    
    source_config = SOURCES["gdpr"]
    file_path = source_config["file"]
    table_name = source_config["table"]
    
    # Parse markdown file
    records = parse_gdpr_markdown(file_path, spark)
    print(f"   Parsed {len(records)} articles from markdown")
    
    # Convert to DataFrame
    df = list_to_dataframe(records, spark)
    
    # Add metadata columns
    df = add_metadata_columns(df)
    
    # Write to Delta table
    write_to_delta(
        df=df,
        table_name=table_name,
        mode="overwrite",
        overwrite_schema=True,
        enable_cdf=True
    )
    
    row_count = get_table_row_count(table_name, spark)
    print(f"✅ GDPR legislation ingested: {row_count} records → {table_name}\n")
    
    return table_name


def ingest_corporate_policy(spark: SparkSession) -> str:
    """
    Ingest retail privacy policy markdown file.
    
    Returns:
        Target table name
    """
    print("📖 Ingesting Corporate Privacy Policy...")
    
    source_config = SOURCES["policy"]
    file_path = source_config["file"]
    table_name = source_config["table"]
    
    # Parse markdown file
    records = parse_policy_markdown(file_path, spark)
    print(f"   Parsed {len(records)} sections from markdown")
    
    # Convert to DataFrame
    df = list_to_dataframe(records, spark)
    
    # Add metadata columns
    df = add_metadata_columns(df)
    
    # Write to Delta table
    write_to_delta(
        df=df,
        table_name=table_name,
        mode="overwrite",
        overwrite_schema=True,
        enable_cdf=True
    )
    
    row_count = get_table_row_count(table_name, spark)
    print(f"✅ Corporate policy ingested: {row_count} records → {table_name}\n")
    
    return table_name


def ingest_enforcement_tracker(spark: SparkSession) -> str:
    """
    Ingest enforcement tracker PDFs using AI document parser.
    
    Returns:
        Target table name
    """
    print("📄 Ingesting Enforcement Tracker PDFs...")
    
    source_config = SOURCES["enforcement"]
    volume_path = source_config["volume_path"]
    table_name = source_config["raw_table"]
    
    # Parse PDFs using AI function (returns DataFrame directly)
    df = parse_enforcement_pdfs(volume_path, spark)
    
    # Add metadata columns
    df = add_metadata_columns(df)
    
    print(f"   Parsed {df.count()} PDF documents")
    
    # Write to Delta table
    write_to_delta(
        df=df,
        table_name=table_name,
        mode="overwrite",
        overwrite_schema=True,
        enable_cdf=True
    )
    
    row_count = get_table_row_count(table_name, spark)
    print(f"✅ Enforcement tracker ingested: {row_count} documents → {table_name}\n")
    
    return table_name


def ingest_all_sources(sources: Optional[List[str]] = None) -> dict:
    """
    Ingest all or specified data sources.
    
    Args:
        sources: List of source names to ingest (gdpr, policy, enforcement)
                 If None, ingests all sources
                 
    Returns:
        Dictionary mapping source name to table name
    """
    spark = get_or_create_spark()
    
    # Default to all sources
    if sources is None:
        sources = ["gdpr", "policy", "enforcement"]
    
    results = {}
    
    print("=" * 70)
    print("🚀 STARTING DOCUMENT INGESTION PIPELINE")
    print("=" * 70 + "\n")
    
    # Ingest each source
    if "gdpr" in sources:
        try:
            results["gdpr"] = ingest_gdpr_legislation(spark)
        except Exception as e:
            print(f"❌ Failed to ingest GDPR: {e}\n")
    
    if "policy" in sources:
        try:
            results["policy"] = ingest_corporate_policy(spark)
        except Exception as e:
            print(f"❌ Failed to ingest policy: {e}\n")
    
    if "enforcement" in sources:
        try:
            results["enforcement"] = ingest_enforcement_tracker(spark)
        except Exception as e:
            print(f"❌ Failed to ingest enforcement tracker: {e}\n")
    
    print("=" * 70)
    print(f"🎉 INGESTION COMPLETE - {len(results)}/{len(sources)} sources successful")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest compliance documents to Delta tables")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["gdpr", "policy", "enforcement"],
        help="Specific sources to ingest (default: all)"
    )
    
    args = parser.parse_args()
    
    ingest_all_sources(sources=args.sources)