"""
Document ingestion pipeline for GDPR compliance sources.
Handles GDPR legislation, corporate policy, and enforcement tracker documents.
"""

import argparse
import logging
from typing import List, Optional, Dict
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

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_gdpr_legislation(spark: SparkSession) -> str:
    """
    Ingest GDPR Chapter 3 markdown file.
    
    Returns:
        Target table name
    """
    logger.info("Initiating GDPR statutory legislation ingestion")
    
    source_config = SOURCES["gdpr"]
    file_path = source_config["file"]
    table_name = source_config["table"]
    
    # Parse markdown file
    records = parse_gdpr_markdown(file_path, spark)
    logger.info("Parsed %d articles from markdown file: %s", len(records), file_path)
    
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
    logger.info("Successfully ingested GDPR legislation: %d records written to %s", row_count, table_name)
    
    return table_name


def ingest_corporate_policy(spark: SparkSession) -> str:
    """
    Ingest retail privacy policy markdown file.
    
    Returns:
        Target table name
    """
    logger.info("Initiating corporate privacy policy ingestion")
    
    source_config = SOURCES["policy"]
    file_path = source_config["file"]
    table_name = source_config["table"]
    
    # Parse markdown file
    records = parse_policy_markdown(file_path, spark)
    logger.info("Parsed %d sections from markdown file: %s", len(records), file_path)
    
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
    logger.info("Successfully ingested corporate policy: %d records written to %s", row_count, table_name)
    
    return table_name


def ingest_enforcement_tracker(spark: SparkSession) -> str:
    """
    Ingest enforcement tracker PDFs using AI document parser.
    
    Returns:
        Target table name
    """
    logger.info("Initiating enforcement tracker PDF ingestion")
    
    source_config = SOURCES["enforcement"]
    volume_path = source_config["volume_path"]
    table_name = source_config["raw_table"]
    
    # Parse PDFs using AI function (returns DataFrame directly)
    df = parse_enforcement_pdfs(volume_path, spark)
    
    # Add metadata columns
    df = add_metadata_columns(df)
    
    doc_count = df.count()
    logger.info("Parsed %d PDF documents from volume: %s", doc_count, volume_path)
    
    # Write to Delta table
    write_to_delta(
        df=df,
        table_name=table_name,
        mode="overwrite",
        overwrite_schema=True,
        enable_cdf=True
    )
    
    row_count = get_table_row_count(table_name, spark)
    logger.info("Successfully ingested enforcement tracker: %d documents written to %s", row_count, table_name)
    
    return table_name


def ingest_all_sources(sources: Optional[List[str]] = None) -> Dict[str, str]:
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
    
    logger.info("=" * 70)
    logger.info("STARTING DOCUMENT INGESTION PIPELINE")
    logger.info("=" * 70)
    
    # Ingest each source
    if "gdpr" in sources:
        try:
            results["gdpr"] = ingest_gdpr_legislation(spark)
        except Exception as e:
            logger.exception("Failed to ingest GDPR legislation: %s", e)
    
    if "policy" in sources:
        try:
            results["policy"] = ingest_corporate_policy(spark)
        except Exception as e:
            logger.exception("Failed to ingest corporate policy: %s", e)
    
    if "enforcement" in sources:
        try:
            results["enforcement"] = ingest_enforcement_tracker(spark)
        except Exception as e:
            logger.exception("Failed to ingest enforcement tracker: %s", e)
    
    logger.info("=" * 70)
    logger.info("INGESTION COMPLETE - %d/%d sources successful", len(results), len(sources))
    logger.info("=" * 70)
    
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