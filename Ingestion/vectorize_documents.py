"""
Generate embeddings for compliance documents using Databricks Foundation Models.
Creates vector representations for GDPR legislation, corporate policy, and enforcement precedents.
"""

import argparse
import logging
from typing import List, Optional, Dict
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr, length

from config import SOURCES, EMBEDDING_MODEL
from utils.spark_helpers import (
    get_or_create_spark,
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


def vectorize_gdpr_legislation(spark: SparkSession) -> str:
    """
    Generate embeddings for GDPR statutory legislation.
    
    Args:
        spark: Active SparkSession
    
    Returns:
        Target embeddings table name
    """
    logger.info("Initiating GDPR statutory legislation vectorization")
    
    source_config = SOURCES["gdpr"]
    source_table = source_config["table"]
    embeddings_table = source_config["embeddings_table"]
    
    logger.info("Source table: %s", source_table)
    logger.info("Target embeddings table: %s", embeddings_table)
    
    # Read source table
    df = spark.table(source_table)
    
    # Generate embeddings on text_content column
    logger.info("Generating embeddings using model: %s", EMBEDDING_MODEL)
    embeddings_df = df.withColumn(
        "embedding",
        expr(f"ai_generate_embedding('{EMBEDDING_MODEL}', text_content)")
    ).withColumn(
        "embedding_model", expr(f"'{EMBEDDING_MODEL}'")
    ).withColumn(
        "text_length", length(col("text_content"))
    )
    
    # Write to embeddings table
    write_to_delta(
        df=embeddings_df,
        table_name=embeddings_table,
        mode="overwrite",
        overwrite_schema=True,
        enable_cdf=False
    )
    
    row_count = get_table_row_count(embeddings_table, spark)
    logger.info("GDPR embeddings generated successfully: %d records written to %s", row_count, embeddings_table)
    
    return embeddings_table


def vectorize_corporate_policy(spark: SparkSession) -> str:
    """
    Generate embeddings for corporate privacy policy.
    
    Args:
        spark: Active SparkSession
    
    Returns:
        Target embeddings table name
    """
    logger.info("Initiating corporate privacy policy vectorization")
    
    source_config = SOURCES["policy"]
    source_table = source_config["table"]
    embeddings_table = source_config["embeddings_table"]
    
    logger.info("Source table: %s", source_table)
    logger.info("Target embeddings table: %s", embeddings_table)
    
    # Read source table
    df = spark.table(source_table)
    
    # Generate embeddings on text_content column
    logger.info("Generating embeddings using model: %s", EMBEDDING_MODEL)
    embeddings_df = df.withColumn(
        "embedding",
        expr(f"ai_generate_embedding('{EMBEDDING_MODEL}', text_content)")
    ).withColumn(
        "embedding_model", expr(f"'{EMBEDDING_MODEL}'")
    ).withColumn(
        "text_length", length(col("text_content"))
    )
    
    # Write to embeddings table
    write_to_delta(
        df=embeddings_df,
        table_name=embeddings_table,
        mode="overwrite",
        overwrite_schema=True,
        enable_cdf=False
    )
    
    row_count = get_table_row_count(embeddings_table, spark)
    logger.info("Corporate policy embeddings generated successfully: %d records written to %s", row_count, embeddings_table)
    
    return embeddings_table


def vectorize_enforcement_tracker(spark: SparkSession, use_translated: bool = True) -> str:
    """
    Generate embeddings for enforcement tracker documents.
    
    Args:
        spark: Active SparkSession
        use_translated: If True, uses translated table; otherwise uses raw table
        
    Returns:
        Target embeddings table name
    """
    logger.info("Initiating enforcement tracker vectorization")
    
    source_config = SOURCES["enforcement"]
    
    # Choose source table based on translation status
    if use_translated:
        source_table = source_config["translated_table"]
        text_column = "full_document_text_translated"
        logger.info("Using translated documents from table: %s", source_table)
    else:
        source_table = source_config["raw_table"]
        text_column = "full_document_text"
        logger.info("Using raw multilingual documents from table: %s", source_table)
    
    embeddings_table = source_config["embeddings_table"]
    logger.info("Target embeddings table: %s", embeddings_table)
    
    # Check if source table exists
    if not table_exists(source_table, spark):
        error_msg = "Source table %s does not exist. Run ingestion/translation first."
        logger.error(error_msg, source_table)
        raise ValueError(f"Source table {source_table} does not exist. Run ingestion/translation first.")
    
    # Read source table
    df = spark.table(source_table)
    
    # Generate embeddings on the appropriate text column
    logger.info("Generating embeddings using model: %s on column: %s", EMBEDDING_MODEL, text_column)
    embeddings_df = df.withColumn(
        "embedding",
        expr(f"ai_generate_embedding('{EMBEDDING_MODEL}', {text_column})")
    ).withColumn(
        "embedding_model", expr(f"'{EMBEDDING_MODEL}'")
    ).withColumn(
        "text_length", length(col(text_column))
    ).withColumn(
        "is_translated", expr(f"{str(use_translated).lower()}")
    )
    
    # Write to embeddings table
    write_to_delta(
        df=embeddings_df,
        table_name=embeddings_table,
        mode="overwrite",
        overwrite_schema=True,
        enable_cdf=False
    )
    
    row_count = get_table_row_count(embeddings_table, spark)
    logger.info("Enforcement tracker embeddings generated successfully: %d documents written to %s", row_count, embeddings_table)
    
    return embeddings_table


def vectorize_all_sources(
    sources: Optional[List[str]] = None,
    use_translated_enforcement: bool = True
) -> Dict[str, str]:
    """
    Generate embeddings for all or specified data sources.
    
    Args:
        sources: List of source names to vectorize (gdpr, policy, enforcement)
                 If None, vectorizes all sources
        use_translated_enforcement: Whether to use translated enforcement docs
                 
    Returns:
        Dictionary mapping source name to embeddings table name
    """
    spark = get_or_create_spark()
    
    # Default to all sources
    if sources is None:
        sources = ["gdpr", "policy", "enforcement"]
    
    results = {}
    
    logger.info("=" * 70)
    logger.info("STARTING VECTORIZATION PIPELINE")
    logger.info("=" * 70)
    logger.info("Embedding model: %s", EMBEDDING_MODEL)
    logger.info("Sources to vectorize: %s", ', '.join(sources))
    
    # Vectorize each source
    if "gdpr" in sources:
        try:
            results["gdpr"] = vectorize_gdpr_legislation(spark)
        except Exception as e:
            logger.exception("Failed to vectorize GDPR statutory legislation: %s", e)
    
    if "policy" in sources:
        try:
            results["policy"] = vectorize_corporate_policy(spark)
        except Exception as e:
            logger.exception("Failed to vectorize corporate policy: %s", e)
    
    if "enforcement" in sources:
        try:
            results["enforcement"] = vectorize_enforcement_tracker(
                spark, 
                use_translated=use_translated_enforcement
            )
        except Exception as e:
            logger.exception("Failed to vectorize enforcement tracker: %s", e)
    
    logger.info("=" * 70)
    logger.info("VECTORIZATION COMPLETE - %d/%d sources successful", len(results), len(sources))
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embeddings for compliance documents")
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=["gdpr", "policy", "enforcement"],
        help="Specific sources to vectorize (default: all)"
    )
    parser.add_argument(
        "--use-raw-enforcement",
        action="store_true",
        help="Use raw (untranslated) enforcement documents instead of translated"
    )
    
    args = parser.parse_args()
    
    vectorize_all_sources(
        sources=args.sources,
        use_translated_enforcement=not args.use_raw_enforcement
    )
