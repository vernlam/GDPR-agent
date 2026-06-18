"""
Generate embeddings for compliance documents using Databricks Foundation Models.
Creates vector representations for GDPR legislation, corporate policy, and enforcement precedents.
"""

import argparse
from typing import List, Optional
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


def vectorize_gdpr_legislation(spark: SparkSession) -> str:
    """
    Generate embeddings for GDPR statutory legislation.
    
    Returns:
        Target embeddings table name
    """
    print("🔢 Vectorizing GDPR Statutory Legislation...")
    
    source_config = SOURCES["gdpr"]
    source_table = source_config["table"]
    embeddings_table = source_config["embeddings_table"]
    
    # Read source table
    df = spark.table(source_table)
    
    # Generate embeddings on text_content column
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
    print(f"✅ GDPR embeddings generated: {row_count} records → {embeddings_table}\n")
    
    return embeddings_table


def vectorize_corporate_policy(spark: SparkSession) -> str:
    """
    Generate embeddings for corporate privacy policy.
    
    Returns:
        Target embeddings table name
    """
    print("🔢 Vectorizing Corporate Privacy Policy...")
    
    source_config = SOURCES["policy"]
    source_table = source_config["table"]
    embeddings_table = source_config["embeddings_table"]
    
    # Read source table
    df = spark.table(source_table)
    
    # Generate embeddings on text_content column
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
    print(f"✅ Corporate policy embeddings generated: {row_count} records → {embeddings_table}\n")
    
    return embeddings_table


def vectorize_enforcement_tracker(spark: SparkSession, use_translated: bool = True) -> str:
    """
    Generate embeddings for enforcement tracker documents.
    
    Args:
        use_translated: If True, uses translated table; otherwise uses raw table
        
    Returns:
        Target embeddings table name
    """
    print("🔢 Vectorizing Enforcement Tracker Documents...")
    
    source_config = SOURCES["enforcement"]
    
    # Choose source table based on translation status
    if use_translated:
        source_table = source_config["translated_table"]
        text_column = "full_document_text_translated"
        print(f"   Using translated documents from {source_table}")
    else:
        source_table = source_config["raw_table"]
        text_column = "full_document_text"
        print(f"   Using raw documents from {source_table}")
    
    embeddings_table = source_config["embeddings_table"]
    
    # Check if source table exists
    if not table_exists(source_table, spark):
        raise ValueError(f"Source table {source_table} does not exist. Run ingestion/translation first.")
    
    # Read source table
    df = spark.table(source_table)
    
    # Generate embeddings on the appropriate text column
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
    print(f"✅ Enforcement embeddings generated: {row_count} documents → {embeddings_table}\n")
    
    return embeddings_table


def vectorize_all_sources(
    sources: Optional[List[str]] = None,
    use_translated_enforcement: bool = True
) -> dict:
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
    
    print("=" * 70)
    print("🚀 STARTING VECTORIZATION PIPELINE")
    print("=" * 70 + "\n")
    print(f"Embedding Model: {EMBEDDING_MODEL}\n")
    
    # Vectorize each source
    if "gdpr" in sources:
        try:
            results["gdpr"] = vectorize_gdpr_legislation(spark)
        except Exception as e:
            print(f"❌ Failed to vectorize GDPR: {e}\n")
    
    if "policy" in sources:
        try:
            results["policy"] = vectorize_corporate_policy(spark)
        except Exception as e:
            print(f"❌ Failed to vectorize policy: {e}\n")
    
    if "enforcement" in sources:
        try:
            results["enforcement"] = vectorize_enforcement_tracker(
                spark, 
                use_translated=use_translated_enforcement
            )
        except Exception as e:
            print(f"❌ Failed to vectorize enforcement tracker: {e}\n")
    
    print("=" * 70)
    print(f"🎉 VECTORIZATION COMPLETE - {len(results)}/{len(sources)} sources successful")
    print("=" * 70)
    
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