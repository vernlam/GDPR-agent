"""
Translation pipeline for enforcement tracker documents.
Translates multilingual GDPR enforcement precedents to English.
"""

import argparse
from pyspark.sql import SparkSession

from config import SOURCES, TRANSLATION_TARGET_LANGUAGE, TRANSLATION_CHUNK_SIZE
from utils.translation_utils import translate_enforcement_documents as translate_util
from utils.spark_helpers import get_or_create_spark, table_exists, get_table_row_count


def translate_enforcement_documents(
    spark: SparkSession = None,
    target_language: str = TRANSLATION_TARGET_LANGUAGE,
    chunk_size: int = TRANSLATION_CHUNK_SIZE
) -> str:
    """
    Translate enforcement tracker documents.
    
    Args:
        spark: Active SparkSession (will create if None)
        target_language: Target language for translation
        chunk_size: Maximum characters per translation chunk
        
    Returns:
        Target table name with translated documents
    """
    if spark is None:
        spark = get_or_create_spark()
    
    source_config = SOURCES["enforcement"]
    source_table = source_config["raw_table"]
    target_table = source_config["translated_table"]
    
    print("=" * 70)
    print("🌐 ENFORCEMENT DOCUMENT TRANSLATION")
    print("=" * 70 + "\n")
    
    # Verify source table exists
    if not table_exists(source_table, spark):
        raise ValueError(f"Source table {source_table} does not exist. Run ingestion first.")
    
    source_count = get_table_row_count(source_table, spark)
    print(f"📄 Source documents: {source_count}")
    print(f"🎯 Target language: {target_language}")
    print(f"✂️  Chunk size: {chunk_size} characters\n")
    
    # Run translation
    translated_df = translate_util(
        source_table=source_table,
        target_table=target_table,
        spark=spark,
        target_language=target_language,
        chunk_size=chunk_size
    )
    
    target_count = get_table_row_count(target_table, spark)
    
    print("\n" + "=" * 70)
    print("✅ TRANSLATION COMPLETE")
    print("=" * 70)
    print(f"📊 Translated documents: {target_count}")
    print(f"💾 Saved to: {target_table}\n")
    
    return target_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate enforcement tracker documents")
    parser.add_argument(
        "--language",
        default=TRANSLATION_TARGET_LANGUAGE,
        help=f"Target language (default: {TRANSLATION_TARGET_LANGUAGE})"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=TRANSLATION_CHUNK_SIZE,
        help=f"Chunk size in characters (default: {TRANSLATION_CHUNK_SIZE})"
    )
    
    args = parser.parse_args()
    
    translate_enforcement_documents(
        target_language=args.language,
        chunk_size=args.chunk_size
    )