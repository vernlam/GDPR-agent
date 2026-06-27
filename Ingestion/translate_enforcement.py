"""
Translation pipeline for enforcement tracker documents.
Translates multilingual GDPR enforcement precedents to English.
"""

import argparse
import logging
from typing import Optional
from pyspark.sql import SparkSession

from config import SOURCES, TRANSLATION_TARGET_LANGUAGE, TRANSLATION_CHUNK_SIZE
from utils.translation_utils import translate_enforcement_documents as translate_util
from utils.spark_helpers import get_or_create_spark, table_exists, get_table_row_count

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def translate_enforcement_documents(
    spark: Optional[SparkSession] = None,
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
    
    logger.info("=" * 70)
    logger.info("ENFORCEMENT DOCUMENT TRANSLATION")
    logger.info("=" * 70)
    
    # Verify source table exists
    if not table_exists(source_table, spark):
        error_msg = "Source table %s does not exist. Run ingestion first."
        logger.error(error_msg, source_table)
        raise ValueError(f"Source table {source_table} does not exist. Run ingestion first.")
    
    source_count = get_table_row_count(source_table, spark)
    logger.info("Source documents to translate: %d", source_count)
    logger.info("Target language: %s", target_language)
    logger.info("Translation chunk size: %d characters", chunk_size)
    
    # Run translation
    logger.info("Initiating translation pipeline")
    try:
        translated_df = translate_util(
            source_table=source_table,
            target_table=target_table,
            spark=spark,
            target_language=target_language,
            chunk_size=chunk_size
        )
    except Exception as e:
        logger.exception("Translation pipeline failed: %s", e)
        raise
    
    target_count = get_table_row_count(target_table, spark)
    
    logger.info("=" * 70)
    logger.info("TRANSLATION COMPLETE")
    logger.info("=" * 70)
    logger.info("Translated documents: %d", target_count)
    logger.info("Translated documents saved to table: %s", target_table)
    
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
