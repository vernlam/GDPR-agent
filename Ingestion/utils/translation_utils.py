"""
Translation utilities for multilingual compliance documents.
Uses Databricks SQL AI translation functions for batch processing.
"""

import re
import logging
from typing import List
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col, expr, explode, concat_ws, struct, array_sort, collect_list, lit, size, length
from pyspark.sql.types import ArrayType, StringType

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def chunk_text(text: str, max_chars: int = 3000) -> List[str]:
    """
    Split text into chunks at sentence boundaries, respecting max character limit.
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        
    Returns:
        List of text chunks
    """
    # Split on sentence boundaries (., !, ?, followed by space or newline)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence exceeds limit, save current chunk and start new
        if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence
    
    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def translate_enforcement_documents(
    source_table: str,
    target_table: str,
    spark: SparkSession,
    target_language: str = "English",
    chunk_size: int = 3000
) -> DataFrame:
    """
    Translate enforcement tracker documents using AI translation.
    
    Args:
        source_table: Fully qualified source table name
        target_table: Fully qualified target table name  
        spark: Active SparkSession
        target_language: Target language for translation
        chunk_size: Maximum characters per translation chunk
        
    Returns:
        Translated DataFrame
    """
    logger.info("Initiating translation pipeline for enforcement documents")
    logger.info("Source table: %s", source_table)
    logger.info("Target table: %s", target_table)
    logger.info("Target language: %s", target_language)
    logger.info("Chunk size: %d characters", chunk_size)
    
    # Register the chunking UDF
    logger.debug("Registering chunk_text UDF")
    spark.udf.register("chunk_text", lambda text: chunk_text(text, chunk_size), ArrayType(StringType()))
    
    # Read source documents
    logger.info("Reading source documents from table: %s", source_table)
    try:
        source_df = spark.table(source_table)
    except Exception as e:
        logger.exception("Failed to read source table %s: %s", source_table, e)
        raise
    
    # Step 1: Create chunks with sequential IDs
    logger.info("Step 1: Chunking documents for translation")
    chunked_df = source_df.select(
        col("source_file_name"),
        col("source_page_count"),
        explode(expr("chunk_text(full_document_text)")).alias("chunk_text")
    ).withColumn(
        "chunk_id",
        expr("row_number() over (partition by source_file_name order by monotonically_increasing_id())")
    )
    
    try:
        chunk_count = chunked_df.count()
        logger.info("Created %d text chunks for translation", chunk_count)
    except Exception as e:
        logger.exception("Failed to create text chunks: %s", e)
        raise
    
    # Step 2: Translate each chunk using AI function
    logger.info("Step 2: Translating chunks using AI translation function")
    translated_df = chunked_df.withColumn(
        "translated_text",
        expr(f"ai_translate(chunk_text, '{target_language}')")
    ).withColumn(
        "original_len", length(col("chunk_text"))
    ).withColumn(
        "translated_len", length(col("translated_text"))
    )
    
    # Step 3: Reassemble chunks back into full documents
    logger.info("Step 3: Reassembling translated chunks into full documents")
    reassembled_df = translated_df.groupBy("source_file_name", "source_page_count").agg(
        concat_ws("\n\n", 
            array_sort(
                collect_list(
                    struct(col("chunk_id"), col("translated_text"))
                )
            )["translated_text"]
        ).alias("full_document_text_translated"),
        
        # Keep original for validation
        concat_ws("\n\n",
            array_sort(
                collect_list(
                    struct(col("chunk_id"), col("chunk_text"))
                )
            )["chunk_text"]
        ).alias("full_document_text_original")
    )
    
    # Step 4: Add translation quality metrics
    logger.info("Step 4: Computing translation quality metrics")
    final_df = reassembled_df.withColumn(
        "translation_ratio",
        expr("length(full_document_text_translated) / length(full_document_text_original)")
    ).withColumn(
        "target_language", lit(target_language)
    )
    
    # Validation check
    logger.info("Validating translation quality")
    try:
        low_quality = final_df.filter(col("translation_ratio") < 0.6).count()
        if low_quality > 0:
            logger.warning("Translation quality warning: %d documents have translation ratio < 0.6", low_quality)
        else:
            logger.info("Translation quality validation passed: all documents meet quality threshold")
    except Exception as e:
        logger.exception("Failed to validate translation quality: %s", e)
        raise
    
    # Save to target table
    logger.info("Writing translated documents to target table: %s", target_table)
    try:
        (final_df.write
            .format("delta")
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(target_table))
        logger.info("Successfully saved translated documents to table: %s", target_table)
    except Exception as e:
        logger.exception("Failed to write translated documents to table %s: %s", target_table, e)
        raise
    
    return final_df
