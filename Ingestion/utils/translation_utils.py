"""
Translation utilities for multilingual compliance documents.
Uses Databricks SQL AI translation functions for batch processing.
"""

from pyspark.sql import DataFrame
from pyspark.sql.functions import col, expr, explode, concat_ws, struct, array_sort, collect_list, lit, size, length
import re


def chunk_text(text: str, max_chars: int = 3000) -> list[str]:
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
    spark,
    target_language: str = "English",
    chunk_size: int = 3000
):
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
    # Register the chunking UDF
    from pyspark.sql.types import ArrayType, StringType
    spark.udf.register("chunk_text", lambda text: chunk_text(text, chunk_size), ArrayType(StringType()))
    
    # Read source documents
    source_df = spark.table(source_table)
    
    # Step 1: Create chunks with sequential IDs
    chunked_df = source_df.select(
        col("source_file_name"),
        col("source_page_count"),
        explode(expr(f"chunk_text(full_document_text)")).alias("chunk_text")
    ).withColumn(
        "chunk_id",
        expr("row_number() over (partition by source_file_name order by monotonically_increasing_id())")
    )
    
    print(f"📦 Created {chunked_df.count()} chunks for translation")
    
    # Step 2: Translate each chunk using AI function
    translated_df = chunked_df.withColumn(
        "translated_text",
        expr(f"ai_translate(chunk_text, '{target_language}')")
    ).withColumn(
        "original_len", length(col("chunk_text"))
    ).withColumn(
        "translated_len", length(col("translated_text"))
    )
    
    # Step 3: Reassemble chunks back into full documents
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
    final_df = reassembled_df.withColumn(
        "translation_ratio",
        expr("length(full_document_text_translated) / length(full_document_text_original)")
    ).withColumn(
        "target_language", lit(target_language)
    )
    
    # Validation check
    low_quality = final_df.filter(col("translation_ratio") < 0.6).count()
    if low_quality > 0:
        print(f"⚠️  Warning: {low_quality} documents have translation ratio < 0.6")
    else:
        print(f"✅ All translations passed quality check")
    
    # Save to target table
    (final_df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(target_table))
    
    print(f"🎉 Translated documents saved to {target_table}")
    
    return final_df