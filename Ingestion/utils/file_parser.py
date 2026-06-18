"""
File parsing utilities for compliance document ingestion.
Handles GDPR statutory legislation, corporate policy markdown, and enforcement tracker PDFs.
"""

import re
from typing import List, Dict
from pyspark.sql import SparkSession
from pyspark.sql.functions import expr, col


def _read_text_file(file_path: str, spark: SparkSession = None) -> str:
    """
    Read text file from Volume path using Spark.
    
    Args:
        file_path: Volume path to file
        spark: SparkSession (will create if None)
        
    Returns:
        File contents as string
    """
    if spark is None:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
    
    df = spark.read.text(file_path, wholetext=True)
    return df.collect()[0][0]


def parse_gdpr_markdown(file_path: str, spark: SparkSession = None) -> List[Dict[str, str]]:
    """
    Parse GDPR Chapter 3 markdown file into structured records.
    
    Args:
        file_path: Path to the GDPR markdown file
        spark: SparkSession (will create if None)
        
    Returns:
        List of dictionaries containing parsed article records
    """
    raw_text = _read_text_file(file_path, spark)
    
    # Split on Article headers (## Article N)
    segments = re.split(r'(?=\n##\s+Article\s+\d+)', raw_text)
    parsed_records = []
    
    for segment in segments:
        lines = [line.strip() for line in segment.split("\n") if line.strip()]
        if not lines:
            continue
            
        # Parse article header
        if lines[0].startswith("## Article"):
            article_id_raw = lines[0].replace("##", "").strip()
            article_id = article_id_raw.replace(" ", "-")
            article_title = (
                lines[1].replace("###", "").strip() 
                if len(lines) > 1 
                else "Statutory Rights"
            )
        else:
            # Preamble or intro content
            article_id = "CHAPTER_III_PREAMBLE"
            article_title = "General Modalities Context"
        
        parsed_records.append({
            "chunk_id": f"GDPR-CH3-{article_id.upper()}",
            "article_id": article_id,
            "article_title": article_title,
            "text_content": segment.strip(),
            "legislation_source": "EU GDPR Official 2016/679",
            "scope_boundary": "Chapter 3 - Rights of the Data Subject"
        })
    
    return parsed_records


def parse_policy_markdown(file_path: str, spark: SparkSession = None) -> List[Dict[str, str]]:
    """
    Parse retail privacy policy markdown file into structured records.
    
    Args:
        file_path: Path to the policy markdown file
        spark: SparkSession (will create if None)
        
    Returns:
        List of dictionaries containing parsed policy section records
    """
    raw_text = _read_text_file(file_path, spark)
    
    # Split on Section headers (## Section N)
    segments = re.split(r'(?=\n##\s+Section\s+\d+)', raw_text)
    parsed_records = []
    
    for segment in segments:
        lines = [line.strip() for line in segment.split("\n") if line.strip()]
        if not lines:
            continue
            
        # Parse section header
        if lines[0].startswith("## Section"):
            section_id_raw = lines[0].replace("##", "").strip()
            section_id = section_id_raw.split(":")[0].replace(" ", "_").lower()
            section_title = (
                section_id_raw.split(":")[1].strip() 
                if ":" in section_id_raw 
                else section_id_raw
            )
        else:
            # Preamble or intro content
            section_id = "policy_preamble"
            section_title = "Global Policy Overview"
        
        parsed_records.append({
            "chunk_id": f"RETAIL-POLICY-{section_id.upper()}",
            "section_id": section_id,
            "section_title": section_title,
            "text_content": segment.strip(),
            "corporate_owner": "Global Retail E-Commerce Corp",
            "document_type": "Internal Data Processing Matrix"
        })
    
    return parsed_records


def parse_enforcement_pdfs(volume_path: str, spark: SparkSession):
    """
    Parse enforcement tracker PDFs using AI document parser.
    This function uses Spark SQL AI functions and returns a DataFrame.
    
    Args:
        volume_path: Path to volume containing PDF files
        spark: Active SparkSession
        
    Returns:
        Spark DataFrame with flattened enforcement documents
    """
    # Step 1: Read binary PDF files
    raw_binary_df = (spark.read
                     .format("binaryFile")
                     .option("pathGlobFilter", "*.pdf")
                     .load(volume_path))
    
    # Step 2: Parse PDFs using AI function
    parsed_raw_df = raw_binary_df.withColumn(
        "ai_output", 
        expr("ai_parse_document(content)")
    )
    
    # Step 3: Flatten the document elements
    final_flattened_df = parsed_raw_df.select(
        expr("element_at(split(path, '/'), -1)").alias("source_file_name"),
        
        # Extract page count
        expr("""
            coalesce(
                cast(ai_output:document.metadata.page_count as int),
                array_max(
                    transform(
                        from_json(cast(ai_output:document.elements as string), 'array<struct<page_number:int>>'),
                        e -> e.page_number
                    )
                ),
                1
            )
        """).alias("source_page_count"),
        
        # Concatenate all text content in reading order
        expr("""
            concat_ws('\\n\\n',
                transform(
                    from_json(cast(ai_output:document.elements as string), 'array<struct<content:string>>'),
                    e -> e.content
                )
            )
        """).alias("full_document_text")
    )
    
    return final_flattened_df