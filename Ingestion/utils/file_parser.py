"""
File parsing utilities for compliance document ingestion.
Handles GDPR statutory legislation, corporate policy markdown, and enforcement tracker PDFs.
"""

import re
import logging
from typing import List, Dict, Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import expr, col

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def _read_text_file(file_path: str, spark: Optional[SparkSession] = None) -> str:
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
    
    logger.debug("Reading text file from path: %s", file_path)
    
    try:
        df = spark.read.text(file_path, wholetext=True)
        content = df.collect()[0][0]
        logger.debug("Successfully read file: %d characters", len(content))
        return content
    except Exception as e:
        logger.exception("Failed to read text file from path %s: %s", file_path, e)
        raise


def parse_gdpr_markdown(file_path: str, spark: Optional[SparkSession] = None) -> List[Dict[str, str]]:
    """
    Parse GDPR Chapter 3 markdown file into structured records.
    
    Args:
        file_path: Path to the GDPR markdown file
        spark: SparkSession (will create if None)
        
    Returns:
        List of dictionaries containing parsed article records
    """
    logger.info("Parsing GDPR markdown file: %s", file_path)
    
    try:
        raw_text = _read_text_file(file_path, spark)
    except Exception as e:
        logger.exception("Failed to parse GDPR markdown file %s: %s", file_path, e)
        raise
    
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
    
    logger.info("Successfully parsed GDPR markdown: %d articles extracted", len(parsed_records))
    return parsed_records


def parse_policy_markdown(file_path: str, spark: Optional[SparkSession] = None) -> List[Dict[str, str]]:
    """
    Parse retail privacy policy markdown file into structured records.
    
    Args:
        file_path: Path to the policy markdown file
        spark: SparkSession (will create if None)
        
    Returns:
        List of dictionaries containing parsed policy section records
    """
    logger.info("Parsing corporate policy markdown file: %s", file_path)
    
    try:
        raw_text = _read_text_file(file_path, spark)
    except Exception as e:
        logger.exception("Failed to parse policy markdown file %s: %s", file_path, e)
        raise
    
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
    
    logger.info("Successfully parsed corporate policy markdown: %d sections extracted", len(parsed_records))
    return parsed_records


def parse_enforcement_pdfs(volume_path: str, spark: SparkSession) -> DataFrame:
    """
    Parse enforcement tracker PDFs using AI document parser.
    This function uses Spark SQL AI functions and returns a DataFrame.
    
    Args:
        volume_path: Path to volume containing PDF files
        spark: Active SparkSession
        
    Returns:
        Spark DataFrame with flattened enforcement documents (cached)
    """
    logger.info("Parsing enforcement tracker PDFs from volume: %s", volume_path)
    
    # Step 1: Read binary PDF files (action - can fail)
    logger.info("Reading PDF files from volume using binaryFile format")
    try:
        raw_binary_df = (spark.read
                         .format("binaryFile")
                         .option("pathGlobFilter", "*.pdf")
                         .load(volume_path))
        pdf_count = raw_binary_df.count()
        logger.info("Found %d PDF files to parse", pdf_count)
    except Exception as e:
        logger.exception("Failed to read PDF files from volume %s: %s", volume_path, e)
        raise
    
    # Step 2: Define transformations (lazy - cannot fail, just building execution plan)
    # Using chained select() calls instead of withColumn() to avoid nested execution plans
    logger.info("Defining PDF parsing transformations")
    final_flattened_df = raw_binary_df.select(
        expr("element_at(split(path, '/'), -1)").alias("source_file_name"),
        expr("ai_parse_document(content)").alias("ai_output")
    ).select(
        col("source_file_name"),
        
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
    ).cache()  # Mark for caching
    
    # Step 3: Execute transformations (action - can fail)
    logger.info("Executing PDF parsing and flattening transformations")
    try:
        result_count = final_flattened_df.count()
        logger.info("Successfully parsed and flattened %d enforcement PDFs", result_count)
        return final_flattened_df
    except Exception as e:
        logger.exception("Failed to parse and flatten enforcement PDFs: %s", e)
        raise
