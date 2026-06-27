"""
Spark helper utilities for data ingestion and processing.
Common operations for DataFrame creation, table management, and transformations.
"""

import logging
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.functions import col, expr, current_timestamp, sha2, concat_ws
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def get_or_create_spark() -> SparkSession:
    """
    Get existing SparkSession or create a new one.
    
    Returns:
        Active SparkSession
    """
    return SparkSession.builder.getOrCreate()


def list_to_dataframe(records: List[Dict], spark: SparkSession) -> DataFrame:
    """
    Convert a list of dictionaries to a Spark DataFrame.
    
    Args:
        records: List of dictionaries with consistent keys
        spark: Active SparkSession
        
    Returns:
        Spark DataFrame
    """
    if not records:
        error_msg = "Cannot create DataFrame from empty list"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.debug("Creating DataFrame from %d records", len(records))
    return spark.createDataFrame(records)


def table_exists(table_name: str, spark: SparkSession) -> bool:
    """
    Check if a table exists in Unity Catalog.
    
    Args:
        table_name: Fully qualified table name (catalog.schema.table)
        spark: Active SparkSession
        
    Returns:
        True if table exists, False otherwise
    """
    try:
        spark.table(table_name)
        logger.debug("Table exists: %s", table_name)
        return True
    except Exception:
        logger.debug("Table does not exist: %s", table_name)
        return False


def write_to_delta(
    df: DataFrame,
    table_name: str,
    mode: str = "overwrite",
    overwrite_schema: bool = True,
    enable_cdf: bool = False,
    partition_by: Optional[List[str]] = None
) -> None:
    """
    Write DataFrame to Delta table with common options.
    
    Args:
        df: DataFrame to write
        table_name: Fully qualified target table name
        mode: Write mode (append, overwrite, etc.)
        overwrite_schema: Whether to overwrite schema on write
        enable_cdf: Enable Change Data Feed for versioning
        partition_by: Optional list of columns to partition by
    """
    logger.info("Writing DataFrame to Delta table: %s (mode=%s)", table_name, mode)
    
    try:
        writer = (df.write
                  .format("delta")
                  .mode(mode)
                  .option("overwriteSchema", str(overwrite_schema).lower()))
        
        if partition_by:
            logger.info("Partitioning by columns: %s", ', '.join(partition_by))
            writer = writer.partitionBy(*partition_by)
        
        writer.saveAsTable(table_name)
        logger.info("Successfully wrote DataFrame to table: %s", table_name)
        
        # Enable Change Data Feed if requested
        if enable_cdf:
            spark = get_or_create_spark()
            spark.sql(f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
            logger.info("Change Data Feed enabled on table: %s", table_name)
            
    except Exception as e:
        logger.exception("Failed to write DataFrame to table %s: %s", table_name, e)
        raise


def add_metadata_columns(df: DataFrame) -> DataFrame:
    """
    Add common metadata columns for data lineage.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with added metadata columns
    """
    logger.debug("Adding metadata columns: ingestion_timestamp, record_hash")
    return (df
            .withColumn("ingestion_timestamp", current_timestamp())
            .withColumn("record_hash", sha2(concat_ws("||", *df.columns), 256)))


def validate_columns(df: DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        True if all columns present, raises ValueError otherwise
    """
    missing = set(required_columns) - set(df.columns)
    if missing:
        error_msg = "Missing required columns: %s"
        logger.error(error_msg, missing)
        raise ValueError(f"Missing required columns: {missing}")
    
    logger.debug("All required columns present: %s", ', '.join(required_columns))
    return True


def get_table_row_count(table_name: str, spark: SparkSession) -> int:
    """
    Get row count for a table.
    
    Args:
        table_name: Fully qualified table name
        spark: Active SparkSession
        
    Returns:
        Row count
    """
    logger.debug("Getting row count for table: %s", table_name)
    try:
        count = spark.table(table_name).count()
        logger.debug("Table %s has %d rows", table_name, count)
        return count
    except Exception as e:
        logger.exception("Failed to get row count for table %s: %s", table_name, e)
        raise


def read_volume_files(
    volume_path: str,
    file_format: str = "binaryFile",
    file_pattern: str = "*",
    spark: Optional[SparkSession] = None
) -> DataFrame:
    """
    Read files from a Unity Catalog volume.
    
    Args:
        volume_path: Path to UC volume (/Volumes/catalog/schema/volume)
        file_format: Format to read (binaryFile, text, json, etc.)
        file_pattern: Glob pattern for file filtering
        spark: Active SparkSession (will create if None)
        
    Returns:
        DataFrame with file contents
    """
    if spark is None:
        spark = get_or_create_spark()
    
    logger.info("Reading files from volume: %s (format=%s, pattern=%s)", 
                volume_path, file_format, file_pattern)
    
    try:
        reader = spark.read.format(file_format)
        
        if file_format == "binaryFile" and file_pattern != "*":
            reader = reader.option("pathGlobFilter", file_pattern)
        
        df = reader.load(volume_path)
        logger.info("Successfully read files from volume: %s", volume_path)
        return df
        
    except Exception as e:
        logger.exception("Failed to read files from volume %s: %s", volume_path, e)
        raise


def optimize_table(
    table_name: str,
    spark: SparkSession,
    zorder_cols: Optional[List[str]] = None
) -> None:
    """
    Run OPTIMIZE on a Delta table.
    
    Args:
        table_name: Fully qualified table name
        spark: Active SparkSession
        zorder_cols: Optional list of columns for ZORDER BY
    """
    optimize_sql = f"OPTIMIZE {table_name}"
    
    if zorder_cols:
        optimize_sql += f" ZORDER BY ({', '.join(zorder_cols)})"
        logger.info("Optimizing table %s with ZORDER BY columns: %s", 
                   table_name, ', '.join(zorder_cols))
    else:
        logger.info("Optimizing table: %s", table_name)
    
    try:
        spark.sql(optimize_sql)
        logger.info("Successfully optimized table: %s", table_name)
    except Exception as e:
        logger.exception("Failed to optimize table %s: %s", table_name, e)
        raise


def get_table_properties(table_name: str, spark: SparkSession) -> Dict[str, str]:
    """
    Get table properties as a dictionary.
    
    Args:
        table_name: Fully qualified table name
        spark: Active SparkSession
        
    Returns:
        Dictionary of table properties
    """
    logger.debug("Retrieving table properties for: %s", table_name)
    
    try:
        result = spark.sql(f"SHOW TBLPROPERTIES {table_name}").collect()
        properties = {row['key']: row['value'] for row in result}
        logger.debug("Retrieved %d properties for table: %s", len(properties), table_name)
        return properties
    except Exception as e:
        logger.exception("Failed to retrieve table properties for %s: %s", table_name, e)
        raise
