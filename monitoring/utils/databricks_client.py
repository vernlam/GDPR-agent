"""
Databricks client utilities.

Provides a unified wrapper for Databricks SDK and Spark operations,
including SQL query execution, serving endpoint status checks, and
metrics persistence to Delta tables.
"""

import logging
from typing import Dict, Any, Optional

from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession, DataFrame

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class DatabricksClient:
    """
    Wrapper for Databricks operations.
    
    Provides unified access to Databricks Workspace SDK and PySpark operations
    for monitoring and data access tasks.
    """
    
    def __init__(self) -> None:
        """
        Initialize Databricks client with Workspace SDK and Spark session.
        
        Raises:
            Exception: If Workspace client or Spark session initialization fails
                      (logged but not re-raised)
        """
        logger.debug("Initializing DatabricksClient")
        
        try:
            self.w = WorkspaceClient()
            logger.debug("WorkspaceClient initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize WorkspaceClient: %s", e)
            raise
        
        try:
            self.spark = SparkSession.builder.getOrCreate()
            logger.debug("SparkSession obtained successfully")
        except Exception as e:
            logger.exception("Failed to obtain SparkSession: %s", e)
            raise
        
        logger.info("DatabricksClient initialized successfully")
    
    def query_table(self, query: str) -> DataFrame:
        """
        Execute SQL query and return DataFrame.
        
        Args:
            query: SQL query string to execute
        
        Returns:
            Spark DataFrame containing query results
        
        Raises:
            Exception: If query execution fails (logged and re-raised)
        """
        query_preview = query[:100] if len(query) > 100 else query
        logger.debug("Executing SQL query: %s", query_preview)
        
        try:
            df = self.spark.sql(query)
            logger.debug("Query executed successfully")
            return df
        except Exception as e:
            logger.exception("Failed to execute SQL query: %s", e)
            raise
    
    def get_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Get serving endpoint status.
        
        Retrieves the current state and configuration of a Databricks
        model serving endpoint.
        
        Args:
            endpoint_name: Name of the serving endpoint to query
        
        Returns:
            Dict containing endpoint status:
            - name: Endpoint name
            - state: Current state (e.g., "READY", "NOT_READY", "UNKNOWN")
        
        Raises:
            Exception: If endpoint retrieval fails (logged and re-raised)
        """
        logger.debug("Retrieving status for endpoint: %s", endpoint_name)
        
        try:
            endpoint = self.w.serving_endpoints.get(endpoint_name)
            
            state = "UNKNOWN"
            if endpoint.state:
                state = str(endpoint.state.ready) if endpoint.state.ready is not None else "UNKNOWN"
            
            status = {
                "name": endpoint_name,
                "state": state
            }
            
            logger.info("Endpoint status retrieved: %s state=%s", endpoint_name, state)
            return status
            
        except Exception as e:
            logger.exception("Failed to get endpoint status for %s: %s", endpoint_name, e)
            raise
    
    def write_metrics(self, df: DataFrame, table_name: str) -> None:
        """
        Write metrics DataFrame to Delta table.
        
        Appends the provided DataFrame to a Delta table for persistent
        storage of monitoring metrics.
        
        Args:
            df: Spark DataFrame containing metrics to persist
            table_name: Fully qualified table name (catalog.schema.table)
        
        Returns:
            None
        
        Raises:
            Exception: If write operation fails (logged and re-raised)
        """
        logger.debug("Writing metrics to table: %s", table_name)
        
        try:
            row_count = df.count()
            logger.debug("DataFrame contains %d rows", row_count)
            
            df.write.mode("append").saveAsTable(table_name)
            
            logger.info("Successfully wrote %d rows to table: %s", row_count, table_name)
            
        except Exception as e:
            logger.exception("Failed to write metrics to table %s: %s", table_name, e)
            raise
