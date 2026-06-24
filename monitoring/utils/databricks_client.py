"""
Databricks client utilities.
"""
from databricks.sdk import WorkspaceClient
from pyspark.sql import SparkSession, DataFrame

class DatabricksClient:
    """Wrapper for Databricks operations"""
    
    def __init__(self):
        self.w = WorkspaceClient()
        self.spark = SparkSession.builder.getOrCreate()
    
    def query_table(self, query: str) -> DataFrame:
        """Execute SQL query and return DataFrame"""
        return self.spark.sql(query)
    
    def get_endpoint_status(self, endpoint_name: str) -> dict:
        """Get serving endpoint status"""
        endpoint = self.w.serving_endpoints.get(endpoint_name)
        return {
            "name": endpoint_name,
            "state": str(endpoint.state.ready) if endpoint.state else "UNKNOWN",
            "url": endpoint.url
        }
    
    def write_metrics(self, df: DataFrame, table_name: str):
        """Write metrics to Delta table"""
        df.write.mode("append").saveAsTable(table_name)