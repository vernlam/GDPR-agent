"""
Performance monitoring (latency, throughput, success rate).
"""
from typing import Dict
import pandas as pd

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient

class PerformanceMonitor:
    """Monitor endpoint performance metrics"""
    
    def __init__(self, db_client: DatabricksClient):
        self.db = db_client
    
    def get_performance_metrics(self, days_back: int = 7) -> pd.DataFrame:
        """Get performance metrics for the specified period"""
        
        query = f"""
            SELECT 
                p.date,
                COUNT(*) as total_requests,
                COUNT(DISTINCT get_json_object(p.request, '$.dataframe_split.data[0][0]')) as unique_questions,
                AVG((r.timestamp_ms - p.timestamp_ms) / 1000) as avg_latency_seconds,
                MAX((r.timestamp_ms - p.timestamp_ms) / 1000) as max_latency_seconds,
                MIN((r.timestamp_ms - p.timestamp_ms) / 1000) as min_latency_seconds,
                SUM(CASE WHEN p.status_code = 200 THEN 1 ELSE 0 END) / COUNT(*) * 100 as success_rate_pct
            FROM {config.payload_table} p
            LEFT JOIN {config.response_table} r
                ON p.request_id = r.request_id
            WHERE p.date >= current_date() - {days_back}
            GROUP BY p.date
            ORDER BY p.date DESC
        """
        
        df = self.db.query_table(query)
        
        if df.count() == 0:
            return pd.DataFrame()
        
        return df.toPandas()
    
    def get_performance_summary(self, days_back: int = 7) -> Dict:
        """Get summarized performance metrics"""
        
        df = self.get_performance_metrics(days_back)
        
        if df.empty:
            return {}
        
        return {
            "total_requests": int(df['total_requests'].sum()),
            "avg_daily_requests": float(df['total_requests'].mean()),
            "avg_latency_seconds": float(df['avg_latency_seconds'].mean()),
            "max_latency_seconds": float(df['max_latency_seconds'].max()),
            "avg_success_rate": float(df['success_rate_pct'].mean()),
            "days_monitored": len(df)
        }