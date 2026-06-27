"""
Performance monitoring (latency, throughput, success rate).

Provides latency tracking, throughput analysis, and success rate monitoring
for GDPR Agent serving endpoints. Tracks daily performance metrics and
aggregates summary statistics for operational visibility.
"""

import logging
from typing import Dict, Any
import pandas as pd

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """
    Monitor endpoint performance metrics.
    
    Tracks latency, throughput, and success rates to ensure endpoint
    performance meets operational requirements and SLAs.
    """
    
    def __init__(self, db_client: DatabricksClient) -> None:
        """
        Initialize performance monitor.
        
        Args:
            db_client: Databricks client for data access
        """
        logger.debug("Initializing PerformanceMonitor")
        self.db = db_client
        logger.info("PerformanceMonitor initialized successfully")
    
    def get_performance_metrics(self, days_back: int = 7) -> pd.DataFrame:
        """
        Get performance metrics for the specified period.
        
        Args:
            days_back: Number of days of performance data to retrieve (default: 7)
        
        Returns:
            DataFrame with daily performance metrics including date, total_requests,
            unique_questions, avg_latency_seconds, max_latency_seconds,
            min_latency_seconds, and success_rate_pct.
            Returns empty DataFrame if no data available.
        
        Raises:
            Exception: If query execution fails (logged but not re-raised)
        """
        logger.info("Retrieving performance metrics for last %d days", days_back)
        
        try:
            query = f"""
                SELECT 
                    date,
                    COUNT(*) as total_requests,
                    COUNT(DISTINCT question) as unique_questions,
                    AVG(latency_ms / 1000.0) as avg_latency_seconds,
                    MAX(latency_ms / 1000.0) as max_latency_seconds,
                    MIN(latency_ms / 1000.0) as min_latency_seconds,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) / COUNT(*) * 100 as success_rate_pct
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {days_back}
                GROUP BY date
                ORDER BY date DESC
            """
            
            logger.debug("Executing performance metrics query on table: %s", config.INFERENCE_LOGS_TABLE)
            df = self.db.query_table(query)
            
            row_count = df.count()
            logger.debug("Query returned %d daily performance records", row_count)
            
            if row_count == 0:
                logger.warning("No performance data found for last %d days", days_back)
                return pd.DataFrame()
            
            logger.debug("Converting Spark DataFrame to pandas")
            pdf = df.toPandas()
            
            if len(pdf) > 0:
                avg_latency = pdf['avg_latency_seconds'].mean()
                avg_success_rate = pdf['success_rate_pct'].mean()
                logger.info("Performance metrics retrieved: %d days, avg latency=%.2fs, avg success rate=%.1f%%",
                           len(pdf), avg_latency, avg_success_rate)
            
            return pdf
            
        except Exception as e:
            logger.exception("Failed to retrieve performance metrics: %s", e)
            return pd.DataFrame()
    
    def get_performance_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get summarized performance metrics.
        
        Args:
            days_back: Number of days to analyze (default: 7)
        
        Returns:
            Dict with aggregated performance metrics:
            - total_requests: Total number of requests across all days
            - avg_daily_requests: Average requests per day
            - avg_latency_seconds: Average latency in seconds
            - max_latency_seconds: Maximum latency observed
            - avg_success_rate: Average success rate percentage
            - days_monitored: Number of days included in analysis
            Returns empty dict if no data available.
        
        Raises:
            Exception: If aggregation fails (logged but not re-raised)
        """
        logger.info("Generating performance summary for last %d days", days_back)
        
        try:
            logger.debug("Retrieving performance metrics for summary")
            df = self.get_performance_metrics(days_back)
            
            if df.empty:
                logger.warning("No performance data available for summary")
                return {}
            
            logger.debug("Aggregating performance metrics")
            summary = {
                "total_requests": int(df['total_requests'].sum()),
                "avg_daily_requests": float(df['total_requests'].mean()),
                "avg_latency_seconds": float(df['avg_latency_seconds'].mean()),
                "max_latency_seconds": float(df['max_latency_seconds'].max()),
                "avg_success_rate": float(df['success_rate_pct'].mean()),
                "days_monitored": len(df)
            }
            
            logger.info("Performance summary generated: %d total requests, %.2fs avg latency, %.1f%% avg success rate over %d days",
                       summary['total_requests'], summary['avg_latency_seconds'], 
                       summary['avg_success_rate'], summary['days_monitored'])
            
            return summary
            
        except KeyError as e:
            logger.exception("Missing expected column in performance summary: %s", e)
            return {}
        except Exception as e:
            logger.exception("Failed to generate performance summary: %s", e)
            return {}
