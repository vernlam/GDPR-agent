"""
Error monitoring and analysis.
"""
from typing import Dict, List
import pandas as pd
from datetime import datetime

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient

class ErrorMonitor:
    """Monitor and analyze errors from the serving endpoint"""
    
    def __init__(self, db_client: DatabricksClient):
        self.db = db_client
    
    def get_errors(self, days_back: int = 7) -> pd.DataFrame:
        """
        Get all errors from the specified period.
        
        Returns:
            DataFrame with error details
        """
        query = f"""
            SELECT 
                p.date,
                p.timestamp_ms,
                p.request_id,
                p.status_code,
                get_json_object(p.request, '$.dataframe_split.data[0][0]') as question,
                get_json_object(r.response, '$.predictions[0].answer') as answer,
                CASE 
                    WHEN p.status_code != 200 THEN 'HTTP_ERROR'
                    WHEN get_json_object(r.response, '$.predictions[0].answer') LIKE '%Error%' THEN 'AGENT_ERROR'
                    WHEN get_json_object(r.response, '$.predictions[0].answer') LIKE '%Exception%' THEN 'AGENT_EXCEPTION'
                    ELSE 'UNKNOWN'
                END as error_type
            FROM {config.payload_table} p
            LEFT JOIN {config.response_table} r
                ON p.request_id = r.request_id
            WHERE p.date >= current_date() - {days_back}
              AND (
                  p.status_code != 200 
                  OR get_json_object(r.response, '$.predictions[0].answer') LIKE '%Error%'
                  OR get_json_object(r.response, '$.predictions[0].answer') LIKE '%Exception%'
              )
            ORDER BY p.timestamp_ms DESC
        """
        
        df = self.db.query_table(query)
        
        if df.count() == 0:
            return pd.DataFrame()
        
        return df.toPandas()
    
    def get_error_summary(self, days_back: int = 7) -> Dict:
        """
        Get summarized error metrics.
        
        Returns:
            Dict with error counts, rates, and patterns
        """
        # Get total requests
        total_query = f"""
            SELECT COUNT(*) as total_requests
            FROM {config.payload_table}
            WHERE date >= current_date() - {days_back}
        """
        total_result = self.db.query_table(total_query).first()
        total_requests = int(total_result.total_requests) if total_result else 0
        
        # Get error counts
        error_query = f"""
            SELECT 
                COUNT(*) as total_errors,
                COUNT(DISTINCT date) as days_with_errors,
                CASE 
                    WHEN p.status_code != 200 THEN 'HTTP_ERROR'
                    WHEN get_json_object(r.response, '$.predictions[0].answer') LIKE '%Error%' THEN 'AGENT_ERROR'
                    ELSE 'OTHER'
                END as error_type,
                COUNT(*) as error_count
            FROM {config.payload_table} p
            LEFT JOIN {config.response_table} r
                ON p.request_id = r.request_id
            WHERE p.date >= current_date() - {days_back}
              AND (
                  p.status_code != 200 
                  OR get_json_object(r.response, '$.predictions[0].answer') LIKE '%Error%'
              )
            GROUP BY error_type
        """
        
        error_df = self.db.query_table(error_query)
        
        if error_df.count() == 0:
            return {
                "total_requests": total_requests,
                "total_errors": 0,
                "error_rate": 0.0,
                "errors_by_type": {},
                "days_with_errors": 0
            }
        
        error_pdf = error_df.toPandas()
        total_errors = int(error_pdf['error_count'].sum())
        
        return {
            "total_requests": total_requests,
            "total_errors": total_errors,
            "error_rate": (total_errors / total_requests * 100) if total_requests > 0 else 0,
            "errors_by_type": dict(zip(error_pdf['error_type'], error_pdf['error_count'])),
            "days_with_errors": int(error_pdf['days_with_errors'].max()) if len(error_pdf) > 0 else 0
        }
    
    def get_error_patterns(self, days_back: int = 7, top_n: int = 10) -> pd.DataFrame:
        """
        Identify common error patterns.
        
        Returns:
            DataFrame with most frequent error patterns
        """
        query = f"""
            SELECT 
                get_json_object(p.request, '$.dataframe_split.data[0][0]') as question,
                get_json_object(r.response, '$.predictions[0].answer') as error_message,
                COUNT(*) as occurrence_count,
                MAX(p.date) as last_occurred
            FROM {config.payload_table} p
            LEFT JOIN {config.response_table} r
                ON p.request_id = r.request_id
            WHERE p.date >= current_date() - {days_back}
              AND (
                  p.status_code != 200 
                  OR get_json_object(r.response, '$.predictions[0].answer') LIKE '%Error%'
              )
            GROUP BY question, error_message
            ORDER BY occurrence_count DESC
            LIMIT {top_n}
        """
        
        df = self.db.query_table(query)
        
        if df.count() == 0:
            return pd.DataFrame()
        
        return df.toPandas()
    
    def log_errors_to_table(self, days_back: int = 1):
        """
        Store error logs in dedicated error table for long-term tracking.
        """
        errors_df = self.get_errors(days_back)
        
        if errors_df.empty:
            print("No errors to log")
            return
        
        # Add logging timestamp
        errors_df['logged_at'] = datetime.now()
        
        # Convert to Spark DataFrame and save
        spark_df = self.db.spark.createDataFrame(errors_df)
        self.db.write_metrics(spark_df, config.ERROR_LOG_TABLE)
        
        print(f"✅ Logged {len(errors_df)} errors to {config.ERROR_LOG_TABLE}")
    
    def check_alert_thresholds(self, days_back: int = 1) -> List[Dict]:
        """
        Check if errors exceed alert thresholds.
        
        Returns:
            List of alert dictionaries if thresholds are breached
        """
        alerts = []
        summary = self.get_error_summary(days_back)
        
        # Check daily error count
        if summary['total_errors'] > config.MAX_ERRORS_PER_DAY:
            alerts.append({
                "severity": "HIGH",
                "metric": "daily_error_count",
                "threshold": config.MAX_ERRORS_PER_DAY,
                "actual": summary['total_errors'],
                "message": f"Daily error count ({summary['total_errors']}) exceeds threshold ({config.MAX_ERRORS_PER_DAY})"
            })
        
        # Check error rate
        if summary['error_rate'] > (100 - config.MIN_SUCCESS_RATE):
            alerts.append({
                "severity": "HIGH",
                "metric": "error_rate",
                "threshold": 100 - config.MIN_SUCCESS_RATE,
                "actual": summary['error_rate'],
                "message": f"Error rate ({summary['error_rate']:.1f}%) is too high"
            })
        
        return alerts