"""
Error monitoring and analysis.

Provides error detection, pattern analysis, and alerting for GDPR Agent
serving endpoints. Tracks error rates, categorizes failure types, and
identifies recurring error patterns for operational monitoring.
"""

import logging
from typing import Dict, List, Any
import pandas as pd
from datetime import datetime

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class ErrorMonitor:
    """
    Monitor and analyze errors from the serving endpoint.
    
    Tracks error occurrences, analyzes patterns, and checks alert thresholds
    to ensure system reliability and rapid incident detection.
    """
    
    def __init__(self, db_client: DatabricksClient) -> None:
        """
        Initialize error monitor.
        
        Args:
            db_client: Databricks client for data access
        """
        logger.debug("Initializing ErrorMonitor")
        self.db = db_client
        logger.info("ErrorMonitor initialized successfully")
    
    def get_errors(self, days_back: int = 7) -> pd.DataFrame:
        """
        Get all errors from the specified period.
        
        Args:
            days_back: Number of days of error data to retrieve (default: 7)
        
        Returns:
            DataFrame with error details including date, timestamp, request_id,
            question, answer, error_message, status, and error_type.
            Returns empty DataFrame if no errors found.
        
        Raises:
            Exception: If query execution fails (logged but not re-raised)
        """
        logger.info("Retrieving errors for last %d days", days_back)
        
        try:
            query = f"""
                SELECT 
                    date,
                    timestamp,
                    request_id,
                    question,
                    answer,
                    error_message,
                    status,
                    CASE 
                        WHEN status = 'error' THEN 'HTTP_ERROR'
                        WHEN status = 'exception' THEN 'AGENT_EXCEPTION'
                        WHEN answer LIKE '%Error%' THEN 'AGENT_ERROR'
                        ELSE 'UNKNOWN'
                    END as error_type
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {days_back}
                  AND status != 'success'
                ORDER BY timestamp DESC
            """
            
            logger.debug("Executing error query on table: %s", config.INFERENCE_LOGS_TABLE)
            df = self.db.query_table(query)
            
            error_count = df.count()
            logger.debug("Query returned %d error records", error_count)
            
            if error_count == 0:
                logger.info("No errors found for last %d days", days_back)
                return pd.DataFrame()
            
            logger.debug("Converting Spark DataFrame to pandas")
            pdf = df.toPandas()
            
            logger.info("Errors retrieved: %d error records over %d days", len(pdf), days_back)
            
            return pdf
            
        except Exception as e:
            logger.exception("Failed to retrieve errors: %s", e)
            return pd.DataFrame()
    
    def get_error_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get summarized error metrics.
        
        Args:
            days_back: Number of days to analyze (default: 7)
        
        Returns:
            Dict with error counts, rates, and patterns:
            - total_requests: Total number of requests
            - total_errors: Total number of errors
            - error_rate: Error rate as percentage
            - errors_by_type: Dict mapping error types to counts
            - days_with_errors: Number of days that had errors
        
        Raises:
            Exception: If query execution or aggregation fails (logged but not re-raised)
        """
        logger.info("Generating error summary for last %d days", days_back)
        
        try:
            # Get total requests
            logger.debug("Querying total request count")
            total_query = f"""
                SELECT COUNT(*) as total_requests
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {days_back}
            """
            total_result = self.db.query_table(total_query).first()
            total_requests = int(total_result.total_requests) if total_result else 0
            
            logger.debug("Total requests found: %d", total_requests)
            
            # Get error counts
            logger.debug("Querying error counts by type")
            error_query = f"""
                SELECT 
                    COUNT(*) as total_errors,
                    COUNT(DISTINCT date) as days_with_errors,
                    CASE 
                        WHEN status = 'error' THEN 'HTTP_ERROR'
                        WHEN status = 'exception' THEN 'AGENT_EXCEPTION'
                        ELSE 'OTHER'
                    END as error_type,
                    COUNT(*) as error_count
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {days_back}
                  AND status != 'success'
                GROUP BY error_type
            """
            
            error_df = self.db.query_table(error_query)
            
            if error_df.count() == 0:
                logger.info("No errors found in summary period")
                return {
                    "total_requests": total_requests,
                    "total_errors": 0,
                    "error_rate": 0.0,
                    "errors_by_type": {},
                    "days_with_errors": 0
                }
            
            logger.debug("Converting error DataFrame to pandas")
            error_pdf = error_df.toPandas()
            total_errors = int(error_pdf['error_count'].sum())
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0.0
            
            summary = {
                "total_requests": total_requests,
                "total_errors": total_errors,
                "error_rate": error_rate,
                "errors_by_type": dict(zip(error_pdf['error_type'], error_pdf['error_count'])),
                "days_with_errors": int(error_pdf['days_with_errors'].max()) if len(error_pdf) > 0 else 0
            }
            
            logger.info("Error summary generated: %d errors out of %d requests (%.2f%% error rate) over %d days",
                       total_errors, total_requests, error_rate, summary['days_with_errors'])
            logger.debug("Errors by type: %s", summary['errors_by_type'])
            
            return summary
            
        except AttributeError as e:
            logger.exception("Error accessing result attributes in error summary: %s", e)
            return {
                "total_requests": 0,
                "total_errors": 0,
                "error_rate": 0.0,
                "errors_by_type": {},
                "days_with_errors": 0
            }
        except KeyError as e:
            logger.exception("Missing expected column in error summary: %s", e)
            return {
                "total_requests": 0,
                "total_errors": 0,
                "error_rate": 0.0,
                "errors_by_type": {},
                "days_with_errors": 0
            }
        except Exception as e:
            logger.exception("Failed to generate error summary: %s", e)
            return {
                "total_requests": 0,
                "total_errors": 0,
                "error_rate": 0.0,
                "errors_by_type": {},
                "days_with_errors": 0
            }
    
    def get_error_patterns(self, days_back: int = 7, top_n: int = 10) -> pd.DataFrame:
        """
        Identify common error patterns.
        
        Args:
            days_back: Number of days to analyze (default: 7)
            top_n: Number of top error patterns to return (default: 10)
        
        Returns:
            DataFrame with most frequent error patterns including question,
            error_message, occurrence_count, and last_occurred.
            Returns empty DataFrame if no errors found.
        
        Raises:
            Exception: If query execution fails (logged but not re-raised)
        """
        logger.info("Identifying error patterns for last %d days (top %d patterns)", days_back, top_n)
        
        try:
            query = f"""
                SELECT 
                    question,
                    error_message,
                    COUNT(*) as occurrence_count,
                    MAX(date) as last_occurred
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {days_back}
                  AND status != 'success'
                GROUP BY question, error_message
                ORDER BY occurrence_count DESC
                LIMIT {top_n}
            """
            
            logger.debug("Executing error pattern query on table: %s", config.INFERENCE_LOGS_TABLE)
            df = self.db.query_table(query)
            
            pattern_count = df.count()
            logger.debug("Query returned %d error patterns", pattern_count)
            
            if pattern_count == 0:
                logger.info("No error patterns found for last %d days", days_back)
                return pd.DataFrame()
            
            logger.debug("Converting Spark DataFrame to pandas")
            pdf = df.toPandas()
            
            if len(pdf) > 0:
                top_pattern_count = pdf['occurrence_count'].iloc[0]
                logger.info("Error patterns retrieved: %d patterns, top pattern occurred %d times",
                           len(pdf), top_pattern_count)
                logger.debug("Most frequent error: %s", pdf['error_message'].iloc[0][:100])
            
            return pdf
            
        except Exception as e:
            logger.exception("Failed to identify error patterns: %s", e)
            return pd.DataFrame()
    
    def log_errors_to_table(self, days_back: int = 1) -> None:
        """
        Store error logs in dedicated error table for long-term tracking.
        
        Args:
            days_back: Number of days of errors to log (default: 1)
        
        Returns:
            None
        
        Raises:
            Exception: If logging errors to table fails (logged but not re-raised)
        """
        logger.info("Logging errors to table for last %d days", days_back)
        
        try:
            logger.debug("Retrieving errors to log")
            errors_df = self.get_errors(days_back)
            
            if errors_df.empty:
                logger.info("No errors to log for period")
                return
            
            logger.debug("Adding metadata to error records")
            errors_df['logged_at'] = datetime.now()
            
            logger.debug("Converting pandas DataFrame to Spark DataFrame")
            spark_df = self.db.spark.createDataFrame(errors_df)
            
            logger.debug("Writing error records to table: %s", config.ERROR_LOG_TABLE)
            self.db.write_metrics(spark_df, config.ERROR_LOG_TABLE)
            
            logger.info("Successfully logged %d error records to %s", len(errors_df), config.ERROR_LOG_TABLE)
            
        except AttributeError as e:
            logger.exception("Databricks client attribute error logging errors: %s", e)
        except ValueError as e:
            logger.exception("Invalid data format for Spark DataFrame conversion: %s", e)
        except Exception as e:
            logger.exception("Failed to log errors to table: %s", e)
    
    def check_alert_thresholds(self, days_back: int = 1) -> List[Dict[str, Any]]:
        """
        Check if errors exceed alert thresholds.
        
        Args:
            days_back: Number of days to check thresholds for (default: 1)
        
        Returns:
            List of alert dictionaries if thresholds are breached. Each alert contains:
            - severity: Alert severity level (e.g., "HIGH")
            - metric: Metric that breached threshold
            - threshold: Configured threshold value
            - actual: Actual observed value
            - message: Human-readable alert message
            Returns empty list if no thresholds breached.
        
        Raises:
            Exception: If threshold check fails (logged but not re-raised)
        """
        logger.info("Checking alert thresholds for last %d days", days_back)
        
        try:
            alerts = []
            
            logger.debug("Retrieving error summary for threshold checks")
            summary = self.get_error_summary(days_back)
            
            # Check daily error count
            logger.debug("Checking daily error count: %d (threshold: %d)",
                        summary['total_errors'], config.MAX_ERRORS_PER_DAY)
            if summary['total_errors'] > config.MAX_ERRORS_PER_DAY:
                alert = {
                    "severity": "HIGH",
                    "metric": "daily_error_count",
                    "threshold": config.MAX_ERRORS_PER_DAY,
                    "actual": summary['total_errors'],
                    "message": f"Daily error count ({summary['total_errors']}) exceeds threshold ({config.MAX_ERRORS_PER_DAY})"
                }
                alerts.append(alert)
                logger.warning("Alert triggered: %s", alert['message'])
            
            # Check error rate
            error_rate_threshold = 100 - config.MIN_SUCCESS_RATE
            logger.debug("Checking error rate: %.2f%% (threshold: %.2f%%)",
                        summary['error_rate'], error_rate_threshold)
            if summary['error_rate'] > error_rate_threshold:
                alert = {
                    "severity": "HIGH",
                    "metric": "error_rate",
                    "threshold": error_rate_threshold,
                    "actual": summary['error_rate'],
                    "message": f"Error rate ({summary['error_rate']:.1f}%) is too high"
                }
                alerts.append(alert)
                logger.warning("Alert triggered: %s", alert['message'])
            
            if len(alerts) == 0:
                logger.info("No alert thresholds breached")
            else:
                logger.warning("Alert check complete: %d alerts triggered", len(alerts))
            
            return alerts
            
        except KeyError as e:
            logger.exception("Missing expected key in alert threshold check: %s", e)
            return []
        except Exception as e:
            logger.exception("Failed to check alert thresholds: %s", e)
            return []
