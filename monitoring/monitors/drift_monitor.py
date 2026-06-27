"""
Query distribution and drift monitoring.

Provides distribution analysis, drift detection, and temporal pattern tracking
for GDPR Agent query patterns. Identifies shifts in user behavior and query
characteristics over time.
"""

import logging
from typing import Dict, List, Tuple, Any
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class DriftMonitor:
    """
    Monitor query distribution and detect drift.
    
    Tracks query patterns, keyword distribution, and temporal characteristics
    to identify changes in user behavior and query composition.
    """
    
    def __init__(self, db_client: DatabricksClient) -> None:
        """
        Initialize drift monitor.
        
        Args:
            db_client: Databricks client for data access
        """
        logger.debug("Initializing DriftMonitor")
        self.db = db_client
        logger.info("DriftMonitor initialized successfully")
    
    def get_query_distribution(self, days_back: int = 7) -> pd.DataFrame:
        """
        Get distribution of queries for the specified period.
        
        Args:
            days_back: Number of days of query data to retrieve (default: 7)
        
        Returns:
            DataFrame with query frequencies, first seen date, and last seen date.
            Returns empty DataFrame if no data available.
        
        Raises:
            Exception: If query execution fails (logged but not re-raised)
        """
        logger.info("Retrieving query distribution for last %d days", days_back)
        
        try:
            query = f"""
                SELECT 
                    question,
                    COUNT(*) as frequency,
                    MIN(date) as first_seen,
                    MAX(date) as last_seen
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {days_back}
                GROUP BY question
                ORDER BY frequency DESC
            """
            
            logger.debug("Executing query distribution query on table: %s", config.INFERENCE_LOGS_TABLE)
            df = self.db.query_table(query)
            
            row_count = df.count()
            logger.debug("Query returned %d unique questions", row_count)
            
            if row_count == 0:
                logger.warning("No query distribution data found for last %d days", days_back)
                return pd.DataFrame()
            
            logger.debug("Converting Spark DataFrame to pandas")
            pdf = df.toPandas()
            
            logger.info("Query distribution retrieved: %d unique questions over %d days", len(pdf), days_back)
            
            return pdf
            
        except Exception as e:
            logger.exception("Failed to retrieve query distribution: %s", e)
            return pd.DataFrame()
    
    def get_top_queries(self, days_back: int = 7, top_n: int = 20) -> pd.DataFrame:
        """
        Get the most common queries.
        
        Args:
            days_back: Number of days to analyze (default: 7)
            top_n: Number of top queries to return (default: 20)
        
        Returns:
            DataFrame with top N queries including frequency and percentage.
            Returns empty DataFrame if no data available.
        
        Raises:
            Exception: If query execution fails (logged but not re-raised)
        """
        logger.info("Retrieving top %d queries for last %d days", top_n, days_back)
        
        try:
            query = f"""
                SELECT 
                    question,
                    COUNT(*) as frequency,
                    COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {days_back}
                GROUP BY question
                ORDER BY frequency DESC
                LIMIT {top_n}
            """
            
            logger.debug("Executing top queries query on table: %s", config.INFERENCE_LOGS_TABLE)
            df = self.db.query_table(query)
            
            row_count = df.count()
            logger.debug("Query returned %d top questions", row_count)
            
            if row_count == 0:
                logger.warning("No query data found for top queries analysis (last %d days)", days_back)
                return pd.DataFrame()
            
            logger.debug("Converting Spark DataFrame to pandas")
            pdf = df.toPandas()
            
            if len(pdf) > 0:
                top_percentage = pdf['percentage'].iloc[0]
                logger.info("Top queries retrieved: %d questions, top query represents %.1f%% of traffic", 
                           len(pdf), top_percentage)
            
            return pdf
            
        except Exception as e:
            logger.exception("Failed to retrieve top queries: %s", e)
            return pd.DataFrame()
    
    def get_keyword_distribution(self, days_back: int = 7, top_n: int = 30) -> pd.DataFrame:
        """
        Analyze keyword distribution in queries.
        
        Extracts and ranks keywords from user queries, filtering out common
        stop words and short terms.
        
        Args:
            days_back: Number of days to analyze (default: 7)
            top_n: Number of top keywords to return (default: 30)
        
        Returns:
            DataFrame with keyword frequencies and days appeared.
            Returns empty DataFrame if no data available.
        
        Raises:
            Exception: If query execution fails (logged but not re-raised)
        """
        logger.info("Analyzing keyword distribution for last %d days (top %d keywords)", days_back, top_n)
        
        try:
            query = f"""
                SELECT 
                    keyword,
                    COUNT(*) as frequency,
                    COUNT(DISTINCT date) as days_appeared
                FROM (
                    SELECT 
                        date,
                        explode(split(lower(question), ' ')) as keyword
                    FROM {config.INFERENCE_LOGS_TABLE}
                    WHERE date >= current_date() - {days_back}
                )
                WHERE LENGTH(keyword) > 4
                  AND keyword NOT RLIKE '[^a-z]'
                  AND keyword NOT IN ('what', 'when', 'where', 'which', 'should', 'would', 'could', 
                                       'does', 'have', 'about', 'under', 'their', 'there', 'these', 
                                       'those', 'with', 'from', 'into', 'that', 'this')
                GROUP BY keyword
                ORDER BY frequency DESC
                LIMIT {top_n}
            """
            
            logger.debug("Executing keyword distribution query on table: %s", config.INFERENCE_LOGS_TABLE)
            df = self.db.query_table(query)
            
            row_count = df.count()
            logger.debug("Query returned %d keywords", row_count)
            
            if row_count == 0:
                logger.warning("No keyword data found for last %d days", days_back)
                return pd.DataFrame()
            
            logger.debug("Converting Spark DataFrame to pandas")
            pdf = df.toPandas()
            
            if len(pdf) > 0:
                top_keyword = pdf['keyword'].iloc[0]
                top_freq = pdf['frequency'].iloc[0]
                logger.info("Keyword distribution retrieved: %d keywords, top keyword '%s' appears %d times", 
                           len(pdf), top_keyword, top_freq)
            
            return pdf
            
        except Exception as e:
            logger.exception("Failed to analyze keyword distribution: %s", e)
            return pd.DataFrame()
    
    def detect_distribution_drift(
        self, 
        baseline_days: int = 30, 
        recent_days: int = 7,
        drift_threshold: float = 0.3
    ) -> Dict[str, Any]:
        """
        Detect if query distribution has changed significantly.
        
        Compares recent query patterns against historical baseline using
        KL divergence and new/disappeared query metrics.
        
        Args:
            baseline_days: Days to use as baseline (e.g., 30 days ago) (default: 30)
            recent_days: Recent days to compare (e.g., last 7 days) (default: 7)
            drift_threshold: KL divergence threshold to flag drift (default: 0.3)
        
        Returns:
            Dict with drift detection results including:
            - drift_detected: Boolean indicating if drift was detected
            - drift_score: KL divergence score
            - drift_threshold: Threshold used for detection
            - new_query_count: Number of queries in recent but not baseline
            - new_query_rate: Percentage of queries that are new
            - disappeared_query_count: Number of queries in baseline but not recent
            - disappeared_query_rate: Percentage of queries that disappeared
            - baseline_unique_queries: Unique query count in baseline period
            - recent_unique_queries: Unique query count in recent period
            - message: Summary message with key metrics
        
        Raises:
            Exception: If drift detection calculation fails (logged but not re-raised)
        """
        logger.info("Detecting distribution drift: baseline=%d days, recent=%d days, threshold=%.2f",
                   baseline_days, recent_days, drift_threshold)
        
        try:
            # Get baseline distribution (older period)
            baseline_query = f"""
                SELECT 
                    question,
                    COUNT(*) as frequency
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {baseline_days}
                  AND date < current_date() - {recent_days}
                GROUP BY question
            """
            
            logger.debug("Retrieving baseline distribution from table: %s", config.INFERENCE_LOGS_TABLE)
            baseline_df = self.db.query_table(baseline_query).toPandas()
            
            # Get recent distribution
            recent_query = f"""
                SELECT 
                    question,
                    COUNT(*) as frequency
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {recent_days}
                GROUP BY question
            """
            
            logger.debug("Retrieving recent distribution from table: %s", config.INFERENCE_LOGS_TABLE)
            recent_df = self.db.query_table(recent_query).toPandas()
            
            logger.debug("Baseline period: %d unique queries, Recent period: %d unique queries",
                        len(baseline_df), len(recent_df))
            
            if baseline_df.empty or recent_df.empty:
                logger.warning("Insufficient data for drift detection: baseline=%d queries, recent=%d queries",
                             len(baseline_df), len(recent_df))
                return {
                    "drift_detected": False,
                    "drift_score": 0.0,
                    "drift_threshold": drift_threshold,
                    "new_query_count": 0,
                    "new_query_rate": 0.0,
                    "disappeared_query_count": 0,
                    "disappeared_query_rate": 0.0,
                    "baseline_unique_queries": len(baseline_df),
                    "recent_unique_queries": len(recent_df),
                    "message": "Insufficient data for drift detection"
                }
            
            logger.debug("Calculating probability distributions")
            
            # Calculate distributions
            baseline_total = baseline_df['frequency'].sum()
            recent_total = recent_df['frequency'].sum()
            
            baseline_df['prob'] = baseline_df['frequency'] / baseline_total
            recent_df['prob'] = recent_df['frequency'] / recent_total
            
            # Merge distributions
            logger.debug("Merging baseline and recent distributions")
            merged = pd.merge(
                baseline_df[['question', 'prob']],
                recent_df[['question', 'prob']],
                on='question',
                how='outer',
                suffixes=('_baseline', '_recent')
            ).fillna(1e-10)  # Small value for smoothing
            
            # Calculate KL divergence (simplified drift metric)
            logger.debug("Calculating KL divergence")
            kl_div = (merged['prob_recent'] * np.log(merged['prob_recent'] / merged['prob_baseline'])).sum()
            
            # Find new queries (appeared recently but not in baseline)
            logger.debug("Identifying new and disappeared queries")
            new_queries = recent_df[~recent_df['question'].isin(baseline_df['question'])]
            new_query_rate = len(new_queries) / len(recent_df) if len(recent_df) > 0 else 0
            
            # Find disappeared queries (in baseline but not recent)
            disappeared_queries = baseline_df[~baseline_df['question'].isin(recent_df['question'])]
            disappeared_query_rate = len(disappeared_queries) / len(baseline_df) if len(baseline_df) > 0 else 0
            
            drift_detected = kl_div > drift_threshold or new_query_rate > 0.3
            
            log_level = logging.WARNING if drift_detected else logging.INFO
            logger.log(log_level, 
                      "Drift detection complete: drift_detected=%s, KL_divergence=%.3f, new_queries=%d (%.1f%%), disappeared=%d (%.1f%%)",
                      drift_detected, kl_div, len(new_queries), new_query_rate * 100,
                      len(disappeared_queries), disappeared_query_rate * 100)
            
            return {
                "drift_detected": bool(drift_detected),
                "drift_score": float(kl_div),
                "drift_threshold": drift_threshold,
                "new_query_count": len(new_queries),
                "new_query_rate": float(new_query_rate),
                "disappeared_query_count": len(disappeared_queries),
                "disappeared_query_rate": float(disappeared_query_rate),
                "baseline_unique_queries": len(baseline_df),
                "recent_unique_queries": len(recent_df),
                "message": f"KL divergence: {kl_div:.3f}, New queries: {len(new_queries)}, Disappeared: {len(disappeared_queries)}"
            }
            
        except KeyError as e:
            logger.exception("Missing expected column in drift detection: %s", e)
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "drift_threshold": drift_threshold,
                "new_query_count": 0,
                "new_query_rate": 0.0,
                "disappeared_query_count": 0,
                "disappeared_query_rate": 0.0,
                "baseline_unique_queries": 0,
                "recent_unique_queries": 0,
                "message": "Error calculating drift metrics"
            }
        except Exception as e:
            logger.exception("Failed to detect distribution drift: %s", e)
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "drift_threshold": drift_threshold,
                "new_query_count": 0,
                "new_query_rate": 0.0,
                "disappeared_query_count": 0,
                "disappeared_query_rate": 0.0,
                "baseline_unique_queries": 0,
                "recent_unique_queries": 0,
                "message": "Error calculating drift metrics"
            }
    
    def get_query_length_stats(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Analyze query length statistics over time.
        
        Args:
            days_back: Number of days to analyze (default: 30)
        
        Returns:
            Dict with length statistics including:
            - avg_length: Average query length in characters
            - min_length: Minimum query length
            - max_length: Maximum query length
            - std_length: Standard deviation of query length
            - median_length: Median query length
            Returns empty dict if no data available.
        
        Raises:
            Exception: If query execution fails (logged but not re-raised)
        """
        logger.info("Analyzing query length statistics for last %d days", days_back)
        
        try:
            query = f"""
                SELECT 
                    AVG(LENGTH(question)) as avg_length,
                    MIN(LENGTH(question)) as min_length,
                    MAX(LENGTH(question)) as max_length,
                    STDDEV(LENGTH(question)) as std_length,
                    percentile(LENGTH(question), 0.5) as median_length
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {days_back}
            """
            
            logger.debug("Executing query length statistics query on table: %s", config.INFERENCE_LOGS_TABLE)
            result = self.db.query_table(query).first()
            
            if result is None:
                logger.warning("No query length data found for last %d days", days_back)
                return {}
            
            stats = {
                "avg_length": float(result.avg_length or 0),
                "min_length": int(result.min_length or 0),
                "max_length": int(result.max_length or 0),
                "std_length": float(result.std_length or 0),
                "median_length": float(result.median_length or 0)
            }
            
            logger.info("Query length stats calculated: avg=%.1f, min=%d, max=%d, median=%.1f",
                       stats['avg_length'], stats['min_length'], stats['max_length'], stats['median_length'])
            
            return stats
            
        except AttributeError as e:
            logger.exception("Error accessing result attributes in query length stats: %s", e)
            return {}
        except Exception as e:
            logger.exception("Failed to calculate query length statistics: %s", e)
            return {}
    
    def analyze_temporal_patterns(self, days_back: int = 30) -> pd.DataFrame:
        """
        Analyze query patterns by time of day and day of week.
        
        Args:
            days_back: Number of days to analyze (default: 30)
        
        Returns:
            DataFrame with temporal patterns including date, hour of day,
            day of week, and request count. Returns empty DataFrame if no data.
        
        Raises:
            Exception: If query execution fails (logged but not re-raised)
        """
        logger.info("Analyzing temporal query patterns for last %d days", days_back)
        
        try:
            query = f"""
                SELECT 
                    date,
                    hour(timestamp) as hour_of_day,
                    dayofweek(date) as day_of_week,
                    COUNT(*) as request_count
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {days_back}
                GROUP BY date, hour_of_day, day_of_week
                ORDER BY date DESC, hour_of_day
            """
            
            logger.debug("Executing temporal patterns query on table: %s", config.INFERENCE_LOGS_TABLE)
            df = self.db.query_table(query)
            
            row_count = df.count()
            logger.debug("Query returned %d temporal pattern rows", row_count)
            
            if row_count == 0:
                logger.warning("No temporal pattern data found for last %d days", days_back)
                return pd.DataFrame()
            
            logger.debug("Converting Spark DataFrame to pandas")
            pdf = df.toPandas()
            
            logger.info("Temporal patterns analyzed: %d data points over %d days", len(pdf), days_back)
            
            return pdf
            
        except Exception as e:
            logger.exception("Failed to analyze temporal patterns: %s", e)
            return pd.DataFrame()
    
    def save_drift_metrics(self) -> None:
        """
        Save drift detection results for historical tracking.
        
        Executes drift detection and persists results to Delta table
        for long-term monitoring and trend analysis.
        
        Returns:
            None
        
        Raises:
            Exception: If saving drift metrics fails (logged but not re-raised)
        """
        logger.info("Saving drift metrics to Delta table")
        
        try:
            logger.debug("Executing drift detection")
            drift_results = self.detect_distribution_drift()
            
            if not drift_results or drift_results.get('message') == 'Insufficient data for drift detection':
                logger.warning("Skipping drift metrics save: insufficient data for detection")
                return
            
            logger.debug("Adding metadata to drift results")
            drift_results['measured_at'] = datetime.now()
            
            logger.debug("Converting drift results to DataFrame")
            drift_df = pd.DataFrame([drift_results])
            
            logger.debug("Converting pandas DataFrame to Spark DataFrame")
            spark_df = self.db.spark.createDataFrame(drift_df)
            
            drift_table = "main.default.gdpr_agent_drift_metrics"
            logger.debug("Writing drift metrics to table: %s", drift_table)
            self.db.write_metrics(spark_df, drift_table)
            
            logger.info("Drift metrics saved successfully to %s (drift_detected=%s, drift_score=%.3f)",
                       drift_table, drift_results['drift_detected'], drift_results['drift_score'])
            
        except AttributeError as e:
            logger.exception("Databricks client attribute error saving drift metrics: %s", e)
        except ValueError as e:
            logger.exception("Invalid data format for Spark DataFrame conversion: %s", e)
        except Exception as e:
            logger.exception("Failed to save drift metrics: %s", e)
