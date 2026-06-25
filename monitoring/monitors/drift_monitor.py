"""
Query distribution and drift monitoring.
"""
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient

class DriftMonitor:
    """Monitor query distribution and detect drift"""
    
    def __init__(self, db_client: DatabricksClient):
        self.db = db_client
    
    def get_query_distribution(self, days_back: int = 7) -> pd.DataFrame:
        """
        Get distribution of queries for the specified period.
        
        Returns:
            DataFrame with query frequencies
        """
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
        
        df = self.db.query_table(query)
        
        if df.count() == 0:
            return pd.DataFrame()
        
        return df.toPandas()
    
    def get_top_queries(self, days_back: int = 7, top_n: int = 20) -> pd.DataFrame:
        """
        Get the most common queries.
        
        Returns:
            DataFrame with top N queries
        """
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
        
        df = self.db.query_table(query)
        
        if df.count() == 0:
            return pd.DataFrame()
        
        return df.toPandas()
    
    def get_keyword_distribution(self, days_back: int = 7, top_n: int = 30) -> pd.DataFrame:
        """
        Analyze keyword distribution in queries.
        
        Returns:
            DataFrame with keyword frequencies
        """
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
        
        df = self.db.query_table(query)
        
        if df.count() == 0:
            return pd.DataFrame()
        
        return df.toPandas()
    
    def detect_distribution_drift(
        self, 
        baseline_days: int = 30, 
        recent_days: int = 7,
        drift_threshold: float = 0.3
    ) -> Dict:
        """
        Detect if query distribution has changed significantly.
        
        Compares recent query patterns against historical baseline.
        
        Args:
            baseline_days: Days to use as baseline (e.g., 30 days ago)
            recent_days: Recent days to compare (e.g., last 7 days)
            drift_threshold: KL divergence threshold to flag drift
        
        Returns:
            Dict with drift metrics and alerts
        """
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
        
        recent_df = self.db.query_table(recent_query).toPandas()
        
        if baseline_df.empty or recent_df.empty:
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
        
        # Calculate distributions
        baseline_total = baseline_df['frequency'].sum()
        recent_total = recent_df['frequency'].sum()
        
        baseline_df['prob'] = baseline_df['frequency'] / baseline_total
        recent_df['prob'] = recent_df['frequency'] / recent_total
        
        # Merge distributions
        merged = pd.merge(
            baseline_df[['question', 'prob']],
            recent_df[['question', 'prob']],
            on='question',
            how='outer',
            suffixes=('_baseline', '_recent')
        ).fillna(1e-10)  # Small value for smoothing
        
        # Calculate KL divergence (simplified drift metric)
        kl_div = (merged['prob_recent'] * np.log(merged['prob_recent'] / merged['prob_baseline'])).sum()
        
        # Find new queries (appeared recently but not in baseline)
        new_queries = recent_df[~recent_df['question'].isin(baseline_df['question'])]
        new_query_rate = len(new_queries) / len(recent_df) if len(recent_df) > 0 else 0
        
        # Find disappeared queries (in baseline but not recent)
        disappeared_queries = baseline_df[~baseline_df['question'].isin(recent_df['question'])]
        disappeared_query_rate = len(disappeared_queries) / len(baseline_df) if len(baseline_df) > 0 else 0
        
        drift_detected = kl_div > drift_threshold or new_query_rate > 0.3
        
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
    
    def get_query_length_stats(self, days_back: int = 30) -> Dict:
        """
        Analyze query length statistics over time.
        
        Returns:
            Dict with length statistics
        """
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
        
        result = self.db.query_table(query).first()
        
        if result is None:
            return {}
        
        return {
            "avg_length": float(result.avg_length or 0),
            "min_length": int(result.min_length or 0),
            "max_length": int(result.max_length or 0),
            "std_length": float(result.std_length or 0),
            "median_length": float(result.median_length or 0)
        }
    
    def analyze_temporal_patterns(self, days_back: int = 30) -> pd.DataFrame:
        """
        Analyze query patterns by time of day and day of week.
        
        Returns:
            DataFrame with temporal patterns
        """
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
        
        df = self.db.query_table(query)
        
        if df.count() == 0:
            return pd.DataFrame()
        
        return df.toPandas()
    
    def save_drift_metrics(self):
        """Save drift detection results for historical tracking"""
        drift_results = self.detect_distribution_drift()
        
        # Add timestamp
        drift_results['measured_at'] = datetime.now()
        
        # Convert to DataFrame
        drift_df = pd.DataFrame([drift_results])
        
        # Save to Delta table
        spark_df = self.db.spark.createDataFrame(drift_df)
        drift_table = "main.default.gdpr_agent_drift_metrics"
        self.db.write_metrics(spark_df, drift_table)
        
        print(f"✅ Saved drift metrics to {drift_table}")
