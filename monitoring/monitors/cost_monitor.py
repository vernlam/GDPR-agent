"""
Cost monitoring for OpenAI API usage.
"""
from typing import Dict
import pandas as pd
from datetime import datetime

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient
from monitoring.utils.metrics import estimate_tokens, estimate_cost

class CostMonitor:
    """Monitor and estimate OpenAI API costs"""
    
    def __init__(self, db_client: DatabricksClient):
        self.db = db_client
    
    def get_daily_costs(self, days_back: int = 30) -> pd.DataFrame:
        """
        Calculate daily cost estimates based on request/response sizes.
        
        Returns:
            DataFrame with daily cost breakdowns
        """
        query = f"""
            SELECT 
                p.date,
                COUNT(*) as request_count,
                AVG(LENGTH(get_json_object(p.request, '$.dataframe_split.data[0][0]'))) as avg_question_length,
                AVG(LENGTH(get_json_object(r.response, '$.predictions[0].answer'))) as avg_answer_length,
                SUM(LENGTH(get_json_object(p.request, '$.dataframe_split.data[0][0]'))) as total_question_chars,
                SUM(LENGTH(get_json_object(r.response, '$.predictions[0].answer'))) as total_answer_chars
            FROM {config.payload_table} p
            LEFT JOIN {config.response_table} r
                ON p.request_id = r.request_id
            WHERE p.date >= current_date() - {days_back}
              AND p.status_code = 200
            GROUP BY p.date
            ORDER BY p.date DESC
        """
        
        df = self.db.query_table(query)
        
        if df.count() == 0:
            return pd.DataFrame()
        
        pdf = df.toPandas()
        
        # Estimate tokens and costs
        # Input tokens: question (~100 chars) + context retrieval (~2000 chars) + system prompt (~500 chars)
        # Output tokens: answer length
        
        pdf['estimated_input_tokens'] = pdf.apply(
            lambda row: estimate_tokens(str(row['total_question_chars'])) + (row['request_count'] * 650),  # 650 tokens for context + system
            axis=1
        )
        
        pdf['estimated_output_tokens'] = pdf['total_answer_chars'].apply(estimate_tokens)
        
        pdf['estimated_cost_usd'] = pdf.apply(
            lambda row: estimate_cost(
                int(row['estimated_input_tokens']),
                int(row['estimated_output_tokens'])
            ),
            axis=1
        )
        
        pdf['cost_per_request'] = pdf['estimated_cost_usd'] / pdf['request_count']
        
        return pdf
    
    def get_cost_summary(self, days_back: int = 30) -> Dict:
        """
        Get summarized cost metrics.
        
        Returns:
            Dict with total costs, averages, and projections
        """
        cost_df = self.get_daily_costs(days_back)
        
        if cost_df.empty:
            return {
                "total_cost_usd": 0.0,
                "avg_daily_cost_usd": 0.0,
                "avg_cost_per_request": 0.0,
                "total_requests": 0,
                "projected_monthly_cost": 0.0,
                "days_analyzed": 0
            }
        
        total_cost = float(cost_df['estimated_cost_usd'].sum())
        total_requests = int(cost_df['request_count'].sum())
        avg_daily_cost = float(cost_df['estimated_cost_usd'].mean())
        
        return {
            "total_cost_usd": total_cost,
            "avg_daily_cost_usd": avg_daily_cost,
            "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0,
            "total_requests": total_requests,
            "projected_monthly_cost": avg_daily_cost * 30,
            "days_analyzed": len(cost_df),
            "total_input_tokens": int(cost_df['estimated_input_tokens'].sum()),
            "total_output_tokens": int(cost_df['estimated_output_tokens'].sum())
        }
    
    def get_cost_trend(self, days_back: int = 30) -> pd.DataFrame:
        """
        Analyze cost trends over time.
        
        Returns:
            DataFrame with rolling averages and trend indicators
        """
        cost_df = self.get_daily_costs(days_back)
        
        if cost_df.empty:
            return pd.DataFrame()
        
        # Sort by date
        cost_df = cost_df.sort_values('date')
        
        # Calculate rolling averages
        cost_df['rolling_7d_cost'] = cost_df['estimated_cost_usd'].rolling(window=7, min_periods=1).mean()
        cost_df['rolling_7d_requests'] = cost_df['request_count'].rolling(window=7, min_periods=1).mean()
        
        return cost_df
    
    def identify_cost_anomalies(self, days_back: int = 30, std_threshold: float = 2.0) -> pd.DataFrame:
        """
        Identify days with abnormally high costs.
        
        Args:
            days_back: Number of days to analyze
            std_threshold: Standard deviations from mean to flag as anomaly
        
        Returns:
            DataFrame with anomalous cost days
        """
        cost_df = self.get_daily_costs(days_back)
        
        if cost_df.empty or len(cost_df) < 3:
            return pd.DataFrame()
        
        mean_cost = cost_df['estimated_cost_usd'].mean()
        std_cost = cost_df['estimated_cost_usd'].std()
        
        # Flag anomalies
        cost_df['is_anomaly'] = cost_df['estimated_cost_usd'] > (mean_cost + std_threshold * std_cost)
        cost_df['deviation_from_mean'] = (cost_df['estimated_cost_usd'] - mean_cost) / std_cost
        
        anomalies = cost_df[cost_df['is_anomaly']]
        
        return anomalies[['date', 'request_count', 'estimated_cost_usd', 'cost_per_request', 'deviation_from_mean']]
    
    def save_cost_metrics(self, days_back: int = 1):
        """Save cost metrics to Delta table for historical tracking"""
        cost_df = self.get_daily_costs(days_back)
        
        if cost_df.empty:
            print("No cost data to save")
            return
        
        # Add metadata
        cost_df['calculated_at'] = datetime.now()
        cost_df['model_used'] = 'gpt-4o-mini'  # Adjust based on actual model
        
        # Convert to Spark DataFrame and save
        spark_df = self.db.spark.createDataFrame(cost_df)
        
        # Use a dedicated cost metrics table
        cost_table = "main.default.gdpr_agent_cost_metrics"
        self.db.write_metrics(spark_df, cost_table)
        
        print(f"✅ Saved cost metrics to {cost_table}")