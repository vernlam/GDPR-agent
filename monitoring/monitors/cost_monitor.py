"""
Cost monitoring for OpenAI API usage.

Provides cost estimation, tracking, and anomaly detection for GDPR Agent
serving endpoints using OpenAI models. Estimates token usage and costs
based on request/response character counts and historical data.
"""

import logging
from typing import Dict, Optional
from datetime import datetime

import pandas as pd

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient
from monitoring.utils.metrics import estimate_tokens, estimate_cost

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class CostMonitor:
    """
    Monitor and estimate OpenAI API costs.
    
    Tracks daily costs, identifies anomalies, and provides cost projections
    based on inference log data.
    """
    
    def __init__(self, db_client: DatabricksClient):
        """
        Initialize cost monitor.
        
        Args:
            db_client: Databricks client for data access
        """
        logger.debug("Initializing CostMonitor")
        self.db = db_client
        logger.info("CostMonitor initialized successfully")
    
    def get_daily_costs(self, days_back: int = 30) -> pd.DataFrame:
        """
        Calculate daily cost estimates based on request/response sizes.
        
        Args:
            days_back: Number of days of historical data to retrieve (default: 30)
        
        Returns:
            DataFrame with daily cost breakdowns including request counts,
            token estimates, and cost estimates. Returns empty DataFrame if no data.
        
        Raises:
            Exception: If query execution or data processing fails (logged but not re-raised)
        """
        logger.info("Calculating daily costs for last %d days", days_back)
        
        try:
            query = f"""
                SELECT 
                    date,
                    COUNT(*) as request_count,
                    AVG(LENGTH(question)) as avg_question_length,
                    AVG(LENGTH(answer)) as avg_answer_length,
                    SUM(LENGTH(question)) as total_question_chars,
                    SUM(LENGTH(COALESCE(answer, ''))) as total_answer_chars
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {days_back}
                  AND status = 'success'
                GROUP BY date
                ORDER BY date DESC
            """
            
            logger.debug("Executing cost query for table: %s", config.INFERENCE_LOGS_TABLE)
            df = self.db.query_table(query)
            
            row_count = df.count()
            logger.debug("Query returned %d rows", row_count)
            
            if row_count == 0:
                logger.warning("No cost data found for last %d days", days_back)
                return pd.DataFrame()
            
            logger.debug("Converting Spark DataFrame to pandas")
            pdf = df.toPandas()
            
            # Estimate tokens and costs
            # Input tokens: question (~100 chars) + context retrieval (~2000 chars) + system prompt (~500 chars)
            # Output tokens: answer length
            
            logger.debug("Estimating tokens and costs from character counts")
            
            # Estimate tokens from character counts (4 chars ≈ 1 token)
            pdf['estimated_input_tokens'] = (pdf['total_question_chars'] // 4) + (pdf['request_count'] * 650)  # 650 tokens for context + system
            pdf['estimated_output_tokens'] = pdf['total_answer_chars'] // 4
            
            logger.debug("Calculating cost estimates using OpenAI pricing")
            pdf['estimated_cost_usd'] = pdf.apply(
                lambda row: estimate_cost(
                    int(row['estimated_input_tokens']),
                    int(row['estimated_output_tokens'])
                ),
                axis=1
            )
            
            pdf['cost_per_request'] = pdf['estimated_cost_usd'] / pdf['request_count']
            
            total_cost = pdf['estimated_cost_usd'].sum()
            total_requests = pdf['request_count'].sum()
            logger.info("Daily costs calculated: %d days, %d requests, $%.2f total estimated cost",
                       len(pdf), total_requests, total_cost)
            
            return pdf
            
        except AttributeError as e:
            logger.exception("DataFrame column access error calculating daily costs: %s", e)
            return pd.DataFrame()
        except KeyError as e:
            logger.exception("Missing expected column in cost data: %s", e)
            return pd.DataFrame()
        except Exception as e:
            logger.exception("Failed to calculate daily costs: %s", e)
            return pd.DataFrame()
    
    def get_cost_summary(self, days_back: int = 30) -> Dict:
        """
        Get summarized cost metrics.
        
        Args:
            days_back: Number of days to analyze (default: 30)
        
        Returns:
            Dict with total costs, averages, projections, and token counts:
            - total_cost_usd: Total estimated cost
            - avg_daily_cost_usd: Average cost per day
            - avg_cost_per_request: Average cost per request
            - total_requests: Total number of requests
            - projected_monthly_cost: Projected cost for 30 days
            - days_analyzed: Number of days included in analysis
            - total_input_tokens: Total estimated input tokens
            - total_output_tokens: Total estimated output tokens
        
        Raises:
            Exception: If cost calculation or aggregation fails (logged but not re-raised)
        """
        logger.info("Generating cost summary for last %d days", days_back)
        
        try:
            cost_df = self.get_daily_costs(days_back)
            
            if cost_df.empty:
                logger.warning("No cost data available for summary")
                return {
                    "total_cost_usd": 0.0,
                    "avg_daily_cost_usd": 0.0,
                    "avg_cost_per_request": 0.0,
                    "total_requests": 0,
                    "projected_monthly_cost": 0.0,
                    "days_analyzed": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0
                }
            
            logger.debug("Aggregating cost metrics")
            total_cost = float(cost_df['estimated_cost_usd'].sum())
            total_requests = int(cost_df['request_count'].sum())
            avg_daily_cost = float(cost_df['estimated_cost_usd'].mean())
            total_input_tokens = int(cost_df['estimated_input_tokens'].sum())
            total_output_tokens = int(cost_df['estimated_output_tokens'].sum())
            
            summary = {
                "total_cost_usd": total_cost,
                "avg_daily_cost_usd": avg_daily_cost,
                "avg_cost_per_request": total_cost / total_requests if total_requests > 0 else 0.0,
                "total_requests": total_requests,
                "projected_monthly_cost": avg_daily_cost * 30,
                "days_analyzed": len(cost_df),
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens
            }
            
            logger.info("Cost summary generated: $%.2f total, $%.2f avg daily, %d requests over %d days",
                       total_cost, avg_daily_cost, total_requests, len(cost_df))
            logger.debug("Projected monthly cost: $%.2f", summary['projected_monthly_cost'])
            
            return summary
            
        except KeyError as e:
            logger.exception("Missing expected column in cost summary: %s", e)
            return {
                "total_cost_usd": 0.0,
                "avg_daily_cost_usd": 0.0,
                "avg_cost_per_request": 0.0,
                "total_requests": 0,
                "projected_monthly_cost": 0.0,
                "days_analyzed": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0
            }
        except Exception as e:
            logger.exception("Failed to generate cost summary: %s", e)
            return {
                "total_cost_usd": 0.0,
                "avg_daily_cost_usd": 0.0,
                "avg_cost_per_request": 0.0,
                "total_requests": 0,
                "projected_monthly_cost": 0.0,
                "days_analyzed": 0,
                "total_input_tokens": 0,
                "total_output_tokens": 0
            }
    
    def get_cost_trend(self, days_back: int = 30) -> pd.DataFrame:
        """
        Analyze cost trends over time.
        
        Args:
            days_back: Number of days to analyze (default: 30)
        
        Returns:
            DataFrame with rolling averages and trend indicators. Includes:
            - All columns from get_daily_costs
            - rolling_7d_cost: 7-day rolling average cost
            - rolling_7d_requests: 7-day rolling average request count
            Returns empty DataFrame if no data.
        
        Raises:
            Exception: If trend calculation fails (logged but not re-raised)
        """
        logger.info("Analyzing cost trends for last %d days", days_back)
        
        try:
            cost_df = self.get_daily_costs(days_back)
            
            if cost_df.empty:
                logger.warning("No cost data available for trend analysis")
                return pd.DataFrame()
            
            logger.debug("Sorting cost data by date")
            # Sort by date
            cost_df = cost_df.sort_values('date')
            
            logger.debug("Calculating 7-day rolling averages")
            # Calculate rolling averages
            cost_df['rolling_7d_cost'] = cost_df['estimated_cost_usd'].rolling(window=7, min_periods=1).mean()
            cost_df['rolling_7d_requests'] = cost_df['request_count'].rolling(window=7, min_periods=1).mean()
            
            logger.info("Cost trend analysis complete: %d days analyzed", len(cost_df))
            logger.debug("Latest 7-day avg cost: $%.2f", cost_df['rolling_7d_cost'].iloc[-1] if len(cost_df) > 0 else 0.0)
            
            return cost_df
            
        except KeyError as e:
            logger.exception("Missing expected column in cost trend analysis: %s", e)
            return pd.DataFrame()
        except Exception as e:
            logger.exception("Failed to analyze cost trends: %s", e)
            return pd.DataFrame()
    
    def identify_cost_anomalies(self, days_back: int = 30, std_threshold: float = 2.0) -> pd.DataFrame:
        """
        Identify days with abnormally high costs.
        
        Args:
            days_back: Number of days to analyze (default: 30)
            std_threshold: Standard deviations from mean to flag as anomaly (default: 2.0)
        
        Returns:
            DataFrame with anomalous cost days including:
            - date: Date of anomaly
            - request_count: Number of requests on that day
            - estimated_cost_usd: Estimated cost for that day
            - cost_per_request: Average cost per request
            - deviation_from_mean: Number of standard deviations from mean
            Returns empty DataFrame if insufficient data (< 3 days).
        
        Raises:
            Exception: If anomaly detection fails (logged but not re-raised)
        """
        logger.info("Identifying cost anomalies for last %d days (threshold: %.1f std)", 
                   days_back, std_threshold)
        
        try:
            cost_df = self.get_daily_costs(days_back)
            
            if cost_df.empty or len(cost_df) < 3:
                logger.warning("Insufficient data for anomaly detection (need at least 3 days, got %d)", 
                             len(cost_df) if not cost_df.empty else 0)
                return pd.DataFrame()
            
            logger.debug("Calculating cost statistics for anomaly detection")
            mean_cost = cost_df['estimated_cost_usd'].mean()
            std_cost = cost_df['estimated_cost_usd'].std()
            
            logger.debug("Cost stats: mean=$%.2f, std=$%.2f", mean_cost, std_cost)
            
            # Flag anomalies
            cost_df['is_anomaly'] = cost_df['estimated_cost_usd'] > (mean_cost + std_threshold * std_cost)
            cost_df['deviation_from_mean'] = (cost_df['estimated_cost_usd'] - mean_cost) / std_cost
            
            anomalies = cost_df[cost_df['is_anomaly']]
            
            logger.info("Found %d cost anomalies out of %d days", len(anomalies), len(cost_df))
            
            if len(anomalies) > 0:
                logger.warning("Cost anomalies detected on dates: %s", 
                             ", ".join(anomalies['date'].astype(str).tolist()))
                logger.debug("Anomaly cost range: $%.2f to $%.2f", 
                           anomalies['estimated_cost_usd'].min(),
                           anomalies['estimated_cost_usd'].max())
            
            return anomalies[['date', 'request_count', 'estimated_cost_usd', 'cost_per_request', 'deviation_from_mean']]
            
        except KeyError as e:
            logger.exception("Missing expected column in anomaly detection: %s", e)
            return pd.DataFrame()
        except ZeroDivisionError as e:
            logger.exception("Zero standard deviation in cost data (all costs identical): %s", e)
            return pd.DataFrame()
        except Exception as e:
            logger.exception("Failed to identify cost anomalies: %s", e)
            return pd.DataFrame()
    
    def save_cost_metrics(self, days_back: int = 1) -> None:
        """
        Save cost metrics to Delta table for historical tracking.
        
        Args:
            days_back: Number of days of metrics to save (default: 1)
        
        Returns:
            None
        
        Raises:
            Exception: If saving metrics fails (logged but not re-raised)
        """
        logger.info("Saving cost metrics for last %d days", days_back)
        
        try:
            cost_df = self.get_daily_costs(days_back)
            
            if cost_df.empty:
                logger.warning("No cost data to save")
                return
            
            logger.debug("Adding metadata columns to cost metrics")
            # Add metadata
            cost_df['calculated_at'] = datetime.now()
            cost_df['model_used'] = 'gpt-4o-mini'  # Adjust based on actual model
            
            logger.debug("Converting pandas DataFrame to Spark DataFrame")
            # Convert to Spark DataFrame and save
            spark_df = self.db.spark.createDataFrame(cost_df)
            
            # Use a dedicated cost metrics table
            cost_table = "main.default.gdpr_agent_cost_metrics"
            logger.debug("Writing cost metrics to table: %s", cost_table)
            
            self.db.write_metrics(spark_df, cost_table)
            
            logger.info("Cost metrics saved successfully to %s (%d rows)", cost_table, len(cost_df))
            
        except AttributeError as e:
            logger.exception("Databricks client attribute error saving cost metrics: %s", e)
        except ValueError as e:
            logger.exception("Invalid data format for Spark DataFrame conversion: %s", e)
        except Exception as e:
            logger.exception("Failed to save cost metrics: %s", e)
