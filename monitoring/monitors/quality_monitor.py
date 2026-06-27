"""
Quality monitoring using LLM-as-a-judge.

Provides automated quality assessment of GDPR Agent responses using
LLM-as-a-judge evaluation. Samples recent queries and evaluates responses
across multiple dimensions: relevance, accuracy, completeness, citation,
and clarity.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient
from monitoring.utils.llm_judge import LLMJudge

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class QualityMonitor:
    """
    Monitor response quality using LLM judge.
    
    Evaluates agent responses using an LLM-as-a-judge approach to assess
    quality dimensions and track quality trends over time.
    """
    
    def __init__(self, db_client: DatabricksClient, judge: LLMJudge) -> None:
        """
        Initialize quality monitor.
        
        Args:
            db_client: Databricks client for data access
            judge: LLM judge for quality evaluation
        """
        logger.debug("Initializing QualityMonitor")
        self.db = db_client
        self.judge = judge
        logger.info("QualityMonitor initialized successfully")
    
    def evaluate_recent_queries(self, days_back: int = None, sample_size: int = None) -> pd.DataFrame:
        """
        Evaluate recent production queries using LLM judge.
        
        Samples 10% of successful queries (excluding refusals) for LLM evaluation.
        Evaluates responses across multiple quality dimensions and persists results
        to Delta table for historical tracking.
        
        Args:
            days_back: Number of days to look back (default from config)
            sample_size: Maximum number of queries to evaluate (default from config)
        
        Returns:
            DataFrame with evaluation results including quality scores across
            dimensions (relevance, accuracy, completeness, citation, clarity, overall).
            Returns empty DataFrame if no valid queries found or evaluation fails.
        
        Raises:
            Exception: If query execution or evaluation fails (logged but not re-raised)
        """
        days_back = days_back or config.DEFAULT_LOOKBACK_DAYS
        sample_size = sample_size or config.SAMPLE_SIZE_FOR_EVALUATION
        
        logger.info("Initiating quality evaluation for last %d days (max sample: %d)", 
                   days_back, sample_size)
        
        try:
            # First, get count of valid queries (successful, non-refusal)
            count_query = f"""
                SELECT COUNT(*) as total_valid_queries
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {days_back}
                  AND status = 'success'
                  AND is_valid_answer = true
            """
            
            logger.debug("Querying total valid query count from table: %s", config.INFERENCE_LOGS_TABLE)
            count_result = self.db.query_table(count_query).first()
            total_valid = int(count_result.total_valid_queries) if count_result else 0
            
            logger.debug("Total valid queries found: %d", total_valid)
            
            if total_valid == 0:
                logger.warning("No valid queries found to evaluate for last %d days", days_back)
                return pd.DataFrame()
            
            # Calculate 10% sample size (but respect max sample_size)
            target_sample = min(int(total_valid * 0.1), sample_size)
            
            if target_sample == 0:
                target_sample = min(1, total_valid)  # At least 1 if any exist
            
            logger.info("Quality evaluation sampling: %d queries (10%% of %d valid queries)", 
                       target_sample, total_valid)
            
            # Query with random sampling
            query = f"""
                SELECT 
                    date,
                    request_id,
                    timestamp,
                    question,
                    answer,
                    context
                FROM {config.INFERENCE_LOGS_TABLE}
                WHERE date >= current_date() - {days_back}
                  AND status = 'success'
                  AND is_valid_answer = true
                ORDER BY RAND()
                LIMIT {target_sample}
            """
            
            logger.debug("Executing random sampling query for %d queries", target_sample)
            df = self.db.query_table(query)
            
            sample_count = df.count()
            logger.debug("Sample query returned %d rows", sample_count)
            
            if sample_count == 0:
                logger.warning("No queries returned from sampling query")
                return pd.DataFrame()
            
            # Convert to Pandas and evaluate
            logger.debug("Converting Spark DataFrame to pandas for evaluation")
            pdf = df.toPandas()
            results = []
            
            logger.info("Starting LLM evaluation of %d queries", len(pdf))
            
            for idx, row in pdf.iterrows():
                question_preview = row['question'][:60] if len(row['question']) > 60 else row['question']
                logger.debug("Evaluating query %d/%d: %s", idx + 1, len(pdf), question_preview)
                
                try:
                    scores = self.judge.evaluate(
                        question=row['question'],
                        answer=row['answer'],
                        context=row.get('context', '')
                    )
                    
                    results.append({
                        'evaluation_date': datetime.now(),
                        'query_date': row['date'],
                        'request_id': row['request_id'],
                        'question': row['question'],
                        'answer': row['answer'],
                        **scores
                    })
                    
                    logger.debug("Query %d evaluated successfully (overall score: %.2f)", 
                               idx + 1, scores.get('overall', 0.0))
                    
                except Exception as eval_error:
                    logger.exception("Failed to evaluate query %d (request_id: %s): %s", 
                                   idx + 1, row['request_id'], eval_error)
                    # Continue with next query rather than failing entire batch
                    continue
            
            if len(results) == 0:
                logger.warning("No queries were successfully evaluated")
                return pd.DataFrame()
            
            logger.debug("Creating results DataFrame from %d successful evaluations", len(results))
            results_df = pd.DataFrame(results)
            
            # Calculate average scores for logging
            avg_overall = results_df.get('overall', pd.Series([0])).mean()
            
            # Save to Delta table
            logger.debug("Converting results to Spark DataFrame for persistence")
            spark_df = self.db.spark.createDataFrame(results_df)
            
            logger.debug("Writing %d evaluation results to table: %s", 
                        len(results), config.QUALITY_METRICS_TABLE)
            self.db.write_metrics(spark_df, config.QUALITY_METRICS_TABLE)
            
            logger.info("Quality evaluation complete: %d queries evaluated, avg overall score: %.2f, results saved to %s",
                       len(results), avg_overall, config.QUALITY_METRICS_TABLE)
            
            return results_df
            
        except AttributeError as e:
            logger.exception("Error accessing result attributes in quality evaluation: %s", e)
            return pd.DataFrame()
        except KeyError as e:
            logger.exception("Missing expected column in quality evaluation: %s", e)
            return pd.DataFrame()
        except Exception as e:
            logger.exception("Failed to evaluate recent queries: %s", e)
            return pd.DataFrame()
    
    def get_quality_summary(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Get aggregated quality metrics summary.
        
        Retrieves average quality scores across all evaluation dimensions
        from the quality metrics table for the specified time period.
        
        Args:
            days_back: Number of days to analyze (default: 7)
        
        Returns:
            Dict with average quality scores:
            - avg_relevance: Average relevance score
            - avg_accuracy: Average accuracy score
            - avg_completeness: Average completeness score
            - avg_citation: Average citation quality score
            - avg_clarity: Average clarity score
            - avg_overall: Average overall quality score
            - evaluated_queries: Total number of evaluated queries
            Returns empty dict if no evaluation data available or query fails.
        
        Raises:
            Exception: If query execution fails (logged but not re-raised)
        """
        logger.info("Retrieving quality summary for last %d days", days_back)
        
        try:
            query = f"""
                SELECT 
                    AVG(relevance) as avg_relevance,
                    AVG(accuracy) as avg_accuracy,
                    AVG(completeness) as avg_completeness,
                    AVG(citation) as avg_citation,
                    AVG(clarity) as avg_clarity,
                    AVG(overall) as avg_overall,
                    COUNT(*) as evaluated_queries
                FROM {config.QUALITY_METRICS_TABLE}
                WHERE evaluation_date >= current_date() - {days_back}
            """
            
            logger.debug("Executing quality summary query on table: %s", config.QUALITY_METRICS_TABLE)
            result = self.db.query_table(query).first()
            
            if result is None:
                logger.warning("No quality metrics found for last %d days", days_back)
                return {}
            
            summary = {
                "avg_relevance": float(result.avg_relevance or 0),
                "avg_accuracy": float(result.avg_accuracy or 0),
                "avg_completeness": float(result.avg_completeness or 0),
                "avg_citation": float(result.avg_citation or 0),
                "avg_clarity": float(result.avg_clarity or 0),
                "avg_overall": float(result.avg_overall or 0),
                "evaluated_queries": int(result.evaluated_queries or 0)
            }
            
            logger.info("Quality summary retrieved: %d queries, avg overall score: %.2f",
                       summary['evaluated_queries'], summary['avg_overall'])
            logger.debug("Detailed scores - relevance: %.2f, accuracy: %.2f, completeness: %.2f, citation: %.2f, clarity: %.2f",
                        summary['avg_relevance'], summary['avg_accuracy'], summary['avg_completeness'],
                        summary['avg_citation'], summary['avg_clarity'])
            
            return summary
            
        except AttributeError as e:
            logger.exception("Error accessing result attributes in quality summary: %s", e)
            return {}
        except Exception as e:
            logger.exception("Failed to retrieve quality summary: %s", e)
            return {}
