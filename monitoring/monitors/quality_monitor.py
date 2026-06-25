"""
Quality monitoring using LLM-as-a-judge.
"""
from datetime import datetime, timedelta
from typing import Dict, List
import pandas as pd

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient
from monitoring.utils.llm_judge import LLMJudge

class QualityMonitor:
    """Monitor response quality using LLM judge"""
    
    def __init__(self, db_client: DatabricksClient, judge: LLMJudge):
        self.db = db_client
        self.judge = judge
    
    def evaluate_recent_queries(self, days_back: int = None, sample_size: int = None) -> pd.DataFrame:
        """
        Evaluate recent production queries.
        
        Samples 10% of successful queries (excluding refusals) for LLM evaluation.
        
        Args:
            days_back: Number of days to look back (default from config)
            sample_size: Number of queries to evaluate (default from config)
        
        Returns:
            DataFrame with evaluation results
        """
        days_back = days_back or config.DEFAULT_LOOKBACK_DAYS
        sample_size = sample_size or config.SAMPLE_SIZE_FOR_EVALUATION
        
        # First, get count of valid queries (successful, non-refusal)
        count_query = f"""
            SELECT COUNT(*) as total_valid_queries
            FROM {config.INFERENCE_LOGS_TABLE}
            WHERE date >= current_date() - {days_back}
              AND status = 'success'
              AND is_valid_answer = true
        """
        
        count_result = self.db.query_table(count_query).first()
        total_valid = int(count_result.total_valid_queries) if count_result else 0
        
        if total_valid == 0:
            print("⚠️  No valid queries found to evaluate")
            return pd.DataFrame()
        
        # Calculate 10% sample size (but respect max sample_size)
        target_sample = min(int(total_valid * 0.1), sample_size)
        
        if target_sample == 0:
            target_sample = min(1, total_valid)  # At least 1 if any exist
        
        print(f"🔍 Evaluating queries from last {days_back} days")
        print(f"   Total valid queries: {total_valid}")
        print(f"   Sampling 10%: {target_sample} queries")
        
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
        
        df = self.db.query_table(query)
        
        if df.count() == 0:
            print("⚠️  No queries found to evaluate")
            return pd.DataFrame()
        
        # Convert to Pandas and evaluate
        pdf = df.toPandas()
        results = []
        
        for idx, row in pdf.iterrows():
            print(f"  Evaluating {idx+1}/{len(pdf)}: {row['question'][:60]}...")
            
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
        
        results_df = pd.DataFrame(results)
        
        # Save to Delta table
        spark_df = self.db.spark.createDataFrame(results_df)
        self.db.write_metrics(spark_df, config.QUALITY_METRICS_TABLE)
        
        print(f"✅ Evaluated {len(results)} queries. Metrics saved.")
        
        return results_df
    
    def get_quality_summary(self, days_back: int = 7) -> Dict:
        """Get quality metrics summary"""
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
        
        result = self.db.query_table(query).first()
        
        if result is None:
            return {}
        
        return {
            "avg_relevance": float(result.avg_relevance or 0),
            "avg_accuracy": float(result.avg_accuracy or 0),
            "avg_completeness": float(result.avg_completeness or 0),
            "avg_citation": float(result.avg_citation or 0),
            "avg_clarity": float(result.avg_clarity or 0),
            "avg_overall": float(result.avg_overall or 0),
            "evaluated_queries": int(result.evaluated_queries or 0)
        }
