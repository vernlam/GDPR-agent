"""
Refusal monitoring using keyword-based classification.
"""
from datetime import datetime
from typing import Dict
import pandas as pd

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient
from monitoring.utils.response_classifier import ResponseClassifier

class RefusalMonitor:
    """Monitor refusal rates and types"""
    
    def __init__(self, db_client: DatabricksClient):
        self.db = db_client
        self.classifier = ResponseClassifier()
    
    def analyze_refusals(self, days_back: int = 7) -> Dict:
        """
        Analyze refusal patterns in recent queries.
        
        Returns:
            Dict with refusal statistics
        """
        print(f"🔍 Analyzing refusals from last {days_back} days...")
        
        # Query all successful requests (status='success' but answer might be a refusal)
        query = f"""
            SELECT 
                request_id,
                timestamp,
                question,
                answer,
                response_classification,
                refusal_type
            FROM {config.INFERENCE_LOGS_TABLE}
            WHERE date >= current_date() - {days_back}
            AND status = 'success'
            AND answer IS NOT NULL
        """
        
        df = self.db.query_table(query)
        
        if df.count() == 0:
            print("⚠️  No queries found")
            return {}
        
        pdf = df.toPandas()
        total = len(pdf)
        
        # Count by classification
        refusals = pdf[pdf['response_classification'] == 'refusal']
        refusal_count = len(refusals)
        refusal_rate = refusal_count / total
        
        # Count by refusal type
        refusal_types = refusals['refusal_type'].value_counts().to_dict()
        
        summary = {
            'total_queries': total,
            'refusal_count': refusal_count,
            'refusal_rate': refusal_rate,
            'refusals_by_type': refusal_types,
            'analysis_date': datetime.now()
        }
        
        print(f"   Total Queries: {total}")
        print(f"   Refusals: {refusal_count} ({refusal_rate*100:.1f}%)")
        
        if refusal_types:
            print(f"   Refusal Breakdown:")
            for rtype, count in refusal_types.items():
                print(f"      • {rtype}: {count}")
        
        return summary
    
    def check_refusal_thresholds(self, days_back: int = 1) -> list:
        """Check if refusal rate exceeds thresholds"""
        summary = self.analyze_refusals(days_back)
        alerts = []
        
        if not summary:
            return alerts
        
        # Alert if refusal rate is high
        if summary['refusal_rate'] > config.MAX_REFUSAL_RATE:
            alerts.append({
                'severity': 'HIGH',
                'metric': 'refusal_rate',
                'threshold': config.MAX_REFUSAL_RATE,
                'actual': summary['refusal_rate'],
                'message': f"Refusal rate ({summary['refusal_rate']*100:.1f}%) exceeds threshold ({config.MAX_REFUSAL_RATE*100:.1f}%)"
            })
        
        # Alert if insufficient_context refusals are too high
        insufficient_context = summary['refusals_by_type'].get('insufficient_context', 0)
        if insufficient_context > config.MAX_INSUFFICIENT_CONTEXT_PER_DAY:
            alerts.append({
                'severity': 'MEDIUM',
                'metric': 'insufficient_context',
                'threshold': config.MAX_INSUFFICIENT_CONTEXT_PER_DAY,
                'actual': insufficient_context,
                'message': f"Insufficient context refusals ({insufficient_context}) indicate retrieval issues"
            })
        
        return alerts