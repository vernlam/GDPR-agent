"""
Refusal monitoring using keyword-based classification.

Provides refusal detection and classification for GDPR Agent responses.
Tracks refusal rates, categorizes refusal types (insufficient_context,
off_topic, etc.), and monitors thresholds for alerting.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient
from monitoring.utils.response_classifier import ResponseClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class RefusalMonitor:
    """
    Monitor refusal rates and types.
    
    Tracks agent refusals using keyword-based classification to identify
    when the agent declines to answer due to insufficient context, off-topic
    queries, or other reasons.
    """
    
    def __init__(self, db_client: DatabricksClient) -> None:
        """
        Initialize refusal monitor.
        
        Args:
            db_client: Databricks client for data access
        """
        logger.debug("Initializing RefusalMonitor")
        self.db = db_client
        self.classifier = ResponseClassifier()
        logger.info("RefusalMonitor initialized successfully")
    
    def analyze_refusals(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze refusal patterns in recent queries.
        
        Queries successful responses and classifies them as refusals or valid
        answers using keyword-based classification. Calculates refusal rates
        and breaks down refusals by type.
        
        Args:
            days_back: Number of days to analyze (default: 7)
        
        Returns:
            Dict with refusal statistics:
            - total_queries: Total number of queries analyzed
            - refusal_count: Number of refusals detected
            - refusal_rate: Refusal rate as decimal (0.0 to 1.0)
            - refusals_by_type: Dict mapping refusal types to counts
            - analysis_date: Timestamp of analysis
            Returns empty dict if no data available or query fails.
        
        Raises:
            Exception: If query execution fails (logged but not re-raised)
        """
        logger.info("Analyzing refusals from last %d days", days_back)
        
        try:
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
            
            logger.debug("Executing refusal analysis query on table: %s", config.INFERENCE_LOGS_TABLE)
            df = self.db.query_table(query)
            
            row_count = df.count()
            logger.debug("Query returned %d rows", row_count)
            
            if row_count == 0:
                logger.warning("No queries found for refusal analysis (last %d days)", days_back)
                return {}
            
            logger.debug("Converting Spark DataFrame to pandas")
            pdf = df.toPandas()
            total = len(pdf)
            
            # Count by classification
            logger.debug("Classifying responses into refusals and valid answers")
            refusals = pdf[pdf['response_classification'] == 'refusal']
            refusal_count = len(refusals)
            refusal_rate = refusal_count / total if total > 0 else 0.0
            
            # Count by refusal type
            logger.debug("Categorizing refusals by type")
            refusal_types = refusals['refusal_type'].value_counts().to_dict()
            
            summary = {
                'total_queries': total,
                'refusal_count': refusal_count,
                'refusal_rate': refusal_rate,
                'refusals_by_type': refusal_types,
                'analysis_date': datetime.now()
            }
            
            logger.info("Refusal analysis complete: %d refusals out of %d queries (%.1f%% refusal rate)",
                       refusal_count, total, refusal_rate * 100)
            
            if refusal_types:
                logger.debug("Refusal breakdown by type: %s", refusal_types)
                for rtype, count in refusal_types.items():
                    logger.debug("  %s: %d refusals", rtype, count)
            
            return summary
            
        except KeyError as e:
            logger.exception("Missing expected column in refusal analysis: %s", e)
            return {}
        except ZeroDivisionError as e:
            logger.exception("Zero division error in refusal rate calculation: %s", e)
            return {}
        except Exception as e:
            logger.exception("Failed to analyze refusals: %s", e)
            return {}
    
    def check_refusal_thresholds(self, days_back: int = 1) -> List[Dict[str, Any]]:
        """
        Check if refusal rate exceeds configured thresholds.
        
        Analyzes refusal patterns and generates alerts when refusal rate or
        specific refusal types exceed operational thresholds.
        
        Args:
            days_back: Number of days to check thresholds for (default: 1)
        
        Returns:
            List of alert dictionaries. Each alert contains:
            - severity: Alert severity level (HIGH or MEDIUM)
            - metric: Metric that breached threshold
            - threshold: Configured threshold value
            - actual: Actual observed value
            - message: Human-readable alert message
            Returns empty list if no thresholds breached or analysis fails.
        
        Raises:
            Exception: If threshold checking fails (logged but not re-raised)
        """
        logger.info("Checking refusal thresholds for last %d days", days_back)
        
        try:
            logger.debug("Retrieving refusal analysis for threshold checks")
            summary = self.analyze_refusals(days_back)
            alerts = []
            
            if not summary:
                logger.warning("No refusal data available for threshold checking")
                return alerts
            
            # Check overall refusal rate
            refusal_rate = summary['refusal_rate']
            logger.debug("Checking overall refusal rate: %.2f%% (threshold: %.2f%%)",
                        refusal_rate * 100, config.MAX_REFUSAL_RATE * 100)
            
            if refusal_rate > config.MAX_REFUSAL_RATE:
                alert = {
                    'severity': 'HIGH',
                    'metric': 'refusal_rate',
                    'threshold': config.MAX_REFUSAL_RATE,
                    'actual': refusal_rate,
                    'message': f"Refusal rate ({refusal_rate*100:.1f}%) exceeds threshold ({config.MAX_REFUSAL_RATE*100:.1f}%)"
                }
                alerts.append(alert)
                logger.warning("Alert triggered: %s", alert['message'])
            
            # Check insufficient_context refusals specifically
            insufficient_context = summary['refusals_by_type'].get('insufficient_context', 0)
            logger.debug("Checking insufficient_context refusals: %d (threshold: %d)",
                        insufficient_context, config.MAX_INSUFFICIENT_CONTEXT_PER_DAY)
            
            if insufficient_context > config.MAX_INSUFFICIENT_CONTEXT_PER_DAY:
                alert = {
                    'severity': 'MEDIUM',
                    'metric': 'insufficient_context',
                    'threshold': config.MAX_INSUFFICIENT_CONTEXT_PER_DAY,
                    'actual': insufficient_context,
                    'message': f"Insufficient context refusals ({insufficient_context}) indicate retrieval issues"
                }
                alerts.append(alert)
                logger.warning("Alert triggered: %s", alert['message'])
            
            if len(alerts) == 0:
                logger.info("No refusal thresholds breached")
            else:
                logger.warning("Refusal threshold check complete: %d alerts triggered", len(alerts))
            
            return alerts
            
        except KeyError as e:
            logger.exception("Missing expected key in threshold checking: %s", e)
            return []
        except Exception as e:
            logger.exception("Failed to check refusal thresholds: %s", e)
            return []
