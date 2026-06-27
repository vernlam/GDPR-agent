"""
Main monitoring orchestration script.

Coordinates execution of all monitoring modules (quality, performance, error,
cost, drift, refusal) for the GDPR agent. Provides comprehensive health checks,
metric collection, alert detection, and summary reporting. Designed for
scheduled execution in production environments with full error handling and
structured logging.
"""

import logging
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, Optional

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient
from monitoring.utils.llm_judge import LLMJudge
from monitoring.monitors.quality_monitor import QualityMonitor
from monitoring.monitors.performance_monitor import PerformanceMonitor
from monitoring.monitors.error_monitor import ErrorMonitor
from monitoring.monitors.drift_monitor import DriftMonitor
from monitoring.monitors.refusal_monitor import RefusalMonitor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Check if cost_monitor exists before importing
try:
    from monitoring.monitors.cost_monitor import CostMonitor
    COST_MONITOR_AVAILABLE = True
    logger.debug("CostMonitor imported successfully")
except ImportError as e:
    COST_MONITOR_AVAILABLE = False
    logger.warning("CostMonitor not available: %s", e)


def run_monitoring(days_back: Optional[int] = None, sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Run complete monitoring suite for GDPR agent.
    
    Executes all monitoring checks in sequence:
    1. Endpoint health check
    2. Quality evaluation (LLM judge)
    3. Performance analysis
    4. Refusal analysis
    5. Error detection and patterns
    6. Cost analysis (if available)
    7. Drift detection
    
    Each module runs independently with error isolation - failures in one
    module do not prevent execution of others.
    
    Args:
        days_back: Number of days to analyze (defaults to config.DEFAULT_LOOKBACK_DAYS)
        sample_size: Number of queries to evaluate with LLM judge
                    (defaults to config.SAMPLE_SIZE_FOR_EVALUATION)
    
    Returns:
        Dict containing results from all monitoring modules, with 'error' key
        for any module that failed. Keys include:
        - endpoint_status: Endpoint health information
        - quality_summary: LLM judge quality metrics
        - performance_summary: Latency and success rate metrics
        - refusal_summary: Refusal rate and type breakdown
        - error_summary: Error counts, rates, and patterns
        - cost_summary: Cost metrics and projections (if available)
        - drift_results: Query distribution drift analysis
        - alerts: List of triggered alerts
        Returns empty dict on catastrophic failure.
    
    Raises:
        Does not raise exceptions; logs errors and returns partial results
    """
    try:
        days_back = days_back or config.DEFAULT_LOOKBACK_DAYS
        sample_size = sample_size or config.SAMPLE_SIZE_FOR_EVALUATION
        
        logger.info("="*80)
        logger.info("Starting GDPR Agent Monitoring - %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        logger.info("Configuration: days_back=%d, sample_size=%d", days_back, sample_size)
        logger.info("="*80)
        
    except Exception as e:
        logger.exception("Failed to initialize monitoring parameters: %s", e)
        return {}
    
    # Initialize clients
    logger.info("Initializing clients and monitors...")
    try:
        db_client = DatabricksClient()
        logger.debug("DatabricksClient initialized")
        
        judge = LLMJudge()
        logger.debug("LLMJudge initialized")
        
    except Exception as e:
        logger.exception("Failed to initialize clients: %s", e)
        return {"error": "Client initialization failed"}
    
    # Initialize monitors
    try:
        quality_monitor = QualityMonitor(db_client, judge)
        logger.debug("QualityMonitor initialized")
        
        performance_monitor = PerformanceMonitor(db_client)
        logger.debug("PerformanceMonitor initialized")
        
        error_monitor = ErrorMonitor(db_client)
        logger.debug("ErrorMonitor initialized")
        
        if COST_MONITOR_AVAILABLE:
            cost_monitor = CostMonitor(db_client)
            logger.debug("CostMonitor initialized")
        else:
            cost_monitor = None
            logger.debug("CostMonitor skipped (not available)")
        
        drift_monitor = DriftMonitor(db_client)
        logger.debug("DriftMonitor initialized")
        
        refusal_monitor = RefusalMonitor(db_client)
        logger.debug("RefusalMonitor initialized")
        
        logger.info("All monitors initialized successfully")
        
    except Exception as e:
        logger.exception("Failed to initialize monitors: %s", e)
        return {"error": "Monitor initialization failed"}
    
    # Store results
    results = {}
    
    # 1. Check endpoint health
    logger.info("Step 1/7: Checking Endpoint Health...")
    try:
        endpoint_status = db_client.get_endpoint_status(config.ENDPOINT_NAME)
        logger.info("Endpoint status: name=%s, state=%s", 
                   endpoint_status.get('name'), endpoint_status.get('state'))
        results['endpoint_status'] = endpoint_status
    except Exception as e:
        logger.exception("Failed to get endpoint status: %s", e)
        results['endpoint_status'] = {"error": str(e)}
    
    # 2. Evaluate quality
    logger.info("Step 2/7: Evaluating Quality (LLM Judge)...")
    try:
        quality_df = quality_monitor.evaluate_recent_queries(days_back, sample_size)
        
        if not quality_df.empty:
            quality_summary = quality_monitor.get_quality_summary(days_back)
            logger.info("Quality Summary (last %d days): relevance=%.2f, accuracy=%.2f, completeness=%.2f, "
                       "citation=%.2f, overall=%.2f, queries=%d",
                       days_back,
                       quality_summary.get('avg_relevance', 0),
                       quality_summary.get('avg_accuracy', 0),
                       quality_summary.get('avg_completeness', 0),
                       quality_summary.get('avg_citation', 0),
                       quality_summary.get('avg_overall', 0),
                       quality_summary.get('evaluated_queries', 0))
            results['quality_summary'] = quality_summary
            
            # Check quality thresholds
            if quality_summary.get('avg_overall', 0) < config.MIN_QUALITY_SCORE:
                logger.warning("Quality score (%.2f) below threshold (%.2f)",
                             quality_summary['avg_overall'], config.MIN_QUALITY_SCORE)
        else:
            logger.warning("No queries available for quality evaluation")
            results['quality_summary'] = {}
    except Exception as e:
        logger.exception("Failed to evaluate quality: %s", e)
        results['quality_summary'] = {"error": str(e)}
    
    # 3. Check performance
    logger.info("Step 3/7: Analyzing Performance...")
    try:
        perf_summary = performance_monitor.get_performance_summary(days_back)
        
        if perf_summary:
            logger.info("Performance Summary (last %d days): requests=%d, avg_daily=%.1f, "
                       "latency=%.2fs, max_latency=%.2fs, success_rate=%.1f%%",
                       days_back,
                       perf_summary.get('total_requests', 0),
                       perf_summary.get('avg_daily_requests', 0),
                       perf_summary.get('avg_latency_seconds', 0),
                       perf_summary.get('max_latency_seconds', 0),
                       perf_summary.get('avg_success_rate', 0))
            results['performance_summary'] = perf_summary
            
            # Check performance thresholds
            if perf_summary.get('avg_latency_seconds', 0) > config.MAX_AVG_LATENCY:
                logger.warning("Average latency (%.2fs) exceeds threshold (%.2fs)",
                             perf_summary['avg_latency_seconds'], config.MAX_AVG_LATENCY)
            if perf_summary.get('avg_success_rate', 100) < config.MIN_SUCCESS_RATE:
                logger.warning("Success rate (%.1f%%) below threshold (%.1f%%)",
                             perf_summary['avg_success_rate'], config.MIN_SUCCESS_RATE)
        else:
            logger.warning("No performance data available")
            results['performance_summary'] = {}
    except Exception as e:
        logger.exception("Failed to analyze performance: %s", e)
        results['performance_summary'] = {"error": str(e)}

    # 4. Analyze refusals
    logger.info("Step 4/7: Analyzing Refusals...")
    try:
        refusal_summary = refusal_monitor.analyze_refusals(days_back)
        
        if refusal_summary:
            logger.info("Refusal Summary: rate=%.1f%%",
                       refusal_summary.get('refusal_rate', 0) * 100)
            
            alerts = refusal_monitor.check_refusal_thresholds(days_back=1)
            if alerts:
                logger.warning("Refusal alerts triggered: %d alerts", len(alerts))
                for alert in alerts:
                    logger.warning("Refusal alert [%s]: %s", 
                                 alert.get('severity'), alert.get('message'))
        
        results['refusal_summary'] = refusal_summary
    except Exception as e:
        logger.exception("Failed to analyze refusals: %s", e)
        results['refusal_summary'] = {"error": str(e)}
    
    # 5. Check for errors
    logger.info("Step 5/7: Analyzing Errors...")
    try:
        error_summary = error_monitor.get_error_summary(days_back)
        
        if error_summary:
            logger.info("Error Summary (last %d days): total=%d, rate=%.2f%%, days_with_errors=%d",
                       days_back,
                       error_summary.get('total_errors', 0),
                       error_summary.get('error_rate', 0),
                       error_summary.get('days_with_errors', 0))
            
            if error_summary.get('errors_by_type'):
                logger.debug("Error types: %s", error_summary['errors_by_type'])
            
            results['error_summary'] = error_summary
            
            # Check for alerts
            alerts = error_monitor.check_alert_thresholds(days_back=1)
            if alerts:
                logger.warning("Error alerts triggered: %d alerts", len(alerts))
                for alert in alerts:
                    logger.warning("Error alert [%s]: %s",
                                 alert.get('severity'), alert.get('message'))
                results['alerts'] = alerts
            
            # Get error patterns
            error_patterns = error_monitor.get_error_patterns(days_back, top_n=5)
            if not error_patterns.empty:
                logger.info("Top error patterns found: %d patterns", len(error_patterns))
                for idx, row in error_patterns.head(3).iterrows():
                    logger.debug("Error pattern: question='%s...' count=%d",
                               row['question'][:50], row['occurrence_count'])
        else:
            logger.info("No errors detected")
            results['error_summary'] = error_summary
    except Exception as e:
        logger.exception("Failed to analyze errors: %s", e)
        results['error_summary'] = {"error": str(e)}
    
    # 6. Cost analysis
    if COST_MONITOR_AVAILABLE and cost_monitor:
        logger.info("Step 6/7: Analyzing Costs...")
        try:
            cost_summary = cost_monitor.get_cost_summary(days_back)
            
            if cost_summary:
                logger.info("Cost Summary (last %d days): total=$%.2f, daily=$%.2f, "
                           "per_request=$%.4f, monthly_projected=$%.2f, input_tokens=%d, output_tokens=%d",
                           days_back,
                           cost_summary.get('total_cost_usd', 0),
                           cost_summary.get('avg_daily_cost_usd', 0),
                           cost_summary.get('avg_cost_per_request', 0),
                           cost_summary.get('projected_monthly_cost', 0),
                           cost_summary.get('total_input_tokens', 0),
                           cost_summary.get('total_output_tokens', 0))
                results['cost_summary'] = cost_summary
                
                # Check for cost anomalies
                anomalies = cost_monitor.identify_cost_anomalies(days_back)
                if not anomalies.empty:
                    logger.warning("Cost anomalies detected: %d days", len(anomalies))
                    for idx, row in anomalies.head(3).iterrows():
                        logger.warning("Cost anomaly on %s: $%.2f (%.1f sigma from mean)",
                                     row['date'], row['estimated_cost_usd'], row['deviation_from_mean'])
            else:
                logger.warning("No cost data available")
                results['cost_summary'] = {}
        except Exception as e:
            logger.exception("Failed to analyze costs: %s", e)
            results['cost_summary'] = {"error": str(e)}
    else:
        logger.info("Step 6/7: Skipping cost analysis (CostMonitor not available)")
        results['cost_summary'] = {"skipped": "CostMonitor not available"}
    
    # 7. Drift detection
    logger.info("Step 7/7: Detecting Drift...")
    try:
        drift_results = drift_monitor.detect_distribution_drift()
        
        if drift_results and 'drift_score' in drift_results:
            drift_detected = drift_results.get('drift_detected', False)
            logger.info("Drift Analysis: detected=%s, score=%.3f, threshold=%.3f",
                       drift_detected,
                       drift_results.get('drift_score', 0),
                       drift_results.get('drift_threshold', 0.3))
            
            # Log detailed metrics if they exist
            if 'baseline_unique_queries' in drift_results:
                logger.debug("Drift details: baseline=%d, recent=%d, new=%d (%.1f%%), disappeared=%d (%.1f%%)",
                           drift_results['baseline_unique_queries'],
                           drift_results['recent_unique_queries'],
                           drift_results['new_query_count'],
                           drift_results['new_query_rate'] * 100,
                           drift_results['disappeared_query_count'],
                           drift_results['disappeared_query_rate'] * 100)
            results['drift_results'] = drift_results
            
            # Get top queries
            top_queries = drift_monitor.get_top_queries(days_back, top_n=5)
            if not top_queries.empty:
                logger.debug("Top 5 queries found")
                for idx, row in top_queries.head(3).iterrows():
                    logger.debug("Top query: '%s...' (count=%d, pct=%.1f%%)",
                               row['question'][:50], row['frequency'], row['percentage'])
            
            # Get keyword trends
            keywords = drift_monitor.get_keyword_distribution(days_back, top_n=10)
            if not keywords.empty:
                top_keywords = keywords.head(10)['keyword'].tolist()
                logger.debug("Top keywords: %s", ', '.join(top_keywords[:5]))
        else:
            logger.warning("Insufficient data for drift detection")
            results['drift_results'] = {}
    except Exception as e:
        logger.exception("Failed to detect drift: %s", e)
        results['drift_results'] = {"error": str(e)}
    
    # Summary
    logger.info("="*80)
    logger.info("Monitoring Complete!")
    logger.info("="*80)
    
    # Collect critical alerts
    critical_alerts = []
    
    try:
        if results.get('quality_summary', {}).get('avg_overall', 5) < config.MIN_QUALITY_SCORE:
            critical_alerts.append("Low quality score")
        
        if results.get('performance_summary', {}).get('avg_success_rate', 100) < config.MIN_SUCCESS_RATE:
            critical_alerts.append("Low success rate")
        
        if results.get('performance_summary', {}).get('avg_latency_seconds', 0) > config.MAX_AVG_LATENCY:
            critical_alerts.append("High latency")
        
        if results.get('error_summary', {}).get('total_errors', 0) > config.MAX_ERRORS_PER_DAY:
            critical_alerts.append("High error count")
        
        if results.get('drift_results', {}).get('drift_detected', False):
            critical_alerts.append("Query distribution drift detected")
        
        if critical_alerts:
            logger.warning("CRITICAL ALERTS TRIGGERED: %d alerts", len(critical_alerts))
            for alert in critical_alerts:
                logger.warning("Critical alert: %s", alert)
        else:
            logger.info("All metrics within acceptable thresholds")
    except Exception as e:
        logger.exception("Error collecting critical alerts: %s", e)
    
    return results


def main() -> int:
    """
    Main entry point for monitoring script.
    
    Parses command-line arguments and executes monitoring suite.
    
    Returns:
        Exit code: 0 for success, 1 for failure
    
    Raises:
        Does not raise exceptions; logs errors and returns exit code
    """
    parser = argparse.ArgumentParser(description="Run GDPR Agent monitoring")
    parser.add_argument("--days-back", type=int, default=7, help="Days to look back (default: 7)")
    parser.add_argument("--sample-size", type=int, default=20, help="Sample size for LLM evaluation (default: 20)")
    
    try:
        args = parser.parse_args()
        logger.info("Monitoring started with args: days_back=%d, sample_size=%d",
                   args.days_back, args.sample_size)
    except Exception as e:
        logger.exception("Failed to parse arguments: %s", e)
        return 1
    
    try:
        results = run_monitoring(args.days_back, args.sample_size)
        
        if results:
            logger.info("Monitoring job completed successfully with %d result sections", len(results))
            return 0
        else:
            logger.error("Monitoring job completed with no results (possible failure)")
            return 1
            
    except Exception as e:
        logger.exception("Monitoring failed with unhandled exception: %s", e)
        return 1


if __name__ == "__main__":
    exit_code = main()
    # Don't call sys.exit() in Databricks jobs - it can cause false failures
    # The exit code is returned for programmatic use but not explicitly exited
    logger.debug("Monitoring script finished with exit code: %d", exit_code)
