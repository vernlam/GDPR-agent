"""
Main monitoring orchestration script.
"""
import sys
import argparse
from datetime import datetime

from monitoring.config import config
from monitoring.utils.databricks_client import DatabricksClient
from monitoring.utils.llm_judge import LLMJudge
from monitoring.monitors.quality_monitor import QualityMonitor
from monitoring.monitors.performance_monitor import PerformanceMonitor
from monitoring.monitors.error_monitor import ErrorMonitor
from monitoring.monitors.cost_monitor import CostMonitor
from monitoring.monitors.drift_monitor import DriftMonitor

def run_monitoring(days_back: int = None, sample_size: int = None):
    """
    Run complete monitoring suite.
    
    Args:
        days_back: Number of days to analyze
        sample_size: Number of queries to evaluate with LLM judge
    """
    days_back = days_back or config.DEFAULT_LOOKBACK_DAYS
    sample_size = sample_size or config.SAMPLE_SIZE_FOR_EVALUATION
    
    print("="*80)
    print(f"🚀 GDPR AGENT MONITORING - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Initialize clients
    db_client = DatabricksClient()
    judge = LLMJudge()
    
    # Initialize monitors
    quality_monitor = QualityMonitor(db_client, judge)
    performance_monitor = PerformanceMonitor(db_client)
    error_monitor = ErrorMonitor(db_client)
    cost_monitor = CostMonitor(db_client)
    drift_monitor = DriftMonitor(db_client)
    
    # Store results
    results = {}
    
    # 1. Check endpoint health
    print("\n1️⃣  Checking Endpoint Health...")
    try:
        endpoint_status = db_client.get_endpoint_status(config.ENDPOINT_NAME)
        print(f"   Status: {endpoint_status['state']}")
        print(f"   URL: {endpoint_status['url']}")
        results['endpoint_status'] = endpoint_status
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['endpoint_status'] = {"error": str(e)}
    
    # 2. Evaluate quality
    print(f"\n2️⃣  Evaluating Quality (LLM Judge)...")
    try:
        quality_df = quality_monitor.evaluate_recent_queries(days_back, sample_size)
        
        if not quality_df.empty:
            quality_summary = quality_monitor.get_quality_summary(days_back)
            print(f"\n   Quality Summary (last {days_back} days):")
            print(f"   - Relevance: {quality_summary['avg_relevance']:.2f}/5")
            print(f"   - Accuracy: {quality_summary['avg_accuracy']:.2f}/5")
            print(f"   - Completeness: {quality_summary['avg_completeness']:.2f}/5")
            print(f"   - Citation: {quality_summary['avg_citation']:.2f}/5")
            print(f"   - Overall: {quality_summary['avg_overall']:.2f}/5")
            print(f"   - Queries Evaluated: {quality_summary['evaluated_queries']}")
            results['quality_summary'] = quality_summary
            
            # Check quality thresholds
            if quality_summary['avg_overall'] < config.MIN_QUALITY_SCORE:
                print(f"\n   ⚠️  WARNING: Overall quality score ({quality_summary['avg_overall']:.2f}) below threshold ({config.MIN_QUALITY_SCORE})")
        else:
            print("   ⚠️  No queries available for quality evaluation")
            results['quality_summary'] = {}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['quality_summary'] = {"error": str(e)}
    
    # 3. Check performance
    print(f"\n3️⃣  Analyzing Performance...")
    try:
        perf_summary = performance_monitor.get_performance_summary(days_back)
        
        if perf_summary:
            print(f"\n   Performance Summary (last {days_back} days):")
            print(f"   - Total Requests: {perf_summary['total_requests']}")
            print(f"   - Avg Daily Requests: {perf_summary['avg_daily_requests']:.1f}")
            print(f"   - Avg Latency: {perf_summary['avg_latency_seconds']:.2f}s")
            print(f"   - Max Latency: {perf_summary['max_latency_seconds']:.2f}s")
            print(f"   - Success Rate: {perf_summary['avg_success_rate']:.1f}%")
            results['performance_summary'] = perf_summary
            
            # Check performance thresholds
            if perf_summary['avg_latency_seconds'] > config.MAX_AVG_LATENCY:
                print(f"\n   ⚠️  WARNING: Avg latency ({perf_summary['avg_latency_seconds']:.2f}s) exceeds threshold ({config.MAX_AVG_LATENCY}s)")
            if perf_summary['avg_success_rate'] < config.MIN_SUCCESS_RATE:
                print(f"\n   ⚠️  WARNING: Success rate ({perf_summary['avg_success_rate']:.1f}%) below threshold ({config.MIN_SUCCESS_RATE}%)")
        else:
            print("   ⚠️  No performance data available")
            results['performance_summary'] = {}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['performance_summary'] = {"error": str(e)}
    
    # 4. Check for errors
    print(f"\n4️⃣  Analyzing Errors...")
    try:
        error_summary = error_monitor.get_error_summary(days_back)
        
        if error_summary:
            print(f"\n   Error Summary (last {days_back} days):")
            print(f"   - Total Errors: {error_summary['total_errors']}")
            print(f"   - Error Rate: {error_summary['error_rate']:.2f}%")
            print(f"   - Days with Errors: {error_summary['days_with_errors']}")
            
            if error_summary['errors_by_type']:
                print(f"   - Error Types:")
                for error_type, count in error_summary['errors_by_type'].items():
                    print(f"     • {error_type}: {count}")
            
            results['error_summary'] = error_summary
            
            # Check for alerts
            alerts = error_monitor.check_alert_thresholds(days_back=1)
            if alerts:
                print(f"\n   🚨 {len(alerts)} Alert(s) Triggered:")
                for alert in alerts:
                    print(f"   - [{alert['severity']}] {alert['message']}")
                results['alerts'] = alerts
            
            # Get error patterns
            error_patterns = error_monitor.get_error_patterns(days_back, top_n=5)
            if not error_patterns.empty:
                print(f"\n   Top Error Patterns:")
                for idx, row in error_patterns.iterrows():
                    print(f"   - {row['question'][:60]}... (occurred {row['occurrence_count']} times)")
        else:
            print("   ✅ No errors detected")
            results['error_summary'] = error_summary
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['error_summary'] = {"error": str(e)}
    
    # 5. Cost analysis
    print(f"\n5️⃣  Analyzing Costs...")
    try:
        cost_summary = cost_monitor.get_cost_summary(days_back)
        
        if cost_summary:
            print(f"\n   Cost Summary (last {days_back} days):")
            print(f"   - Total Cost: ${cost_summary['total_cost_usd']:.2f}")
            print(f"   - Avg Daily Cost: ${cost_summary['avg_daily_cost_usd']:.2f}")
            print(f"   - Cost per Request: ${cost_summary['avg_cost_per_request']:.4f}")
            print(f"   - Projected Monthly: ${cost_summary['projected_monthly_cost']:.2f}")
            print(f"   - Total Input Tokens: {cost_summary['total_input_tokens']:,}")
            print(f"   - Total Output Tokens: {cost_summary['total_output_tokens']:,}")
            results['cost_summary'] = cost_summary
            
            # Check for cost anomalies
            anomalies = cost_monitor.identify_cost_anomalies(days_back)
            if not anomalies.empty:
                print(f"\n   ⚠️  Cost Anomalies Detected ({len(anomalies)} days):")
                for idx, row in anomalies.head(3).iterrows():
                    print(f"   - {row['date']}: ${row['estimated_cost_usd']:.2f} ({row['deviation_from_mean']:.1f}σ from mean)")
        else:
            print("   ⚠️  No cost data available")
            results['cost_summary'] = {}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['cost_summary'] = {"error": str(e)}
    
    # 6. Drift detection
    print(f"\n6️⃣  Detecting Drift...")
    try:
        drift_results = drift_monitor.detect_distribution_drift()
        
        if drift_results:
            print(f"\n   Drift Analysis:")
            drift_status = '⚠️  YES' if drift_results['drift_detected'] else '✅ NO'
            print(f"   - Drift Detected: {drift_status}")
            print(f"   - Drift Score: {drift_results['drift_score']:.3f} (threshold: {drift_results['drift_threshold']})")
            print(f"   - Baseline Queries: {drift_results['baseline_unique_queries']}")
            print(f"   - Recent Queries: {drift_results['recent_unique_queries']}")
            print(f"   - New Queries: {drift_results['new_query_count']} ({drift_results['new_query_rate']:.1%})")
            print(f"   - Disappeared Queries: {drift_results['disappeared_query_count']} ({drift_results['disappeared_query_rate']:.1%})")
            results['drift_results'] = drift_results
            
            # Get top queries
            top_queries = drift_monitor.get_top_queries(days_back, top_n=5)
            if not top_queries.empty:
                print(f"\n   Top 5 Queries:")
                for idx, row in top_queries.iterrows():
                    print(f"   - {row['question'][:60]}... ({row['frequency']} times, {row['percentage']:.1f}%)")
            
            # Get keyword trends
            keywords = drift_monitor.get_keyword_distribution(days_back, top_n=10)
            if not keywords.empty:
                print(f"\n   Top Keywords:")
                top_keywords = keywords.head(10)['keyword'].tolist()
                print(f"   {', '.join(top_keywords)}")
        else:
            print("   ⚠️  Insufficient data for drift detection")
            results['drift_results'] = {}
    except Exception as e:
        print(f"   ❌ Error: {e}")
        results['drift_results'] = {"error": str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("✅ Monitoring Complete!")
    print("="*80)
    
    # Print critical alerts summary
    critical_alerts = []
    
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
        print("\n🚨 CRITICAL ALERTS:")
        for alert in critical_alerts:
            print(f"   - {alert}")
    else:
        print("\n✅ All metrics within acceptable thresholds")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GDPR Agent monitoring")
    parser.add_argument("--days-back", type=int, default=7, help="Days to look back")
    parser.add_argument("--sample-size", type=int, default=20, help="Sample size for LLM evaluation")
    
    args = parser.parse_args()
    
    try:
        results = run_monitoring(args.days_back, args.sample_size)
        sys.exit(0)
    except Exception as e:
        print(f"❌ Monitoring failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)