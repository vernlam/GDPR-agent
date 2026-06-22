# eval_harness/utils.py

import json
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
from pathlib import Path


def load_test_dataset(path: str) -> dict:
    """Load test dataset from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def save_results(results: pd.DataFrame, output_dir: str, run_name: str = None):
    """Save evaluation results with timestamp"""
    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = Path(output_dir) / f"eval_results_{run_name}.csv"
    results.to_csv(output_path, index=False)
    
    print(f"💾 Results saved to {output_path}")
    return str(output_path)


def calculate_metrics(results_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate aggregate metrics from results"""
    return {
        "total_cases": len(results_df),
        "passed": int(results_df['passed'].sum()),
        "failed": int((~results_df['passed']).sum()),
        "pass_rate": float(results_df['passed'].mean()),
        "avg_score": float(results_df['score'].mean()),
        "source_accuracy": float(results_df.get('source_correct', pd.Series([0])).mean()) if 'source_correct' in results_df else None,
        "content_match": float(results_df.get('content_match', pd.Series([0])).mean()) if 'content_match' in results_df else None
    }


def print_category_breakdown(results_df: pd.DataFrame):
    """Print detailed breakdown by category"""
    print(f"\n{'='*80}")
    print(f"📈 Results by Category")
    print(f"{'='*80}\n")
    
    category_stats = results_df.groupby('category').agg({
        'passed': ['sum', 'count', lambda x: f"{x.sum()}/{len(x)}"],
        'score': ['mean', 'std']
    }).round(3)
    
    category_stats.columns = ['Passed', 'Total', 'Pass/Total', 'Avg Score', 'Std Dev']
    print(category_stats)


def filter_failed_cases(results_df: pd.DataFrame) -> pd.DataFrame:
    """Return only failed test cases for debugging"""
    failed = results_df[~results_df['passed']].copy()
    failed = failed.sort_values('score')
    return failed


def generate_report(results_df: pd.DataFrame, output_path: str = None):
    """Generate comprehensive evaluation report"""
    metrics = calculate_metrics(results_df)
    
    report = f"""
{'='*80}
GDPR Agent Evaluation Report
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL METRICS
{'-'*80}
Total Test Cases: {metrics['total_cases']}
Passed: {metrics['passed']} ({metrics['pass_rate']*100:.1f}%)
Failed: {metrics['failed']}
Average Score: {metrics['avg_score']:.3f}

CATEGORY BREAKDOWN
{'-'*80}
"""
    
    # Add category breakdown
    category_stats = results_df.groupby('category').agg({
        'passed': ['sum', 'count'],
        'score': 'mean'
    }).round(3)
    
    for category in category_stats.index:
        passed = category_stats.loc[category, ('passed', 'sum')]
        total = category_stats.loc[category, ('passed', 'count')]
        score = category_stats.loc[category, ('score', 'mean')]
        report += f"{category}: {passed}/{total} passed (avg score: {score:.3f})\n"
    
    # Failed cases
    failed = filter_failed_cases(results_df)
    if len(failed) > 0:
        report += f"\n{'-'*80}\nFAILED CASES ({len(failed)})\n{'-'*80}\n"
        for _, row in failed.iterrows():
            report += f"• {row['case_id']} | Score: {row['score']:.2f} | {row['category']}\n"
            report += f"  {row.get('feedback', 'No feedback')}\n\n"
    
    report += f"{'='*80}\n"
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"📄 Report saved to {output_path}")
    
    return report


def validate_test_case(test_case: dict) -> bool:
    """Validate test case has required fields"""
    required = ['id', 'question', 'category', 'expected_behavior']
    return all(field in test_case for field in required)


def compare_runs(results_df1: pd.DataFrame, results_df2: pd.DataFrame, 
                 label1: str = "Run 1", label2: str = "Run 2"):
    """Compare two evaluation runs"""
    metrics1 = calculate_metrics(results_df1)
    metrics2 = calculate_metrics(results_df2)
    
    print(f"\n{'='*80}")
    print(f"📊 Comparison: {label1} vs {label2}")
    print(f"{'='*80}\n")
    
    comparison = pd.DataFrame({
        label1: metrics1,
        label2: metrics2,
        'Delta': [metrics2[k] - metrics1[k] if isinstance(metrics1[k], (int, float)) else None 
                  for k in metrics1.keys()]
    })
    
    print(comparison)
    return comparison