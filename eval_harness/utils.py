"""
Utility functions for GDPR Agent evaluation harness.
Provides dataset loading, metrics calculation, report generation, and results comparison.
"""

import logging
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_dataset(path: str) -> Dict:
    """
    Load test dataset from JSON file.
    
    Args:
        path: Path to JSON file containing test cases
        
    Returns:
        Dictionary containing test cases and metadata
        
    Raises:
        Exception: If file cannot be read or JSON is invalid (logged but not re-raised)
    """
    logger.info("Loading test dataset from %s", path)
    
    try:
        with open(path, 'r') as f:
            dataset = json.load(f)
        
        num_cases = len(dataset.get("test_cases", []))
        logger.info("Test dataset loaded successfully: %d test cases", num_cases)
        logger.debug("Dataset keys: %s", list(dataset.keys()))
        
        return dataset
        
    except FileNotFoundError as e:
        logger.exception("Test dataset file not found: %s", path)
        raise
    except json.JSONDecodeError as e:
        logger.exception("Invalid JSON in test dataset file: %s", e)
        raise
    except Exception as e:
        logger.exception("Failed to load test dataset: %s", e)
        raise


def save_results(results: pd.DataFrame, output_dir: str, run_name: Optional[str] = None) -> str:
    """
    Save evaluation results to CSV with timestamp.
    
    Args:
        results: DataFrame containing evaluation results
        output_dir: Directory to save results file
        run_name: Optional run name (defaults to timestamp)
        
    Returns:
        Path to saved results file as string
        
    Raises:
        Exception: If file cannot be written or directory doesn't exist (logged but not re-raised)
    """
    logger.info("Saving evaluation results to %s", output_dir)
    
    try:
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
            logger.debug("Using generated run name: %s", run_name)
        else:
            logger.debug("Using provided run name: %s", run_name)
        
        output_path = Path(output_dir) / f"eval_results_{run_name}.csv"
        logger.debug("Output file path: %s", output_path)
        
        results.to_csv(output_path, index=False)
        
        logger.info("Results saved successfully to %s", output_path)
        logger.debug("Saved %d rows to CSV", len(results))
        
        return str(output_path)
        
    except FileNotFoundError as e:
        logger.exception("Output directory not found: %s", output_dir)
        raise
    except PermissionError as e:
        logger.exception("Permission denied writing to %s", output_dir)
        raise
    except Exception as e:
        logger.exception("Failed to save results: %s", e)
        raise


def calculate_metrics(results_df: pd.DataFrame) -> Dict[str, Optional[float]]:
    """
    Calculate aggregate metrics from evaluation results.
    
    Args:
        results_df: DataFrame containing evaluation results with 'passed' and 'score' columns
        
    Returns:
        Dictionary containing metrics: total_cases, passed, failed, pass_rate, avg_score,
        source_accuracy, content_match
        
    Raises:
        Exception: If required columns are missing or calculation fails (logged but not re-raised)
    """
    logger.debug("Calculating aggregate metrics from results")
    
    try:
        metrics = {
            "total_cases": len(results_df),
            "passed": int(results_df['passed'].sum()),
            "failed": int((~results_df['passed']).sum()),
            "pass_rate": float(results_df['passed'].mean()),
            "avg_score": float(results_df['score'].mean()),
            "source_accuracy": float(results_df.get('source_correct', pd.Series([0])).mean()) if 'source_correct' in results_df else None,
            "content_match": float(results_df.get('content_match', pd.Series([0])).mean()) if 'content_match' in results_df else None
        }
        
        logger.info("Metrics calculated: total=%d, passed=%d, failed=%d, pass_rate=%.2f", 
                   metrics['total_cases'], metrics['passed'], metrics['failed'], metrics['pass_rate'])
        logger.debug("Additional metrics: avg_score=%.3f, source_accuracy=%s, content_match=%s",
                    metrics['avg_score'], metrics['source_accuracy'], metrics['content_match'])
        
        return metrics
        
    except KeyError as e:
        logger.exception("Required column missing from results DataFrame: %s", e)
        raise
    except Exception as e:
        logger.exception("Failed to calculate metrics: %s", e)
        raise


def print_category_breakdown(results_df: pd.DataFrame) -> None:
    """
    Display detailed breakdown of results by category.
    
    Args:
        results_df: DataFrame containing evaluation results with 'category', 'passed', and 'score' columns
        
    Raises:
        Exception: If required columns are missing or aggregation fails (logged but not re-raised)
    """
    logger.info("Generating category breakdown")
    
    try:
        logger.info("=" * 80)
        logger.info("Results by Category")
        logger.info("=" * 80)
        
        category_stats = results_df.groupby('category').agg({
            'passed': ['sum', 'count', lambda x: f"{x.sum()}/{len(x)}"],
            'score': ['mean', 'std']
        }).round(3)
        
        category_stats.columns = ['Passed', 'Total', 'Pass/Total', 'Avg Score', 'Std Dev']
        
        logger.debug("Category statistics calculated for %d categories", len(category_stats))
        
        # Log the DataFrame as string
        logger.info("\n%s", category_stats.to_string())
        
        logger.debug("Category breakdown display complete")
        
    except KeyError as e:
        logger.exception("Required column missing for category breakdown: %s", e)
        raise
    except Exception as e:
        logger.exception("Failed to generate category breakdown: %s", e)
        raise


def filter_failed_cases(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only failed test cases sorted by score for debugging.
    
    Args:
        results_df: DataFrame containing evaluation results with 'passed' and 'score' columns
        
    Returns:
        DataFrame containing only failed cases, sorted by score (lowest first)
        
    Raises:
        Exception: If required columns are missing or filtering fails (logged but not re-raised)
    """
    logger.debug("Filtering failed test cases")
    
    try:
        failed = results_df[~results_df['passed']].copy()
        failed = failed.sort_values('score')
        
        logger.info("Filtered %d failed cases from %d total cases", len(failed), len(results_df))
        
        if len(failed) > 0:
            logger.debug("Failed cases score range: %.3f to %.3f", failed['score'].min(), failed['score'].max())
        
        return failed
        
    except KeyError as e:
        logger.exception("Required column missing for filtering failed cases: %s", e)
        raise
    except Exception as e:
        logger.exception("Failed to filter failed cases: %s", e)
        raise


def generate_report(results_df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """
    Generate comprehensive evaluation report with metrics and failed cases.
    
    Args:
        results_df: DataFrame containing evaluation results
        output_path: Optional path to save report as text file
        
    Returns:
        Formatted report as string
        
    Raises:
        Exception: If report generation or file writing fails (logged but not re-raised)
    """
    logger.info("Generating comprehensive evaluation report")
    
    try:
        # Calculate metrics
        logger.debug("Calculating metrics for report")
        metrics = calculate_metrics(results_df)
        
        # Build report header
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
        logger.debug("Adding category breakdown to report")
        category_stats = results_df.groupby('category').agg({
            'passed': ['sum', 'count'],
            'score': 'mean'
        }).round(3)
        
        for category in category_stats.index:
            passed = category_stats.loc[category, ('passed', 'sum')]
            total = category_stats.loc[category, ('passed', 'count')]
            score = category_stats.loc[category, ('score', 'mean')]
            report += f"{category}: {passed}/{total} passed (avg score: {score:.3f})\n"
        
        logger.debug("Added %d categories to report", len(category_stats))
        
        # Add failed cases
        logger.debug("Adding failed cases to report")
        failed = filter_failed_cases(results_df)
        
        if len(failed) > 0:
            report += f"\n{'-'*80}\nFAILED CASES ({len(failed)})\n{'-'*80}\n"
            for _, row in failed.iterrows():
                report += f"• {row['case_id']} | Score: {row['score']:.2f} | {row['category']}\n"
                report += f"  {row.get('feedback', 'No feedback')}\n\n"
            logger.debug("Added %d failed cases to report", len(failed))
        else:
            logger.debug("No failed cases to add to report")
        
        report += f"{'='*80}\n"
        
        # Save to file if path provided
        if output_path:
            try:
                logger.debug("Saving report to file: %s", output_path)
                with open(output_path, 'w') as f:
                    f.write(report)
                logger.info("Report saved to %s", output_path)
            except Exception as e:
                logger.exception("Failed to save report to file %s: %s", output_path, e)
                raise
        
        logger.info("Report generation complete")
        return report
        
    except Exception as e:
        logger.exception("Failed to generate report: %s", e)
        raise


def validate_test_case(test_case: Dict) -> bool:
    """
    Validate that test case has all required fields.
    
    Args:
        test_case: Dictionary containing test case data
        
    Returns:
        True if all required fields present, False otherwise
        
    Raises:
        Exception: If validation check fails (logged but not re-raised)
    """
    logger.debug("Validating test case")
    
    try:
        required = ['id', 'question', 'category', 'expected_behavior']
        is_valid = all(field in test_case for field in required)
        
        if is_valid:
            logger.debug("Test case validation passed: %s", test_case.get('id', 'unknown'))
        else:
            missing = [field for field in required if field not in test_case]
            logger.warning("Test case validation failed, missing fields: %s", missing)
        
        return is_valid
        
    except Exception as e:
        logger.exception("Failed to validate test case: %s", e)
        raise


def compare_runs(results_df1: pd.DataFrame, results_df2: pd.DataFrame, 
                 label1: str = "Run 1", label2: str = "Run 2") -> pd.DataFrame:
    """
    Compare metrics between two evaluation runs.
    
    Args:
        results_df1: DataFrame containing first run results
        results_df2: DataFrame containing second run results
        label1: Label for first run (default: "Run 1")
        label2: Label for second run (default: "Run 2")
        
    Returns:
        DataFrame containing comparison of metrics with delta
        
    Raises:
        Exception: If metric calculation or comparison fails (logged but not re-raised)
    """
    logger.info("Comparing evaluation runs: %s vs %s", label1, label2)
    
    try:
        # Calculate metrics for both runs
        logger.debug("Calculating metrics for %s", label1)
        metrics1 = calculate_metrics(results_df1)
        
        logger.debug("Calculating metrics for %s", label2)
        metrics2 = calculate_metrics(results_df2)
        
        # Build comparison DataFrame
        logger.debug("Building comparison DataFrame")
        comparison = pd.DataFrame({
            label1: metrics1,
            label2: metrics2,
            'Delta': [metrics2[k] - metrics1[k] if isinstance(metrics1[k], (int, float)) and isinstance(metrics2[k], (int, float)) else None 
                      for k in metrics1.keys()]
        })
        
        logger.info("=" * 80)
        logger.info("Comparison: %s vs %s", label1, label2)
        logger.info("=" * 80)
        logger.info("\n%s", comparison.to_string())
        
        logger.info("Run comparison complete")
        
        return comparison
        
    except Exception as e:
        logger.exception("Failed to compare runs: %s", e)
        raise
