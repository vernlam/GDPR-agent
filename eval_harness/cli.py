"""
Command-line interface for GDPR Agent evaluation harness.
Orchestrates test execution, metric calculation, and pass/fail determination with CI/CD integration.
"""

import argparse
import logging
import sys
import traceback
from typing import NoReturn

from .runner import EvaluationRunner
from .utils import calculate_metrics, generate_report

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def main() -> NoReturn:
    """
    Execute GDPR Agent evaluation with command-line arguments.
    
    Parses CLI arguments, runs evaluation via EvaluationRunner, saves results,
    generates reports, computes metrics, and exits with appropriate code based
    on pass rate threshold.
    
    CLI Arguments (via argparse):
        --dataset (required): Path to test dataset JSON file
        --threshold: Pass rate threshold (0.0-1.0, default 0.90)
        --experiment: MLflow experiment path (optional)
        --output: CSV output path (default ./eval_results.csv)
        --report: Report output path (default ./eval_report.txt)
    
    Returns:
        NoReturn: Function always exits via sys.exit()
    
    Exit Codes:
        0: Evaluation passed (pass_rate >= threshold)
        1: Evaluation failed (pass_rate < threshold) or exception occurred
        
    Raises:
        This function catches all exceptions and exits with code 1 after logging.
    """
    logger.info("Starting GDPR Agent evaluation CLI")
    
    # Parse command-line arguments
    try:
        parser = argparse.ArgumentParser(description="Run GDPR Agent Evaluation")
        parser.add_argument("--dataset", required=True, 
                            help="Path to test dataset JSON")
        parser.add_argument("--threshold", type=float, default=0.90, 
                            help="Pass rate threshold (0.0-1.0, default 0.90)")
        parser.add_argument("--experiment", default=None, 
                            help="MLflow experiment path")
        parser.add_argument("--output", default="./eval_results.csv",
                            help="CSV output path (default ./eval_results.csv)")
        parser.add_argument("--report", default="./eval_report.txt",
                            help="Report output path (default ./eval_report.txt)")
        
        args = parser.parse_args()
        logger.debug("Arguments parsed successfully")
        logger.debug("Dataset: %s", args.dataset)
        logger.debug("Threshold: %.2f", args.threshold)
        logger.debug("Experiment: %s", args.experiment)
        logger.debug("Output: %s", args.output)
        logger.debug("Report: %s", args.report)
    except Exception as e:
        logger.exception("Failed to parse command-line arguments: %s", e)
        sys.exit(1)
    
    # Write to stderr for CI/CD visibility
    print(f"Running GDPR Agent Evaluation (Direct Mode)", file=sys.stderr)
    print(f"   Threshold: {args.threshold * 100}%", file=sys.stderr)
    
    logger.info("Starting evaluation run")
    logger.info("Threshold: %.1f%%", args.threshold * 100)
    
    # Initialize evaluation runner
    try:
        logger.debug("Initializing EvaluationRunner")
        runner = EvaluationRunner(
            dataset_path=args.dataset,
            experiment_name=args.experiment
        )
        logger.info("EvaluationRunner initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize EvaluationRunner: %s", e)
        print(f"Error: Failed to initialize evaluation runner", file=sys.stderr)
        sys.exit(1)
    
    # Run evaluation
    try:
        logger.debug("Running evaluation")
        results_df = runner.run_evaluation()
        logger.info("Evaluation run completed successfully")
        logger.debug("Results shape: %s", results_df.shape)
        
        logger.debug("Printing summary")
        runner.print_summary(results_df)
        logger.debug("Summary printed")
    except Exception as e:
        logger.exception("Failed to run evaluation: %s", e)
        print(f"Error: Evaluation run failed", file=sys.stderr)
        sys.exit(1)
    
    # Save results to CSV
    try:
        logger.debug("Saving results to CSV: %s", args.output)
        results_df.to_csv(args.output, index=False)
        logger.info("Results saved to: %s", args.output)
    except Exception as e:
        logger.exception("Failed to save results to CSV: %s", e)
        print(f"Error: Failed to save results", file=sys.stderr)
        sys.exit(1)
    
    # Generate report
    try:
        logger.debug("Generating report: %s", args.report)
        generate_report(results_df, output_path=args.report)
        logger.info("Report generated: %s", args.report)
    except Exception as e:
        logger.exception("Failed to generate report: %s", e)
        print(f"Error: Failed to generate report", file=sys.stderr)
        sys.exit(1)
    
    # Calculate metrics and check threshold
    try:
        logger.debug("Calculating metrics")
        metrics = calculate_metrics(results_df)
        pass_rate = metrics['pass_rate']
        logger.info("Pass rate: %.1f%%", pass_rate * 100)
        logger.debug("All metrics: %s", metrics)
    except Exception as e:
        logger.exception("Failed to calculate metrics: %s", e)
        print(f"Error: Failed to calculate metrics", file=sys.stderr)
        sys.exit(1)
    
    # Determine pass/fail and exit
    try:
        if pass_rate >= args.threshold:
            logger.info("Evaluation PASSED: %.1f%% >= %.1f%%", 
                       pass_rate * 100, args.threshold * 100)
            
            # Write to stderr for CI/CD visibility
            print(f"PASSED: {pass_rate*100:.1f}% >= {args.threshold*100}%", file=sys.stderr)
            
            logger.info("Exiting with code 0 (success)")
            sys.exit(0)
        else:
            logger.warning("Evaluation FAILED: %.1f%% < %.1f%%", 
                          pass_rate * 100, args.threshold * 100)
            
            # Write to stderr for CI/CD visibility
            print(f"FAILED: {pass_rate*100:.1f}% < {args.threshold*100}%", file=sys.stderr)
            
            logger.info("Exiting with code 1 (failure)")
            sys.exit(1)
    except Exception as e:
        logger.exception("Failed during threshold comparison: %s", e)
        print(f"Error: Failed to determine pass/fail", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
