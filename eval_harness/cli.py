# eval_harness/cli.py
import sys
import argparse
from .runner import EvaluationRunner
from .utils import generate_report, calculate_metrics

def main():
    parser = argparse.ArgumentParser(description="Run GDPR Agent Evaluation")
    parser.add_argument("--dataset", required=True, help="Path to test dataset JSON")
    parser.add_argument("--threshold", type=float, default=0.90, help="Pass rate threshold")
    parser.add_argument("--experiment", default=None, help="MLflow experiment path")
    parser.add_argument("--output", default="./eval_results.csv")
    parser.add_argument("--report", default="./eval_report.txt")
    
    args = parser.parse_args()
    
    print(f"🚀 Running GDPR Agent Evaluation (Direct Mode)")
    print(f"   Threshold: {args.threshold * 100}%\n")
    
    try:
        runner = EvaluationRunner(
            dataset_path=args.dataset,
            experiment_name=args.experiment
        )
        
        results_df = runner.run_evaluation()
        runner.print_summary(results_df)
        
        # Save results
        results_df.to_csv(args.output, index=False)
        generate_report(results_df, output_path=args.report)
        
        # Check threshold
        metrics = calculate_metrics(results_df)
        pass_rate = metrics['pass_rate']
        
        if pass_rate >= args.threshold:
            print(f"\n✅ PASSED: {pass_rate*100:.1f}% >= {args.threshold*100}%")
            sys.exit(0)
        else:
            print(f"\n❌ FAILED: {pass_rate*100:.1f}% < {args.threshold*100}%")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()