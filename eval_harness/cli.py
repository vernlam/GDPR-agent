# eval_harness/cli.py
import sys
import argparse
from .runner import EvaluationRunner
from .utils import generate_report, print_category_breakdown, calculate_metrics

def main():
    parser = argparse.ArgumentParser(description="Run GDPR Agent Evaluation")
    parser.add_argument("--endpoint", required=True, help="Model serving endpoint name")
    parser.add_argument("--dataset", required=True, help="Path to test dataset JSON")
    parser.add_argument("--threshold", type=float, default=0.90, help="Pass rate threshold (default: 0.90)")
    parser.add_argument("--output", default="./eval_results.csv", help="Output CSV path")
    parser.add_argument("--report", default="./eval_report.txt", help="Report output path")
    
    args = parser.parse_args()
    
    print(f"🚀 Running GDPR Agent Evaluation")
    print(f"   Endpoint: {args.endpoint}")
    print(f"   Dataset: {args.dataset}")
    print(f"   Threshold: {args.threshold * 100}%\n")
    
    try:
        runner = EvaluationRunner(
            endpoint_name=args.endpoint,
            dataset_path=args.dataset
        )
        
        results_df = runner.run_evaluation()
        runner.print_summary(results_df)
        
        # Save results
        results_df.to_csv(args.output, index=False)
        print(f"\n💾 Results saved to {args.output}")
        
        # Generate report
        generate_report(results_df, output_path=args.report)
        
        # Check threshold
        metrics = calculate_metrics(results_df)
        pass_rate = metrics['pass_rate']
        
        print(f"\n{'='*80}")
        if pass_rate >= args.threshold:
            print(f"✅ EVALUATION PASSED: {pass_rate*100:.1f}% >= {args.threshold*100}%")
            sys.exit(0)
        else:
            print(f"❌ EVALUATION FAILED: {pass_rate*100:.1f}% < {args.threshold*100}%")
            print(f"   Required improvement: {(args.threshold - pass_rate)*100:.1f} percentage points")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()