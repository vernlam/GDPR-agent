import json
import pandas as pd
from typing import List, Dict
from databricks.sdk import WorkspaceClient

from .evaluator import evaluate_case

class EvaluationRunner:
    def __init__(self, endpoint_name: str, dataset_path: str):
        self.endpoint_name = endpoint_name
        self.dataset_path = dataset_path
        self.w = WorkspaceClient()
        
    def load_dataset(self) -> dict:
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def run_evaluation(self, limit: int = None) -> pd.DataFrame:
        """Run evaluation on test cases"""
        dataset = self.load_dataset()
        test_cases = dataset["test_cases"][:limit] if limit else dataset["test_cases"]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n{'='*80}")
            print(f"Case: {test_case['id']}")
            print(f"Question: {test_case['question'][:80]}...")
            
            # Query endpoint
            response = self.w.serving_endpoints.query(
                name=self.endpoint_name,
                dataframe_records=[{"question": test_case["question"]}]
            )
            
            agent_response = {
                "answer": response.predictions[0]['answer'],
                "context": response.predictions[0]['context']
            }
            
            # Evaluate
            eval_result = evaluate_case(test_case, agent_response)
            
            # Record
            results.append({
                "case_id": test_case["id"],
                "category": test_case["category"],
                "passed": eval_result["passed"],
                "score": eval_result["scores"]["total"],
                "feedback": " | ".join(eval_result["feedback"])
            })
            
            status = "✅ PASS" if eval_result["passed"] else "❌ FAIL"
            print(f"{status} | Score: {eval_result['scores']['total']:.2f}")
        
        return pd.DataFrame(results)
    
    def print_summary(self, results_df: pd.DataFrame):
        """Print evaluation summary"""
        print(f"\n{'='*80}")
        print(f"📊 EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Total Cases: {len(results_df)}")
        print(f"Passed: {results_df['passed'].sum()}")
        print(f"Failed: {(~results_df['passed']).sum()}")
        print(f"Pass Rate: {results_df['passed'].mean()*100:.1f}%")
        print(f"Avg Score: {results_df['score'].mean():.2f}")
        
        print(f"\n📈 By Category:")
        print(results_df.groupby('category').agg({
            'passed': ['sum', 'count'],
            'score': 'mean'
        }).round(2))
    
    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """Save results to CSV"""
        results_df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to {output_path}")