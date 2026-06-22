# eval_harness/runner.py
import json
import pandas as pd
import mlflow
from typing import List, Dict
from databricks.sdk import WorkspaceClient
from datetime import datetime

from .evaluator import evaluate_case

class EvaluationRunner:
    def __init__(self, endpoint_name: str, dataset_path: str, experiment_name: str = None):
        self.endpoint_name = endpoint_name
        self.dataset_path = dataset_path
        self.w = WorkspaceClient()
        
        # Set up MLflow experiment
        if experiment_name is None:
            experiment_name = f"/Users/{self.w.current_user.me().user_name}/gdpr-agent-evaluation"
        
        mlflow.set_experiment(experiment_name)
        
    def load_dataset(self) -> dict:
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def run_evaluation(self, limit: int = None) -> pd.DataFrame:
        """Run evaluation on test cases and log to MLflow"""
        dataset = self.load_dataset()
        test_cases = dataset["test_cases"][:limit] if limit else dataset["test_cases"]
        
        # Start MLflow run
        with mlflow.start_run(run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("endpoint_name", self.endpoint_name)
            mlflow.log_param("dataset_path", self.dataset_path)
            mlflow.log_param("num_test_cases", len(test_cases))
            mlflow.log_param("dataset_version", dataset.get("version", "unknown"))
            
            results = []
            category_scores = {}
            
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
                
                # Track by category
                category = test_case["category"]
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(eval_result["scores"]["total"])
                
                # Record
                results.append({
                    "case_id": test_case["id"],
                    "category": category,
                    "passed": eval_result["passed"],
                    "score": eval_result["scores"]["total"],
                    "source_correct": eval_result["scores"]["source_correct"],
                    "content_match": eval_result["scores"]["content_match"],
                    "feedback": " | ".join(eval_result["feedback"])
                })
                
                status = "✅ PASS" if eval_result["passed"] else "❌ FAIL"
                print(f"{status} | Score: {eval_result['scores']['total']:.2f}")
            
            results_df = pd.DataFrame(results)
            
            # Log overall metrics
            mlflow.log_metric("pass_rate", results_df['passed'].mean())
            mlflow.log_metric("avg_score", results_df['score'].mean())
            mlflow.log_metric("total_passed", int(results_df['passed'].sum()))
            mlflow.log_metric("total_failed", int((~results_df['passed']).sum()))
            mlflow.log_metric("source_accuracy", results_df['source_correct'].mean())
            mlflow.log_metric("content_match_avg", results_df['content_match'].mean())
            
            # Log category-level metrics
            for category, scores in category_scores.items():
                mlflow.log_metric(f"pass_rate_{category}", sum(s >= 0.7 for s in scores) / len(scores))
                mlflow.log_metric(f"avg_score_{category}", sum(scores) / len(scores))
            
            # Log artifacts
            results_df.to_csv("evaluation_results.csv", index=False)
            mlflow.log_artifact("evaluation_results.csv")
            
            # Log failed cases for debugging
            failed_df = results_df[~results_df['passed']]
            if len(failed_df) > 0:
                failed_df.to_csv("failed_cases.csv", index=False)
                mlflow.log_artifact("failed_cases.csv")
            
            # Log dataset metadata
            with open("dataset_info.json", "w") as f:
                json.dump({
                    "version": dataset.get("version", "unknown"),
                    "total_cases": len(test_cases),
                    "categories": list(category_scores.keys())
                }, f, indent=2)
            mlflow.log_artifact("dataset_info.json")
            
            print(f"\n✅ Results logged to MLflow experiment")
            print(f"Run ID: {mlflow.active_run().info.run_id}")
            
            return results_df
    
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