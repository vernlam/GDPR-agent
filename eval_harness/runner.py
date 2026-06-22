# eval_harness/runner.py
import sys
import json
import pandas as pd
import mlflow
import os
from pathlib import Path
from datetime import datetime
from .evaluator import evaluate_case
from gdpr_agent.agent import GDPRAgent

class EvaluationRunner:
    def __init__(self, dataset_path: str, experiment_name: str = None):
        """Direct evaluation by importing and building the agent"""
        self.dataset_path = dataset_path
        self.agent = GDPRAgent()
        
        # Set up MLflow
        if experiment_name is None:
            experiment_name = "/Shared/gdpr-agent-ci-evaluation"
        mlflow.set_experiment(experiment_name)
    
    def load_dataset(self) -> dict:
        with open(self.dataset_path, 'r') as f:
            return json.load(f)
    
    def run_evaluation(self, limit: int = None) -> pd.DataFrame:
        """Run evaluation by calling agent directly"""
        dataset = self.load_dataset()
        test_cases = dataset["test_cases"][:limit] if limit else dataset["test_cases"]
        
        with mlflow.start_run(run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_param("evaluation_mode", "direct")
            mlflow.log_param("dataset_path", self.dataset_path)
            mlflow.log_param("num_test_cases", len(test_cases))
            
            results = []
            category_scores = {}
            
            for idx, test_case in enumerate(test_cases):
                print(f"\n{'='*80}")
                print(f"[{idx+1}/{len(test_cases)}] Case: {test_case['id']}")
                print(f"Question: {test_case['question'][:80]}...")
                
                # Call agent directly using your state structure
                state = {
                    "original_question": test_case["question"],
                    "current_query": test_case["question"],
                    "retrieved_context": "",
                    "generated_answer": "",
                    "retrieval_loop_count": 0,
                    "generation_loop_count": 0
                }
                
                try:
                    final_state = self.agent_app.invoke(state)
                    agent_response = {
                        "answer": final_state.get("generated_answer", ""),
                        "context": final_state.get("retrieved_context", "")
                    }
                except Exception as e:
                    print(f"❌ Agent error: {e}")
                    agent_response = {"answer": "", "context": ""}
                
                # Evaluate
                eval_result = evaluate_case(test_case, agent_response)
                
                # Track by category
                category = test_case["category"]
                if category not in category_scores:
                    category_scores[category] = []
                category_scores[category].append(eval_result["scores"]["total"])
                
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
            
            # Log metrics
            mlflow.log_metric("pass_rate", results_df['passed'].mean())
            mlflow.log_metric("avg_score", results_df['score'].mean())
            mlflow.log_metric("total_passed", int(results_df['passed'].sum()))
            mlflow.log_metric("total_failed", int((~results_df['passed']).sum()))
            
            # Log category metrics
            for category, scores in category_scores.items():
                mlflow.log_metric(f"pass_rate_{category}", sum(s >= 0.7 for s in scores) / len(scores))
            
            # Log artifacts
            results_df.to_csv("evaluation_results.csv", index=False)
            mlflow.log_artifact("evaluation_results.csv")
            
            failed_df = results_df[~results_df['passed']]
            if len(failed_df) > 0:
                failed_df.to_csv("failed_cases.csv", index=False)
                mlflow.log_artifact("failed_cases.csv")
            
            return results_df
    
    def print_summary(self, results_df: pd.DataFrame):
        print(f"\n{'='*80}")
        print(f"📊 EVALUATION RESULTS")
        print(f"{'='*80}")
        print(f"Total Cases: {len(results_df)}")
        print(f"Passed: {results_df['passed'].sum()}")
        print(f"Failed: {(~results_df['passed']).sum()}")
        print(f"Pass Rate: {results_df['passed'].mean()*100:.1f}%")
        print(f"Avg Score: {results_df['score'].mean():.2f}")