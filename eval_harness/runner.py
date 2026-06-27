"""
Evaluation runner orchestrating agent invocation and scoring across test cases.
Loads datasets, invokes GDPR agent, scores responses, and logs metrics to MLflow.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import mlflow

from .evaluator import evaluate_case
from gdpr_agent.agent import GDPRAgent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class EvaluationRunner:
    """
    Orchestrates end-to-end evaluation of GDPR Agent across test datasets.
    
    Loads test cases from JSON, invokes the agent for each question, scores responses
    against expected behavior, tracks metrics by category, and logs all results to MLflow.
    Handles agent failures gracefully and generates CSV artifacts for analysis.
    """
    
    def __init__(self, dataset_path: str, experiment_name: Optional[str] = None) -> None:
        """
        Initialize evaluation runner with dataset and MLflow experiment.
        
        Args:
            dataset_path: Path to JSON file containing test cases
            experiment_name: MLflow experiment name (defaults to shared CI experiment)
            
        Raises:
            Exception: If agent initialization or MLflow setup fails (logged but not re-raised)
        """
        logger.info("Initializing EvaluationRunner")
        logger.debug("Dataset path: %s", dataset_path)
        
        self.dataset_path = dataset_path
        
        # Initialize agent
        try:
            logger.debug("Initializing GDPR agent")
            self.agent = GDPRAgent()
            logger.info("GDPR agent initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize GDPR agent: %s", e)
            raise
        
        # Set up MLflow experiment
        try:
            if experiment_name is None:
                experiment_name = "/Shared/gdpr-agent-ci-evaluation"
                logger.debug("Using default MLflow experiment: %s", experiment_name)
            else:
                logger.debug("Using custom MLflow experiment: %s", experiment_name)
            
            mlflow.set_experiment(experiment_name)
            logger.info("MLflow experiment set: %s", experiment_name)
            
        except Exception as e:
            logger.exception("Failed to set MLflow experiment: %s", e)
            raise
    
    def load_dataset(self) -> Dict:
        """
        Load test cases from JSON dataset file.
        
        Returns:
            Dictionary containing test cases and metadata
            
        Raises:
            Exception: If dataset file cannot be read or parsed (logged but not re-raised)
        """
        logger.info("Loading dataset from %s", self.dataset_path)
        
        try:
            with open(self.dataset_path, 'r') as f:
                dataset = json.load(f)
            
            num_cases = len(dataset.get("test_cases", []))
            logger.info("Dataset loaded successfully: %d test cases", num_cases)
            logger.debug("Dataset keys: %s", list(dataset.keys()))
            
            return dataset
            
        except FileNotFoundError as e:
            logger.exception("Dataset file not found: %s", self.dataset_path)
            raise
        except json.JSONDecodeError as e:
            logger.exception("Invalid JSON in dataset file: %s", e)
            raise
        except Exception as e:
            logger.exception("Failed to load dataset: %s", e)
            raise
    
    def run_evaluation(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Execute evaluation across all test cases with agent invocation and scoring.
        
        Iterates through test cases, invokes agent for each question, evaluates responses
        against expected behavior, tracks pass/fail by category, and logs all metrics
        and artifacts to MLflow. Handles agent errors gracefully with empty responses.
        
        Args:
            limit: Maximum number of test cases to evaluate (None for all)
            
        Returns:
            DataFrame containing evaluation results with scores and feedback per case
            
        Raises:
            Exception: If dataset loading or MLflow operations fail (logged but not re-raised)
        """
        logger.info("Starting evaluation run")
        
        # Load and prepare test cases
        try:
            dataset = self.load_dataset()
            test_cases = dataset["test_cases"][:limit] if limit else dataset["test_cases"]
            logger.info("Prepared %d test cases for evaluation", len(test_cases))
            
            if limit:
                logger.debug("Applied limit: evaluating first %d cases", limit)
                
        except Exception as e:
            logger.exception("Failed to load test cases: %s", e)
            raise
        
        # Start MLflow run
        run_name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info("Starting MLflow run: %s", run_name)
        
        try:
            with mlflow.start_run(run_name=run_name):
                # Log parameters
                try:
                    mlflow.log_param("evaluation_mode", "direct")
                    mlflow.log_param("dataset_path", self.dataset_path)
                    mlflow.log_param("num_test_cases", len(test_cases))
                    logger.debug("MLflow parameters logged")
                except Exception as e:
                    logger.exception("Failed to log MLflow parameters: %s", e)
                
                results: List[Dict] = []
                category_scores: Dict[str, List[float]] = {}
                
                # Evaluate each test case
                logger.info("Beginning test case iteration")
                
                for idx, test_case in enumerate(test_cases):
                    case_id = test_case['id']
                    question = test_case['question']
                    category = test_case['category']
                    
                    logger.info("=" * 80)
                    logger.info("Evaluating case [%d/%d]: %s", idx + 1, len(test_cases), case_id)
                    logger.debug("Category: %s", category)
                    logger.debug("Question preview: %s...", question[:80])
                    
                    # Invoke agent
                    try:
                        logger.debug("Invoking agent with question")
                        result = self.agent.invoke({"question": question})
                        
                        agent_response = {
                            "answer": result.get("answer", ""),
                            "context": result.get("context", "")
                        }
                        
                        logger.debug("Agent responded successfully")
                        logger.debug("Answer length: %d chars", len(agent_response["answer"]))
                        logger.debug("Context length: %d chars", len(str(agent_response["context"])))
                        
                    except Exception as e:
                        logger.exception("Agent invocation failed for case %s: %s", case_id, e)
                        agent_response = {"answer": "", "context": ""}
                        logger.warning("Using empty response for failed case %s", case_id)
                    
                    # Evaluate response
                    try:
                        logger.debug("Evaluating agent response")
                        eval_result = evaluate_case(test_case, agent_response)
                        
                        passed = eval_result["passed"]
                        total_score = eval_result["scores"]["total"]
                        
                        logger.debug("Evaluation complete: passed=%s, score=%.2f", passed, total_score)
                        
                    except Exception as e:
                        logger.exception("Evaluation failed for case %s: %s", case_id, e)
                        # Use default failure result
                        eval_result = {
                            "passed": False,
                            "scores": {"total": 0.0, "source_correct": 0.0, "content_match": 0.0},
                            "feedback": [f"Error: Evaluation failed - {str(e)}"]
                        }
                        logger.warning("Using default failure result for case %s", case_id)
                    
                    # Track by category
                    if category not in category_scores:
                        category_scores[category] = []
                        logger.debug("Created new category tracker: %s", category)
                    
                    category_scores[category].append(eval_result["scores"]["total"])
                    
                    # Append result
                    results.append({
                        "case_id": case_id,
                        "category": category,
                        "passed": eval_result["passed"],
                        "score": eval_result["scores"]["total"],
                        "source_correct": eval_result["scores"]["source_correct"],
                        "content_match": eval_result["scores"]["content_match"],
                        "feedback": " | ".join(eval_result["feedback"])
                    })
                    
                    # Log status
                    status = "PASS" if eval_result["passed"] else "FAIL"
                    logger.info("Case %s: %s | Score: %.2f", case_id, status, eval_result["scores"]["total"])
                
                logger.info("Completed evaluation of all %d test cases", len(test_cases))
                
                # Create results DataFrame
                try:
                    results_df = pd.DataFrame(results)
                    logger.debug("Created results DataFrame with %d rows", len(results_df))
                except Exception as e:
                    logger.exception("Failed to create results DataFrame: %s", e)
                    raise
                
                # Log aggregate metrics
                try:
                    pass_rate = results_df['passed'].mean()
                    avg_score = results_df['score'].mean()
                    total_passed = int(results_df['passed'].sum())
                    total_failed = int((~results_df['passed']).sum())
                    
                    mlflow.log_metric("pass_rate", pass_rate)
                    mlflow.log_metric("avg_score", avg_score)
                    mlflow.log_metric("total_passed", total_passed)
                    mlflow.log_metric("total_failed", total_failed)
                    
                    logger.info("Aggregate metrics: pass_rate=%.2f, avg_score=%.2f", pass_rate, avg_score)
                    logger.info("Total passed: %d, Total failed: %d", total_passed, total_failed)
                    
                except Exception as e:
                    logger.exception("Failed to log aggregate metrics: %s", e)
                
                # Log category metrics
                try:
                    logger.debug("Logging category-specific metrics")
                    for category, scores in category_scores.items():
                        category_pass_rate = sum(s >= 0.7 for s in scores) / len(scores)
                        mlflow.log_metric(f"pass_rate_{category}", category_pass_rate)
                        logger.debug("Category %s: pass_rate=%.2f", category, category_pass_rate)
                    
                    logger.info("Logged metrics for %d categories", len(category_scores))
                    
                except Exception as e:
                    logger.exception("Failed to log category metrics: %s", e)
                
                # Log artifacts
                try:
                    logger.debug("Saving evaluation results CSV")
                    results_df.to_csv("evaluation_results.csv", index=False)
                    mlflow.log_artifact("evaluation_results.csv")
                    logger.info("Logged evaluation_results.csv artifact")
                    
                except Exception as e:
                    logger.exception("Failed to save/log results artifact: %s", e)
                
                # Log failed cases artifact
                try:
                    failed_df = results_df[~results_df['passed']]
                    
                    if len(failed_df) > 0:
                        logger.debug("Saving failed cases CSV (%d failures)", len(failed_df))
                        failed_df.to_csv("failed_cases.csv", index=False)
                        mlflow.log_artifact("failed_cases.csv")
                        logger.info("Logged failed_cases.csv artifact: %d failed cases", len(failed_df))
                    else:
                        logger.info("No failed cases to log")
                        
                except Exception as e:
                    logger.exception("Failed to save/log failed cases artifact: %s", e)
                
                logger.info("Evaluation run completed successfully")
                return results_df
                
        except Exception as e:
            logger.exception("MLflow run failed: %s", e)
            raise
    
    def print_summary(self, results_df: pd.DataFrame) -> None:
        """
        Display evaluation summary with pass/fail statistics.
        
        Args:
            results_df: DataFrame containing evaluation results
            
        Raises:
            Exception: If summary calculation fails (logged but not re-raised)
        """
        logger.info("Generating evaluation summary")
        
        try:
            total_cases = len(results_df)
            passed = int(results_df['passed'].sum())
            failed = int((~results_df['passed']).sum())
            pass_rate = results_df['passed'].mean() * 100
            avg_score = results_df['score'].mean()
            
            logger.info("=" * 80)
            logger.info("EVALUATION RESULTS SUMMARY")
            logger.info("=" * 80)
            logger.info("Total Cases: %d", total_cases)
            logger.info("Passed: %d", passed)
            logger.info("Failed: %d", failed)
            logger.info("Pass Rate: %.1f%%", pass_rate)
            logger.info("Avg Score: %.2f", avg_score)
            logger.info("=" * 80)
            
        except Exception as e:
            logger.exception("Failed to generate summary: %s", e)
            raise
