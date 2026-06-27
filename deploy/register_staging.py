"""
Model registration automation for staging environment.
Creates versioned model entries with wrapper, signature, dependencies, and evaluation metrics.
"""

import argparse
import contextlib
import io
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Union
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.models.resources import DatabricksVectorSearchIndex

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class GDPRAgentWrapper(mlflow.pyfunc.PythonModel):
    """
    MLflow PyFunc wrapper for GDPR Agent inference.
    
    Handles model lifecycle (load_context) and inference (predict) with multiple
    input formats, error handling, and request-level metadata logging.
    """
    
    def __init__(self) -> None:
        """
        Initialize wrapper without agent instance.
        
        Agent creation is deferred to load_context() for proper model serving lifecycle.
        """
        logger.debug("Initializing GDPRAgentWrapper (agent deferred to load_context)")
        self.agent = None
    
    def load_context(self, context: Any) -> None:
        """
        Load agent instance when model is deployed or served.
        
        Called automatically by MLflow serving infrastructure before first inference.
        
        Args:
            context: MLflow context object (contains artifacts, dependencies, etc.)
            
        Raises:
            ImportError: If gdpr_agent.agent module cannot be imported
            Exception: If GDPRAgent initialization fails (re-raised after logging)
        """
        logger.info("Loading GDPR agent in model context")
        try:
            from gdpr_agent.agent import GDPRAgent
            logger.debug("Successfully imported GDPRAgent class")
            self.agent = GDPRAgent()
            logger.info("GDPR agent loaded successfully")
        except ImportError as e:
            logger.exception("Failed to import GDPRAgent module: %s", e)
            raise
        except Exception as e:
            logger.exception("Failed to initialize GDPRAgent: %s", e)
            raise
    
    def predict(self, context: Any, model_input: Union[pd.DataFrame, Dict, List, Any]) -> List[Dict[str, Any]]:
        """
        Handle inference requests with multiple input formats and metadata tracking.
        
        Processes questions individually with per-request error isolation, latency tracking,
        and structured metadata for external logging/monitoring.
        
        Args:
            context: MLflow context (ignored, kept for interface compatibility)
            model_input: Input data in one of several formats:
                - pd.DataFrame with 'question' column
                - dict with 'question' key
                - list of dicts/strings
                - single string (coerced to list)
                
        Returns:
            List of result dictionaries, one per question, each containing:
                - answer (str): Agent response or error message
                - context (list): Retrieved context documents
                - metadata (dict): request_id, timestamp, latency_ms, status, question, error_message
                
        Raises:
            Exception: If agent loading fails (logged but not re-raised; returns error result instead)
        """
        logger.info("Received predict request")
        
        # Ensure agent is loaded (defensive check for direct invocation)
        if self.agent is None:
            logger.warning("Agent not loaded via load_context, loading now")
            try:
                from gdpr_agent.agent import GDPRAgent
                self.agent = GDPRAgent()
                logger.debug("Agent loaded on-demand in predict method")
            except Exception as e:
                logger.exception("Failed to load agent in predict method: %s", e)
                raise
        
        # Import dependencies locally for model serving isolation
        try:
            import uuid
            logger.debug("Imported uuid for request ID generation")
        except ImportError as e:
            logger.exception("Failed to import uuid: %s", e)
            raise
        
        # Normalize input to list of question strings
        try:
            if isinstance(model_input, pd.DataFrame):
                questions = model_input['question'].tolist()
                logger.debug("Parsed DataFrame input: %d questions", len(questions))
            elif isinstance(model_input, dict):
                questions = [model_input.get('question', '')]
                logger.debug("Parsed dict input: 1 question")
            elif isinstance(model_input, list):
                questions = [q.get('question', '') if isinstance(q, dict) else str(q) for q in model_input]
                logger.debug("Parsed list input: %d questions", len(questions))
            else:
                questions = [str(model_input)]
                logger.debug("Coerced input to string: 1 question")
        except Exception as e:
            logger.exception("Failed to parse model_input: %s", e)
            raise
        
        logger.info("Processing %d questions", len(questions))
        
        # Process each question with error isolation
        results = []
        
        for idx, question in enumerate(questions, 1):
            request_id = str(uuid.uuid4())
            start_time = datetime.now()
            logger.debug("Processing question %d/%d, request_id=%s", idx, len(questions), request_id)
            
            try:
                # Invoke agent
                logger.debug("Invoking agent for request_id=%s", request_id)
                response = self.agent.invoke({"question": question})
                
                end_time = datetime.now()
                latency_ms = (end_time - start_time).total_seconds() * 1000
                logger.info("Agent invocation succeeded: request_id=%s, latency=%.2fms", request_id, latency_ms)
                
                result = {
                    'answer': response.get('answer', ''),
                    'context': response.get('context', []),
                    'metadata': {
                        'request_id': request_id,
                        'timestamp': start_time.isoformat(),
                        'latency_ms': latency_ms,
                        'status': 'success',
                        'question': question
                    }
                }
                results.append(result)
                logger.debug("Result appended for request_id=%s", request_id)
                
            except Exception as e:
                end_time = datetime.now()
                latency_ms = (end_time - start_time).total_seconds() * 1000
                logger.exception("Agent invocation failed: request_id=%s, latency=%.2fms, error=%s", 
                               request_id, latency_ms, e)
                
                results.append({
                    'answer': f"Error: {str(e)}",
                    'context': [],
                    'metadata': {
                        'request_id': request_id,
                        'timestamp': start_time.isoformat(),
                        'latency_ms': latency_ms,
                        'status': 'error',
                        'question': question,
                        'error_message': str(e)
                    }
                })
                logger.debug("Error result appended for request_id=%s", request_id)
        
        logger.info("Predict completed: %d results", len(results))
        return results


def register_staging_model(commit_sha: str, pass_rate: float) -> str:
    """
    Register a new staging model version with evaluation metrics and dependencies.
    
    Creates MLflow run, logs parameters/metrics, defines signature, packages wrapper
    with code paths and pip dependencies, registers to Unity Catalog, and sets tags.
    
    Args:
        commit_sha: Git commit SHA for version tracking
        pass_rate: Evaluation pass rate (0.0 to 1.0)
        
    Returns:
        Registered model version number as string
        
    Raises:
        Exception: If experiment setting fails (re-raised after logging)
        Exception: If MLflow run creation fails (re-raised after logging)
        Exception: If model registration (log_model) fails (re-raised after logging)
        Exception: If model version lookup fails (re-raised after logging)
        Exception: If tag setting fails (re-raised after logging)
    """
    logger.info("Starting staging model registration")
    logger.debug("Commit SHA: %s, Pass rate: %.2f", commit_sha, pass_rate)
    
    # Suppress MLflow verbose output to stdout (keep stderr for CI/CD visibility)
    try:
        logging.getLogger("mlflow").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore")
        logger.debug("MLflow verbose logging suppressed")
    except Exception as e:
        logger.warning("Failed to suppress MLflow logging: %s", e)
    
    # Add GDPR-agent to path if running in Databricks workspace
    try:
        cwd = os.getcwd()
        if '/Workspace/' in cwd:
            sys.path.insert(0, '/Workspace/Repos/vernonc.lam@gmail.com/GDPR-agent')
            logger.debug("Added GDPR-agent to sys.path (Databricks environment)")
        else:
            logger.debug("Not in Databricks workspace, using default sys.path")
    except Exception as e:
        logger.warning("Failed to adjust sys.path: %s", e)
    
    # Set MLflow experiment
    try:
        experiment_name = "/Shared/gdpr-agent-staging"
        mlflow.set_experiment(experiment_name)
        logger.info("MLflow experiment set: %s", experiment_name)
    except Exception as e:
        logger.exception("Failed to set MLflow experiment: %s", e)
        raise
    
    # Redirect stdout to suppress MLflow's verbose output (logs go to stderr)
    with contextlib.redirect_stdout(io.StringIO()):
        run_name = f"staging_{commit_sha[:7]}"
        logger.debug("Starting MLflow run: %s", run_name)
        
        try:
            with mlflow.start_run(run_name=run_name) as run:
                logger.info("MLflow run started: run_id=%s", run.info.run_id)
                
                # Write to stderr for CI/CD visibility
                print(f"Registering model for commit {commit_sha[:7]}", file=sys.stderr)
                print(f"   Run ID: {run.info.run_id}", file=sys.stderr)
                
                # Log parameters
                try:
                    params = {
                        "commit_sha": commit_sha,
                        "llm_model": "gpt-4o-mini",
                        "deployment_target": "staging",
                        "timestamp": datetime.now().isoformat()
                    }
                    mlflow.log_params(params)
                    logger.info("Logged %d parameters", len(params))
                    logger.debug("Parameters: %s", params)
                except Exception as e:
                    logger.exception("Failed to log parameters: %s", e)
                    raise
                
                # Log metrics
                try:
                    metrics = {"eval_pass_rate": pass_rate}
                    mlflow.log_metrics(metrics)
                    logger.info("Logged metrics: eval_pass_rate=%.2f", pass_rate)
                except Exception as e:
                    logger.exception("Failed to log metrics: %s", e)
                    raise
                
                # Define input/output signature
                try:
                    input_example = pd.DataFrame({
                        "question": ["What are the GDPR requirements for data deletion?"]
                    })
                    
                    output_example = [{
                        "answer": "Under GDPR Article 17, individuals have the right to erasure...",
                        "context": ["Article 17: Right to erasure"],
                        "sources": ["legislation"]
                    }]
                    
                    signature = mlflow.models.infer_signature(
                        model_input=input_example,
                        model_output=output_example
                    )
                    logger.info("Model signature inferred successfully")
                    logger.debug("Input schema: %s", signature.inputs)
                    logger.debug("Output schema: %s", signature.outputs)
                except Exception as e:
                    logger.exception("Failed to create model signature: %s", e)
                    raise
                
                # Determine code path based on environment
                try:
                    dbx_code_path = '/Workspace/Repos/vernonc.lam@gmail.com/GDPR-agent/gdpr_agent'
                    if os.path.exists(dbx_code_path):
                        code_paths = [dbx_code_path]
                        logger.debug("Using Databricks code path: %s", dbx_code_path)
                    else:
                        code_paths = ["./gdpr_agent"]
                        logger.debug("Using GitHub Actions code path: ./gdpr_agent")
                except Exception as e:
                    logger.exception("Failed to determine code paths: %s", e)
                    raise
                
                # Define vector search index resources
                try:
                    resources = [
                        DatabricksVectorSearchIndex(
                            index_name="main.default.gdpr_law_vector_index"
                        ),
                        DatabricksVectorSearchIndex(
                            index_name="main.default.gdpr_fines_vector_index"
                        ),
                        DatabricksVectorSearchIndex(
                            index_name="main.default.privacy_policy_vector_index"
                        )
                    ]
                    logger.info("Defined %d vector search index resources", len(resources))
                except Exception as e:
                    logger.exception("Failed to define vector search resources: %s", e)
                    raise
                
                # Register model
                try:
                    logger.debug("Logging model to MLflow with wrapper, signature, code paths, dependencies")
                    model_info = mlflow.pyfunc.log_model(
                        artifact_path="model",
                        python_model=GDPRAgentWrapper(),
                        registered_model_name="main.default.gdpr_agent_staging",
                        signature=signature,
                        input_example=input_example,
                        code_paths=code_paths,
                        pip_requirements=[
                            "openai>=1.12.0",
                            "langgraph>=0.2.0",
                            "databricks-vectorsearch",
                            "mlflow",
                            "pandas"
                        ],
                        resources=resources
                    )
                    logger.info("Model registered successfully to main.default.gdpr_agent_staging")
                except Exception as e:
                    logger.exception("Failed to register model: %s", e)
                    raise
                
                # Get the version that was just registered
                try:
                    client = MlflowClient()
                    logger.debug("MLflow client initialized for version lookup")
                    
                    # Search by name only (Unity Catalog limitation)
                    filter_string = "name='main.default.gdpr_agent_staging'"
                    logger.debug("Searching model versions: filter=%s", filter_string)
                    all_versions = client.search_model_versions(filter_string=filter_string)
                    logger.debug("Found %d total versions", len(all_versions))
                    
                    # Find the version matching our run_id
                    model_version = None
                    for v in all_versions:
                        if v.run_id == run.info.run_id:
                            model_version = v.version
                            logger.debug("Matched version %s to run_id %s", model_version, run.info.run_id)
                            break
                    
                    if model_version is None:
                        error_msg = f"Could not find registered model version for run_id: {run.info.run_id}"
                        logger.error(error_msg)
                        raise Exception(error_msg)
                    
                    logger.info("Model version identified: %s", model_version)
                except Exception as e:
                    logger.exception("Failed to lookup model version: %s", e)
                    raise
                
                # Add tags to the model version
                try:
                    tags = [
                        ("commit_sha", commit_sha),
                        ("eval_pass_rate", str(pass_rate)),
                        ("deployment_status", "staging")
                    ]
                    logger.debug("Setting %d tags on model version %s", len(tags), model_version)
                    
                    for tag_key, tag_value in tags:
                        client.set_model_version_tag(
                            name="main.default.gdpr_agent_staging",
                            version=model_version,
                            key=tag_key,
                            value=tag_value
                        )
                        logger.debug("Set tag: %s = %s", tag_key, tag_value)
                    
                    logger.info("All %d tags set successfully", len(tags))
                except Exception as e:
                    logger.exception("Failed to set model version tags: %s", e)
                    raise
                
                # Write to stderr for CI/CD visibility
                print(f"Registered staging model version: {model_version}", file=sys.stderr)
                print(f"   Pass rate: {pass_rate}", file=sys.stderr)
                print(f"   MLflow Run: {run.info.run_id}", file=sys.stderr)
                
                logger.info("Staging model registration completed successfully")
                logger.info("Model version: %s", model_version)
                
                return model_version
                
        except Exception as e:
            logger.exception("MLflow run failed: %s", e)
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Register GDPR Agent model to staging registry")
    parser.add_argument("--commit-sha", required=True, 
                        help="Git commit SHA for version tracking")
    parser.add_argument("--pass-rate", type=float, required=True, 
                        help="Evaluation pass rate (0.0 to 1.0)")
    
    args = parser.parse_args()
    
    logger.info("Starting registration script")
    logger.debug("Arguments: commit_sha=%s, pass_rate=%.2f", args.commit_sha, args.pass_rate)
    
    try:
        version = register_staging_model(args.commit_sha, args.pass_rate)
        
        # Clean stdout print for GitHub Actions variables
        print(version)
        logger.info("Registration script completed successfully")
        
    except Exception as e:
        logger.exception("Registration script failed: %s", e)
        sys.exit(1)
