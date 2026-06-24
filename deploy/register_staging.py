"""
Register model to MLflow Model Registry (Staging).
Each successful eval creates a new version.
"""

import mlflow
import argparse
import sys
import os
from datetime import datetime
import pandas as pd


class GDPRAgentWrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for GDPRAgent"""
    
    def __init__(self):
        """Initialize without creating agent yet (happens in load_context)"""
        self.agent = None
    
    def load_context(self, context):
        """Called when model is loaded for serving"""
        # Import here to avoid issues during serialization
        from gdpr_agent.agent import GDPRAgent
        self.agent = GDPRAgent()
    
    def predict(self, context, model_input):  # ← INDENTED (inside class)
        """
        Handle inference requests with logging.
        """
        import pandas as pd
        from datetime import datetime
        import uuid
        
        # Ensure agent is loaded
        if self.agent is None:
            from gdpr_agent.agent import GDPRAgent
            self.agent = GDPRAgent()
        
        # Handle different input types
        if isinstance(model_input, pd.DataFrame):
            questions = model_input['question'].tolist()
        elif isinstance(model_input, dict):
            questions = [model_input.get('question', '')]
        elif isinstance(model_input, list):
            questions = [q.get('question', '') if isinstance(q, dict) else str(q) for q in model_input]
        else:
            questions = [str(model_input)]
        
        # Process each question
        results = []
        logs = []
        
        for question in questions:
            request_id = str(uuid.uuid4())
            start_time = datetime.now()
            
            try:
                # Call your agent's invoke method
                response = self.agent.invoke({"question": question})
                
                end_time = datetime.now()
                latency_ms = (end_time - start_time).total_seconds() * 1000
                
                result = {
                    'answer': response.get('answer', ''),
                    'context': response.get('context', []),
                }
                results.append(result)
                
                # Log to table
                logs.append({
                    'timestamp': start_time,
                    'request_id': request_id,
                    'question': question,
                    'answer': result['answer'],
                    'context': str(result['context']),
                    'latency_ms': latency_ms,
                    'status': 'success',
                    'error_message': None
                })
                
            except Exception as e:
                end_time = datetime.now()
                latency_ms = (end_time - start_time).total_seconds() * 1000
                
                results.append({
                    'answer': f"Error: {str(e)}",
                    'context': [],
                })
                
                logs.append({
                    'timestamp': start_time,
                    'request_id': request_id,
                    'question': question,
                    'answer': None,
                    'context': None,
                    'latency_ms': latency_ms,
                    'status': 'error',
                    'error_message': str(e)
                })
        
        # Write logs to Delta table (async, don't block response)
        try:
            self._write_logs_async(logs)
        except:
            pass  # Don't fail the request if logging fails
        
        return results
    
    def _write_logs_async(self, logs):  # ← INDENTED (inside class)
        """Write logs to Delta table asynchronously"""
        try:
            from pyspark.sql import SparkSession
            import pandas as pd
            
            spark = SparkSession.builder.getOrCreate()
            logs_df = pd.DataFrame(logs)
            logs_df['date'] = pd.to_datetime(logs_df['timestamp']).dt.date
            
            spark_df = spark.createDataFrame(logs_df)
            spark_df.write.mode("append").saveAsTable("main.default.gdpr_agent_inference_logs")
        except Exception as e:
            print(f"Warning: Failed to write logs: {e}")


def register_staging_model(commit_sha: str, pass_rate: float):
    """
    Register a new model version to the staging registry.
    
    Args:
        commit_sha: Git commit SHA
        pass_rate: Evaluation pass rate
    
    Returns:
        Model version number
    """
    import mlflow
    from mlflow.models.resources import DatabricksVectorSearchIndex
    import logging
    import warnings
    import contextlib
    import io

    # Suppress MLflow verbose output to stdout
    logging.getLogger("mlflow").setLevel(logging.ERROR)
    warnings.filterwarnings("ignore")
    
    # Ensure we're importing from the correct location
    if '/Workspace/' in os.getcwd():
        sys.path.insert(0, '/Workspace/Repos/vernonc.lam@gmail.com/GDPR-agent')
    
    # Use staging experiment
    mlflow.set_experiment("/Shared/gdpr-agent-staging")
    
    with contextlib.redirect_stdout(io.StringIO()):
        with mlflow.start_run(run_name=f"staging_{commit_sha[:7]}") as run:
            
            # Log evaluation metrics
            mlflow.log_params({
                "commit_sha": commit_sha,
                "llm_model": "gpt-4o-mini",
                "deployment_target": "staging",
                "timestamp": datetime.now().isoformat()
            })
            
            mlflow.log_metrics({
                "eval_pass_rate": pass_rate,
            })
            
            # Define input/output signature (required for Unity Catalog)
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
            
            # Determine code path based on environment
            if os.path.exists('/Workspace/Repos/vernonc.lam@gmail.com/GDPR-agent/gdpr_agent'):
                # Running in Databricks
                code_paths = ["/Workspace/Repos/vernonc.lam@gmail.com/GDPR-agent/gdpr_agent"]
            else:
                # Running in GitHub Actions
                code_paths = ["./gdpr_agent"]
            
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

            # Register model using the wrapper WITH code paths
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
            
            # Get the version that was just registered
            client = mlflow.tracking.MlflowClient()
            
            # Search by name only (Unity Catalog limitation)
            all_versions = client.search_model_versions(
                filter_string=f"name='main.default.gdpr_agent_staging'"
            )
            
            # Find the version matching our run_id
            model_version = None
            for v in all_versions:
                if v.run_id == run.info.run_id:
                    model_version = v.version
                    break
            
            if model_version is None:
                raise Exception(f"Could not find registered model version for run_id: {run.info.run_id}")
            
            # Add tags to the model version
            client.set_model_version_tag(
                name="main.default.gdpr_agent_staging",
                version=model_version,
                key="commit_sha",
                value=commit_sha
            )
            
            client.set_model_version_tag(
                name="main.default.gdpr_agent_staging",
                version=model_version,
                key="eval_pass_rate",
                value=str(pass_rate)
            )
            
            client.set_model_version_tag(
                name="main.default.gdpr_agent_staging",
                version=model_version,
                key="deployment_status",
                value="staging"
            )
            
    print(f"✅ Registered staging model version: {model_version}", file=sys.stderr)
    print(f"📊 Pass rate: {pass_rate}", file=sys.stderr)
    print(f"🔗 MLflow Run: {run.info.run_id}", file=sys.stderr)
    
    return model_version

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--pass-rate", type=float, required=True)
    
    args = parser.parse_args()
    
    version = register_staging_model(args.commit_sha, args.pass_rate)
    print(version)  # Output for GitHub Actions to capture