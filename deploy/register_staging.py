"""
Register model to MLflow Model Registry (Staging).
Each successful eval creates a new version.
"""

import mlflow
import argparse
from datetime import datetime
from gdpr_agent import GDPRAgent

def register_staging_model(commit_sha: str, pass_rate: float):
    """
    Register a new model version to the staging registry.
    
    Args:
        commit_sha: Git commit SHA
        pass_rate: Evaluation pass rate
    
    Returns:
        Model version number
    """
    
    # Use staging experiment
    mlflow.set_experiment("/Shared/gdpr-agent-staging")
    
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
        
        # Register model
        model_info = mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=GDPRAgent(),
            registered_model_name="gdpr_agent_staging",  # Separate staging registry
            pip_requirements=[
                "openai>=1.12.0",
                "langgraph>=0.0.20",
                "databricks-vectorsearch",
                "mlflow"
            ]
        )
        
        # Get the version that was just registered
        client = mlflow.tracking.MlflowClient()
        model_version = client.search_model_versions(
            filter_string=f"name='gdpr_agent_staging' and run_id='{run.info.run_id}'"
        )[0].version
        
        # Add tags to the model version
        client.set_model_version_tag(
            name="gdpr_agent_staging",
            version=model_version,
            key="commit_sha",
            value=commit_sha
        )
        
        client.set_model_version_tag(
            name="gdpr_agent_staging",
            version=model_version,
            key="eval_pass_rate",
            value=str(pass_rate)
        )
        
        client.set_model_version_tag(
            name="gdpr_agent_staging",
            version=model_version,
            key="deployment_status",
            value="staging"
        )
        
        print(f"✅ Registered staging model version: {model_version}")
        print(f"📊 Pass rate: {pass_rate}")
        print(f"🔗 MLflow Run: {run.info.run_id}")
        
        return model_version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit-sha", required=True)
    parser.add_argument("--pass-rate", type=float, required=True)
    
    args = parser.parse_args()
    
    version = register_staging_model(args.commit_sha, args.pass_rate)
    print(version)  # Output for GitHub Actions to capture