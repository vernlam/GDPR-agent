"""
Copy model from staging to production registry.
"""

import mlflow
import argparse

def promote_to_production(staging_version: str):
    """
    Copy a staging model version to the production registry.
    
    Args:
        staging_version: Version number from gdpr_agent_staging
    
    Returns:
        New production version number
    """
    
    client = mlflow.tracking.MlflowClient()
    
    # Get staging model details
    staging_mv = client.get_model_version(
        name="gdpr_agent_staging",
        version=staging_version
    )
    
    print(f"📦 Promoting staging version {staging_version} to production")
    print(f"   Run ID: {staging_mv.run_id}")
    
    # Register to production model
    model_uri = f"models:/gdpr_agent_staging/{staging_version}"
    
    result = mlflow.register_model(
        model_uri=model_uri,
        name="gdpr_agent_prod"  # Production registry
    )
    
    prod_version = result.version
    
    # Copy tags
    for tag_key, tag_value in staging_mv.tags.items():
        client.set_model_version_tag(
            name="gdpr_agent_prod",
            version=prod_version,
            key=tag_key,
            value=tag_value
        )
    
    # Mark as production
    client.set_model_version_tag(
        name="gdpr_agent_prod",
        version=prod_version,
        key="deployment_status",
        value="production"
    )
    
    print(f"✅ Promoted to production version: {prod_version}")
    
    return prod_version


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--staging-version", required=True)
    
    args = parser.parse_args()
    
    version = promote_to_production(args.staging_version)
    print(version)