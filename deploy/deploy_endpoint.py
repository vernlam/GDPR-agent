"""
Deploy model to Databricks Model Serving endpoint.
Supports both Staging and Production workloads.
"""

import argparse
import time
import sys
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route
)


def deploy_endpoint(
    endpoint_name: str,
    model_name: str,
    model_version: str,
    workload_size: str = "Small",
    scale_to_zero: bool = True
):
    """
    Deploy or update a model serving endpoint.
    """
    w = WorkspaceClient()
    
    print(f"🚀 Deploying {model_name} v{model_version} to endpoint: {endpoint_name}", file=sys.stderr)
    print(f"⚙️ Config -> Workload: {workload_size} | Scale-to-Zero: {scale_to_zero}", file=sys.stderr)
    
    # Extract model name for served entity
    short_model_name = model_name.split('.')[-1]
    
    # Configure the served entity
    served_entities = [
        ServedEntityInput(
            name=f"{short_model_name}_v{model_version}",
            entity_name=model_name,
            entity_version=model_version,
            workload_size=workload_size,
            scale_to_zero_enabled=scale_to_zero,
            environment_vars={
                "OPENAI_API_KEY": "{{secrets/openai/GDPR_agent}}"
            }
        )
    ]
    
    # Build traffic config
    traffic_config = TrafficConfig(
        routes=[
            Route(
                served_model_name=f"{short_model_name}_v{model_version}",
                traffic_percentage=100
            )
        ]
    )
    
    # Build endpoint configuration 
    endpoint_config = EndpointCoreConfigInput(
        name=endpoint_name,
        served_entities=served_entities,
        traffic_config=traffic_config
    )
    
    try:
        # Try to get existing endpoint
        existing = w.serving_endpoints.get(endpoint_name)
        print(f"📝 Endpoint '{endpoint_name}' exists. Updating configuration...", file=sys.stderr)
        
        # Update existing endpoint configuration smoothly
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=endpoint_config.served_entities,
            traffic_config=endpoint_config.traffic_config
        )
        
        print(f"⏳ Updating endpoint (5-10 minutes)...", file=sys.stderr)
        
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "does not exist" in str(e).lower():
            print(f"✨ Creating new endpoint '{endpoint_name}'...", file=sys.stderr)
            
            # Create fresh model serving endpoint
            w.serving_endpoints.create(
                name=endpoint_name,
                config=endpoint_config
            )
            
            print(f"⏳ Creating endpoint (10-15 minutes)...", file=sys.stderr)
        else:
            print(f"❌ Error: {e}", file=sys.stderr)
            raise e
    
    # Wait for endpoint to reach operational status
    wait_for_endpoint(w, endpoint