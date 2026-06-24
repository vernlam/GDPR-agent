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
    scale_to_zero: bool = True,
    timeout: int = 3600
):
    """
    Deploy or update a model serving endpoint.
    Handles active updates gracefully by waiting for locks to clear.
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
        print(f"📝 Endpoint '{endpoint_name}' exists. Verifying deployment state...", file=sys.stderr)
        
        # Check for active configuration update lock
        if existing.state and existing.state.config_update:
            config_status = str(existing.state.config_update).upper()
            if "IN_PROGRESS" in config_status or "UPDATE" in config_status:
                print(f"⏳ Backend has an active config update in progress. Waiting for lock to clear...", file=sys.stderr)
                wait_for_endpoint(w, endpoint_name, timeout=timeout)
        
        # Update existing endpoint configuration smoothly
        try:
            w.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=endpoint_config.served_entities,
                traffic_config=endpoint_config.traffic_config
            )
        except Exception as update_err:
            # Fallback catch if the status block hadn't propagated the lock yet
            if "RESOURCE_CONFLICT" in str(update_err).upper() or "BEING_UPDATED" in str(update_err).replace(" ", "_").upper():
                print(f"⏳ Hit conflict warning. Waiting out the active backend deployment track...", file=sys.stderr)
                wait_for_endpoint(w, endpoint_name, timeout=timeout)
                # Re-try modification step once lock releases
                w.serving_endpoints.update_config(
                    name=endpoint_name,
                    served_entities=endpoint_config.served_entities,
                    traffic_config=endpoint_config.traffic_config
                )
            else:
                raise update_err
        
        print(f"⏳ Updating endpoint (5-10 minutes)...", file=sys.stderr)
        
    except Exception as e:
        if "RESOURCE_DOES_NOT_EXIST" in str(e) or "does not exist" in str(e).lower():
            print(f"✨ Creating new endpoint '{endpoint_name}'...", file=sys.stderr)
            
            # Create fresh model serving endpoint
            w.serving_endpoints.create(
                name=endpoint_name,
                config=endpoint_config
            )
            
            print(f"⏳ Creating endpoint (10-15 minutes for first deployment, may take up to 45 minutes with dependencies)...", file=sys.stderr)
        else:
            print(f"❌ Error: {e}", file=sys.stderr)
            raise e
    
    # Wait for endpoint to reach operational status
    wait_for_endpoint(w, endpoint_name, timeout=timeout)
    
    print(f"\n✅ Deployment complete!", file=sys.stderr)
    print(f"🔗 Endpoint URL: https://{w.config.host}/serving-endpoints/{endpoint_name}", file=sys.stderr)
    
    return endpoint_name


def wait_for_endpoint(client: WorkspaceClient, endpoint_name: str, timeout: int = 3600):
    """
    Wait for endpoint to reach READY state.
    Default timeout is 60 minutes for initial deployments with dependencies.
    """
    start_time = time.time()
    last_state = None
    
    while time.time() - start_time < timeout:
        try:
            endpoint = client.serving_endpoints.get(endpoint_name)
            
            # Extract state from endpoint object
            state = "UNKNOWN"
            if endpoint.state:
                # Check config_update status first
                if hasattr(endpoint.state, 'config_update') and endpoint.state.config_update:
                    config_status = str(endpoint.state.config_update).upper()
                    if "NOT_UPDATING" in config_status:
                        state = "READY"
                    elif "IN_PROGRESS" in config_status or "UPDATE" in config_status:
                        state = "IN_PROGRESS"
                    elif "FAILED" in config_status:
                        state = "FAILED"
                    else:
                        state = config_status
                # Check ready status
                elif hasattr(endpoint.state, 'ready') and endpoint.state.ready:
                    ready_status = str(endpoint.state.ready).upper()
                    if "READY" in ready_status:
                        state = "READY"
                    elif "NOT_READY" in ready_status:
                        state = "NOT_READY"
                    else:
                        state = ready_status
                else:
                    # Default to READY if no update in progress and endpoint exists
                    state = "READY"
            else:
                state = "READY"
            
            # Only print if state changed
            if state != last_state:
                elapsed = int(time.time() - start_time)
                minutes = elapsed // 60
                seconds = elapsed % 60
                print(f"   Status: {state} ({minutes}m {seconds}s elapsed)", file=sys.stderr)
                last_state = state
            
            # Check if ready (case-insensitive)
            if state == "READY" or "NOT_UPDATING" in state:
                print(f"✅ Endpoint is ready!", file=sys.stderr)
                return endpoint
            elif "FAIL" in state:
                raise Exception(f"❌ Endpoint deployment failed! Current State: {state}")
            
        except Exception as e:
            if "does not exist" in str(e).lower():
                if time.time() - start_time < 120:
                    print(f"   Waiting for endpoint to be created...", file=sys.stderr)
                else:
                    raise e
            else:
                raise e
        
        time.sleep(30)
    
    minutes = timeout // 60
    raise TimeoutError(f"❌ Endpoint did not become ready within {timeout}s ({minutes} minutes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy model to serving endpoint")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--workload-size", default="Small", choices=["Small", "Medium", "Large"])
    parser.add_argument("--scale-to-zero", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True)
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout in seconds (default: 3600 = 60 minutes)")
    
    args = parser.parse_args()
    
    endpoint_name = deploy_endpoint(
        endpoint_name=args.endpoint_name,
        model_name=args.model_name,
        model_version=args.model_version,
        workload_size=args.workload_size,
        scale_to_zero=args.scale_to_zero,
        timeout=args.timeout
    )
    
    print(endpoint_name)  # Clean standard stdout print for GitHub variables