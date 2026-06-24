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
        
        # 🟢 FIXED: Clean, crash-proof check for an active configuration update lock
        if existing.state and existing.state.config_update:
            print(f"⏳ Backend has an active config update in progress. Waiting for lock to clear...", file=sys.stderr)
            wait_for_endpoint(w, endpoint_name)
        
        # Update existing endpoint configuration smoothly
        try:
            w.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=endpoint_config.served_entities,
                traffic_config=endpoint_config.traffic_config
            )
        except Exception as update_err:
            # 🟢 EXCEPTION BACKOFF: Fallback catch if the status block hadn't propagated the lock yet
            if "RESOURCE_CONFLICT" in str(update_err).upper() or "BEING_UPDATED" in str(update_err).replace(" ", "_").upper():
                print(f"⏳ Hit conflict warning. Waiting out the active backend deployment track...", file=sys.stderr)
                wait_for_endpoint(w, endpoint_name)
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
            
            print(f"⏳ Creating endpoint (10-15 minutes)...", file=sys.stderr)
        else:
            print(f"❌ Error: {e}", file=sys.stderr)
            raise e
    
    # Wait for endpoint to reach operational status
    wait_for_endpoint(w, endpoint_name)
    
    print(f"\n✅ Deployment complete!", file=sys.stderr)
    print(f"🔗 Endpoint URL: https://{w.config.host}/serving-endpoints/{endpoint_name}", file=sys.stderr)
    
    return endpoint_name


def wait_for_endpoint(client: WorkspaceClient, endpoint_name: str, timeout: int = 1800):
    """
    Wait for endpoint to reach READY state.
    """
    start_time = time.time()
    last_state = None
    
    while time.time() - start_time < timeout:
        try:
            endpoint = client.serving_endpoints.get(endpoint_name)
            
            # Safe object verification extraction
            if endpoint.state:
                if endpoint.state.config_update:
                    state = "IN_PROGRESS"
                elif endpoint.state.ready:
                    state = str(endpoint.state.ready).upper()
                else:
                    state = "READY"
            else:
                state = "READY"
            
            # Only print if state changed
            if state != last_state:
                elapsed = int(time.time() - start_time)
                print(f"   Status: {state} ({elapsed}s elapsed)", file=sys.stderr)
                last_state = state
            
            # Case-insensitive substring verification avoids Enum errors
            if any(x in state.upper() for x in ["NOT_UPDATING", "READY", "READY_STATE"]):
                print(f"✅ Endpoint is ready!", file=sys.stderr)
                return endpoint
            elif "FAILED" in state.upper():
                raise Exception(f"❌ Endpoint update failed! Current State: {state}")
            
        except Exception as e:
            if "does not exist" in str(e).lower():
                if time.time() - start_time < 60:
                    print(f"   Waiting for endpoint to be created...", file=sys.stderr)
                else:
                    raise e
            else:
                raise e
        
        time.sleep(30)
    
    raise TimeoutError(f"❌ Endpoint did not become ready within {timeout}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy model to serving endpoint")
    parser.add_argument("--endpoint-name", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--workload-size", default="Small", choices=["Small", "Medium", "Large"])
    
    # Clean boolean helper text conversion
    parser.add_argument("--scale-to-zero", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), default=True)
    
    args = parser.parse_args()
    
    endpoint_name = deploy_endpoint(
        endpoint_name=args.endpoint_name,
        model_name=args.model_name,
        model_version=args.model_version,
        workload_size=args.workload_size,
        scale_to_zero=args.scale_to_zero
    )
    
    print(endpoint_name)  # Clean standard stdout print for GitHub variables