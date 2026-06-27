"""
Model serving endpoint deployment automation for Databricks.
Handles endpoint creation, updates, and graceful conflict resolution for staging and production workloads.
"""

import argparse
import logging
import time
import sys
from typing import Optional
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
    ServingEndpointDetailed
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def deploy_endpoint(
    endpoint_name: str,
    model_name: str,
    model_version: str,
    workload_size: str = "Small",
    scale_to_zero: bool = True,
    timeout: int = 3600
) -> str:
    """
    Deploy or update a model serving endpoint with graceful conflict handling.
    
    Creates a new endpoint if it doesn't exist, or updates an existing endpoint's
    configuration. Handles resource conflicts by waiting for in-progress updates
    to complete before retrying the operation.
    
    Args:
        endpoint_name: Name of the serving endpoint to create or update
        model_name: Fully qualified model name (e.g., "catalog.schema.model")
        model_version: Model version to deploy (as string)
        workload_size: Compute workload size - "Small", "Medium", or "Large" (default: "Small")
        scale_to_zero: Whether to enable scale-to-zero for cost savings (default: True)
        timeout: Maximum time in seconds to wait for deployment (default: 3600 = 60 minutes)
        
    Returns:
        The endpoint name that was deployed
        
    Raises:
        Exception: If deployment fails after retries or timeout is exceeded
        TimeoutError: If endpoint doesn't become ready within timeout period
    """
    logger.info("Starting endpoint deployment")
    logger.debug("Endpoint: %s, Model: %s v%s", endpoint_name, model_name, model_version)
    logger.debug("Configuration: workload=%s, scale_to_zero=%s, timeout=%ds", 
                 workload_size, scale_to_zero, timeout)
    
    try:
        w = WorkspaceClient()
        logger.debug("WorkspaceClient initialized successfully")
    except Exception as e:
        logger.exception("Failed to initialize WorkspaceClient: %s", e)
        raise
    
    # Extract model name for served entity
    short_model_name = model_name.split('.')[-1]
    logger.debug("Short model name extracted: %s", short_model_name)
    
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
    
    logger.debug("Endpoint configuration built successfully")
    
    try:
        # Try to get existing endpoint
        logger.debug("Checking if endpoint exists: %s", endpoint_name)
        existing = w.serving_endpoints.get(endpoint_name)
        logger.info("Endpoint exists, preparing to update configuration")
        
        # Check for active configuration update lock
        if existing.state and existing.state.config_update:
            config_status = str(existing.state.config_update).upper()
            if "IN_PROGRESS" in config_status or "UPDATE" in config_status:
                logger.warning("Active configuration update in progress, waiting for lock to clear")
                wait_for_endpoint(w, endpoint_name, timeout=timeout)
        
        # Update existing endpoint configuration smoothly
        try:
            logger.debug("Attempting to update endpoint configuration")
            w.serving_endpoints.update_config(
                name=endpoint_name,
                served_entities=endpoint_config.served_entities,
                traffic_config=endpoint_config.traffic_config
            )
            logger.info("Endpoint configuration update initiated successfully")
        except Exception as update_err:
            # Fallback catch if the status block hadn't propagated the lock yet
            error_str = str(update_err).upper().replace(" ", "_")
            if "RESOURCE_CONFLICT" in error_str or "BEING_UPDATED" in error_str:
                logger.warning("Resource conflict detected, waiting for active deployment to complete")
                wait_for_endpoint(w, endpoint_name, timeout=timeout)
                # Re-try modification step once lock releases
                logger.debug("Retrying configuration update after lock release")
                w.serving_endpoints.update_config(
                    name=endpoint_name,
                    served_entities=endpoint_config.served_entities,
                    traffic_config=endpoint_config.traffic_config
                )
                logger.info("Endpoint configuration update succeeded after retry")
            else:
                logger.exception("Endpoint update failed with unexpected error: %s", update_err)
                raise update_err
        
        logger.info("Endpoint update in progress (typically 5-10 minutes)")
        
    except Exception as e:
        error_str = str(e)
        if "RESOURCE_DOES_NOT_EXIST" in error_str or "does not exist" in error_str.lower():
            logger.info("Endpoint does not exist, creating new endpoint: %s", endpoint_name)
            
            # Create fresh model serving endpoint
            try:
                w.serving_endpoints.create(
                    name=endpoint_name,
                    config=endpoint_config
                )
                logger.info("Endpoint creation initiated successfully")
                logger.info("Initial deployment in progress (typically 10-15 minutes, may take up to 45 minutes with dependencies)")
            except Exception as create_err:
                logger.exception("Failed to create endpoint: %s", create_err)
                raise create_err
        else:
            logger.exception("Unexpected error during endpoint deployment: %s", e)
            raise e
    
    # Wait for endpoint to reach operational status
    logger.debug("Waiting for endpoint to become ready")
    wait_for_endpoint(w, endpoint_name, timeout=timeout)
    
    logger.info("Deployment completed successfully")
    endpoint_url = f"https://{w.config.host}/serving-endpoints/{endpoint_name}"
    logger.info("Endpoint URL: %s", endpoint_url)
    
    # Write status to stderr for CI/CD visibility
    print(f"Deployment complete! Endpoint URL: {endpoint_url}", file=sys.stderr)
    
    return endpoint_name


def wait_for_endpoint(
    client: WorkspaceClient, 
    endpoint_name: str, 
    timeout: int = 3600
) -> Optional[ServingEndpointDetailed]:
    """
    Wait for endpoint to reach READY state with status polling.
    
    Polls the endpoint status every 30 seconds until it reaches a READY state
    or the timeout is exceeded. Handles transient errors during endpoint creation.
    
    Args:
        client: Initialized Databricks WorkspaceClient
        endpoint_name: Name of the endpoint to monitor
        timeout: Maximum time in seconds to wait (default: 3600 = 60 minutes)
        
    Returns:
        The endpoint object once it reaches READY state, or None if error occurs
        
    Raises:
        TimeoutError: If endpoint doesn't become ready within timeout period
        Exception: If endpoint deployment fails or endpoint doesn't exist after initial grace period
    """
    logger.info("Starting endpoint readiness check: %s", endpoint_name)
    logger.debug("Timeout: %d seconds (%d minutes)", timeout, timeout // 60)
    
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
            
            # Only print/log if state changed
            if state != last_state:
                elapsed = int(time.time() - start_time)
                minutes = elapsed // 60
                seconds = elapsed % 60
                logger.info("Status: %s (%dm %ds elapsed)", state, minutes, seconds)
                # Also write to stderr for CI/CD visibility
                print(f"   Status: {state} ({minutes}m {seconds}s elapsed)", file=sys.stderr)
                last_state = state
            
            # Check if ready (case-insensitive)
            if state == "READY" or "NOT_UPDATING" in state:
                logger.info("Endpoint is ready")
                print("Endpoint is ready!", file=sys.stderr)
                return endpoint
            elif "FAIL" in state:
                error_msg = f"Endpoint deployment failed! Current state: {state}"
                logger.error(error_msg)
                raise Exception(error_msg)
            
        except Exception as e:
            error_str = str(e).lower()
            if "does not exist" in error_str:
                elapsed = time.time() - start_time
                if elapsed < 120:
                    logger.debug("Endpoint not yet created, waiting... (elapsed: %.1fs)", elapsed)
                    print("   Waiting for endpoint to be created...", file=sys.stderr)
                else:
                    logger.error("Endpoint still does not exist after %.1fs", elapsed)
                    raise e
            else:
                logger.exception("Error while checking endpoint status: %s", e)
                raise e
        
        time.sleep(30)
    
    minutes = timeout // 60
    error_msg = f"Endpoint did not become ready within {timeout}s ({minutes} minutes)"
    logger.error(error_msg)
    raise TimeoutError(error_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy model to serving endpoint")
    parser.add_argument("--endpoint-name", required=True, help="Name of the serving endpoint")
    parser.add_argument("--model-name", required=True, help="Fully qualified model name (catalog.schema.model)")
    parser.add_argument("--model-version", required=True, help="Model version to deploy")
    parser.add_argument("--workload-size", default="Small", choices=["Small", "Medium", "Large"], 
                        help="Compute workload size (default: Small)")
    parser.add_argument("--scale-to-zero", type=lambda x: (str(x).lower() in ['true', '1', 'yes']), 
                        default=True, help="Enable scale-to-zero (default: true)")
    parser.add_argument("--timeout", type=int, default=3600, 
                        help="Timeout in seconds (default: 3600 = 60 minutes)")
    
    args = parser.parse_args()
    
    logger.info("Starting deployment script")
    logger.debug("Arguments: endpoint=%s, model=%s, version=%s, workload=%s, scale_to_zero=%s, timeout=%d",
                 args.endpoint_name, args.model_name, args.model_version, 
                 args.workload_size, args.scale_to_zero, args.timeout)
    
    try:
        endpoint_name = deploy_endpoint(
            endpoint_name=args.endpoint_name,
            model_name=args.model_name,
            model_version=args.model_version,
            workload_size=args.workload_size,
            scale_to_zero=args.scale_to_zero,
            timeout=args.timeout
        )
        
        # Clean standard stdout print for GitHub Actions variables
        print(endpoint_name)
        logger.info("Deployment script completed successfully")
        
    except Exception as e:
        logger.exception("Deployment script failed: %s", e)
        sys.exit(1)
