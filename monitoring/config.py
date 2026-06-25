"""
Monitoring configuration and constants.
"""
import os
from dataclasses import dataclass

def _get_openai_api_key() -> str:
    """
    Get OpenAI API key from Databricks Secrets.
    
    Falls back to environment variable for local development.
    """
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        return w.dbutils.secrets.get(scope="openai", key="GDPR_agent")
    except Exception:
        # Fallback to environment variable for local dev
        return os.getenv("OPENAI_API_KEY", "")

@dataclass
class MonitoringConfig:
    """Configuration for GDPR agent monitoring"""
    
    # Endpoint configuration
    ENDPOINT_NAME: str = "gdpr-agent-staging"
    
    # Custom inference logs table (created by logging notebook)
    INFERENCE_LOGS_TABLE: str = "main.default.gdpr_agent_inference_logs"
    
    # Table names for monitoring metrics (created by monitors)
    QUALITY_METRICS_TABLE: str = "main.default.gdpr_agent_quality_metrics"
    PERFORMANCE_METRICS_TABLE: str = "main.default.gdpr_agent_performance_metrics"
    ERROR_LOG_TABLE: str = "main.default.gdpr_agent_error_logs"
    
    # Monitoring parameters
    DEFAULT_LOOKBACK_DAYS: int = 7
    SAMPLE_SIZE_FOR_EVALUATION: int = 20
    
    # Alert thresholds
    MIN_QUALITY_SCORE: float = 3.0
    MIN_SUCCESS_RATE: float = 95.0
    MAX_AVG_LATENCY: float = 10.0
    MAX_ERRORS_PER_DAY: int = 5
    MAX_REFUSAL_RATE: float = 0.15
    MAX_INSUFFICIENT_CONTEXT_PER_DAY: int = 10
    
    # OpenAI configuration - fetched from Databricks Secrets
    OPENAI_API_KEY: str = _get_openai_api_key()
    LLM_JUDGE_MODEL: str = "gpt-4o-mini"
    
    # Cost estimation (per 1M tokens)
    OPENAI_INPUT_COST: float = 0.15
    OPENAI_OUTPUT_COST: float = 0.60

# For backwards compatibility with monitors that expect these
@property
def payload_table(self):
    """Legacy property - maps to custom inference logs table"""
    return self.INFERENCE_LOGS_TABLE

@property
def response_table(self):
    """Legacy property - maps to custom inference logs table"""  
    return self.INFERENCE_LOGS_TABLE

# Global config instance
config = MonitoringConfig()
