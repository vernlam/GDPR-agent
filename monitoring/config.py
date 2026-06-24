"""
Monitoring configuration and constants.
"""
import os
from dataclasses import dataclass

@dataclass
class MonitoringConfig:
    """Configuration for GDPR agent monitoring"""
    
    # Endpoint configuration
    ENDPOINT_NAME: str = "gdpr-agent-staging"
    
    # NEW: System tables for AI Gateway inference logs
    INFERENCE_REQUEST_LOGS: str = "system.serving.gdpr_agent_inference_logs"
    
    # Table names for monitoring metrics (you create these)
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
    
    # OpenAI configuration
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_JUDGE_MODEL: str = "gpt-4o-mini"
    
    # Cost estimation (per 1M tokens)
    OPENAI_INPUT_COST: float = 0.15
    OPENAI_OUTPUT_COST: float = 0.60
    
    @property
    def payload_table(self) -> str:
        """Legacy property - now uses system table"""
        return self.INFERENCE_REQUEST_LOGS
    
    @property
    def response_table(self) -> str:
        """Legacy property - now uses system table"""
        return self.INFERENCE_REQUEST_LOGS

# Global config instance
config = MonitoringConfig()