"""
Monitoring configuration and constants.

Provides centralized configuration for GDPR agent monitoring system,
including endpoint settings, table names, alert thresholds, and OpenAI
API integration. Supports Databricks Secrets for secure credential management
with fallback to environment variables for local development.
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def _get_openai_api_key() -> str:
    """
    Get OpenAI API key from Databricks Secrets.
    
    Attempts to retrieve the API key from Databricks Secrets (scope="openai",
    key="GDPR_agent"). Falls back to environment variable OPENAI_API_KEY for
    local development if Databricks SDK is unavailable or secrets retrieval fails.
    
    Returns:
        OpenAI API key string (empty string if not found)
    
    Raises:
        Does not raise exceptions; returns empty string on error
    """
    logger.debug("Attempting to retrieve OpenAI API key")
    
    try:
        from databricks.sdk import WorkspaceClient
        logger.debug("Databricks SDK imported successfully")
        
        w = WorkspaceClient()
        logger.debug("WorkspaceClient initialized")
        
        api_key = w.dbutils.secrets.get(scope="openai", key="GDPR_agent")
        logger.info("Successfully retrieved OpenAI API key from Databricks Secrets")
        return api_key
        
    except ImportError as e:
        logger.warning("Databricks SDK not available, falling back to environment variable: %s", e)
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            logger.info("Retrieved OpenAI API key from environment variable")
        else:
            logger.warning("No OpenAI API key found in environment variable OPENAI_API_KEY")
        return api_key
        
    except AttributeError as e:
        logger.warning("dbutils.secrets not available (likely not in Databricks notebook context): %s", e)
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            logger.info("Retrieved OpenAI API key from environment variable")
        else:
            logger.warning("No OpenAI API key found in environment variable OPENAI_API_KEY")
        return api_key
        
    except Exception as e:
        logger.exception("Failed to retrieve OpenAI API key from Databricks Secrets: %s", e)
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            logger.info("Retrieved OpenAI API key from environment variable (fallback)")
        else:
            logger.warning("No OpenAI API key found in environment variable OPENAI_API_KEY (fallback)")
        return api_key


@dataclass
class MonitoringConfig:
    """
    Configuration for GDPR agent monitoring system.
    
    Centralizes all configuration parameters including endpoint names,
    Delta table locations, monitoring parameters, alert thresholds, and
    OpenAI API settings. Provides backward compatibility properties for
    legacy code.
    """
    
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
    
    def __post_init__(self) -> None:
        """
        Post-initialization validation and logging.
        
        Logs configuration summary and validates critical settings.
        
        Raises:
            Does not raise exceptions; logs warnings for invalid settings
        """
        try:
            logger.info("MonitoringConfig initialized with endpoint: %s", self.ENDPOINT_NAME)
            logger.debug("Inference logs table: %s", self.INFERENCE_LOGS_TABLE)
            logger.debug("Quality metrics table: %s", self.QUALITY_METRICS_TABLE)
            logger.debug("Performance metrics table: %s", self.PERFORMANCE_METRICS_TABLE)
            logger.debug("Error log table: %s", self.ERROR_LOG_TABLE)
            logger.debug("Default lookback days: %d", self.DEFAULT_LOOKBACK_DAYS)
            logger.debug("Sample size for evaluation: %d", self.SAMPLE_SIZE_FOR_EVALUATION)
            
            # Validate API key is present
            if not self.OPENAI_API_KEY:
                logger.warning("OpenAI API key is not set - quality evaluation will fail")
            else:
                logger.debug("OpenAI API key is configured (length: %d chars)", len(self.OPENAI_API_KEY))
            
            # Log alert thresholds
            logger.debug("Alert thresholds: min_quality=%.1f, min_success_rate=%.1f%%, "
                        "max_latency=%.1fs, max_errors=%d/day, max_refusal_rate=%.1f%%, "
                        "max_insufficient_context=%d/day",
                        self.MIN_QUALITY_SCORE, self.MIN_SUCCESS_RATE,
                        self.MAX_AVG_LATENCY, self.MAX_ERRORS_PER_DAY,
                        self.MAX_REFUSAL_RATE * 100, self.MAX_INSUFFICIENT_CONTEXT_PER_DAY)
            
        except Exception as e:
            logger.exception("Error during MonitoringConfig post-initialization: %s", e)
    
    @property
    def payload_table(self) -> str:
        """
        Legacy property - maps to custom inference logs table.
        
        Provided for backward compatibility with code that expects
        a separate payload_table attribute.
        
        Returns:
            Path to inference logs table
        """
        logger.debug("Legacy payload_table property accessed (returns INFERENCE_LOGS_TABLE)")
        return self.INFERENCE_LOGS_TABLE
    
    @property
    def response_table(self) -> str:
        """
        Legacy property - maps to custom inference logs table.
        
        Provided for backward compatibility with code that expects
        a separate response_table attribute.
        
        Returns:
            Path to inference logs table
        """
        logger.debug("Legacy response_table property accessed (returns INFERENCE_LOGS_TABLE)")
        return self.INFERENCE_LOGS_TABLE
    
    def get_table_config(self) -> dict:
        """
        Get all table configuration as a dictionary.
        
        Returns:
            Dict containing all table paths keyed by purpose
            (returns empty dict on error)
        
        Raises:
            Does not raise exceptions; returns empty dict on error
        """
        try:
            config_dict = {
                'inference_logs': self.INFERENCE_LOGS_TABLE,
                'quality_metrics': self.QUALITY_METRICS_TABLE,
                'performance_metrics': self.PERFORMANCE_METRICS_TABLE,
                'error_logs': self.ERROR_LOG_TABLE,
            }
            logger.debug("Retrieved table configuration with %d entries", len(config_dict))
            return config_dict
        except Exception as e:
            logger.exception("Error getting table configuration: %s", e)
            return {}
    
    def get_alert_thresholds(self) -> dict:
        """
        Get all alert threshold configuration as a dictionary.
        
        Returns:
            Dict containing all alert thresholds keyed by metric name
            (returns empty dict on error)
        
        Raises:
            Does not raise exceptions; returns empty dict on error
        """
        try:
            thresholds = {
                'min_quality_score': self.MIN_QUALITY_SCORE,
                'min_success_rate': self.MIN_SUCCESS_RATE,
                'max_avg_latency': self.MAX_AVG_LATENCY,
                'max_errors_per_day': self.MAX_ERRORS_PER_DAY,
                'max_refusal_rate': self.MAX_REFUSAL_RATE,
                'max_insufficient_context_per_day': self.MAX_INSUFFICIENT_CONTEXT_PER_DAY,
            }
            logger.debug("Retrieved alert thresholds with %d entries", len(thresholds))
            return thresholds
        except Exception as e:
            logger.exception("Error getting alert thresholds: %s", e)
            return {}
    
    def validate_config(self) -> tuple:
        """
        Validate configuration for completeness and consistency.
        
        Checks that all required settings are present and within valid ranges.
        
        Returns:
            Tuple of (is_valid: bool, errors: list of error messages)
            (returns (False, ["Validation failed"]) on exception)
        
        Raises:
            Does not raise exceptions; returns error tuple on failure
        """
        try:
            errors = []
            
            # Check required string fields are not empty
            if not self.ENDPOINT_NAME:
                errors.append("ENDPOINT_NAME is empty")
            if not self.INFERENCE_LOGS_TABLE:
                errors.append("INFERENCE_LOGS_TABLE is empty")
            if not self.LLM_JUDGE_MODEL:
                errors.append("LLM_JUDGE_MODEL is empty")
            
            # Check numeric ranges
            if self.DEFAULT_LOOKBACK_DAYS < 1:
                errors.append(f"DEFAULT_LOOKBACK_DAYS must be >= 1 (got {self.DEFAULT_LOOKBACK_DAYS})")
            if self.SAMPLE_SIZE_FOR_EVALUATION < 1:
                errors.append(f"SAMPLE_SIZE_FOR_EVALUATION must be >= 1 (got {self.SAMPLE_SIZE_FOR_EVALUATION})")
            
            # Check threshold ranges
            if self.MIN_QUALITY_SCORE < 0 or self.MIN_QUALITY_SCORE > 5:
                errors.append(f"MIN_QUALITY_SCORE must be 0-5 (got {self.MIN_QUALITY_SCORE})")
            if self.MIN_SUCCESS_RATE < 0 or self.MIN_SUCCESS_RATE > 100:
                errors.append(f"MIN_SUCCESS_RATE must be 0-100 (got {self.MIN_SUCCESS_RATE})")
            if self.MAX_AVG_LATENCY < 0:
                errors.append(f"MAX_AVG_LATENCY must be >= 0 (got {self.MAX_AVG_LATENCY})")
            if self.MAX_REFUSAL_RATE < 0 or self.MAX_REFUSAL_RATE > 1:
                errors.append(f"MAX_REFUSAL_RATE must be 0-1 (got {self.MAX_REFUSAL_RATE})")
            
            # Warn if API key is missing but don't fail validation
            if not self.OPENAI_API_KEY:
                logger.warning("OpenAI API key is not configured - quality evaluation will not work")
            
            is_valid = len(errors) == 0
            
            if is_valid:
                logger.info("Configuration validation passed")
            else:
                logger.warning("Configuration validation failed with %d errors: %s", 
                             len(errors), "; ".join(errors))
            
            return (is_valid, errors)
            
        except Exception as e:
            logger.exception("Error during configuration validation: %s", e)
            return (False, ["Validation failed due to exception"])


# Global config instance
try:
    config = MonitoringConfig()
    logger.info("Global MonitoringConfig instance created successfully")
except Exception as e:
    logger.exception("Failed to create global MonitoringConfig instance: %s", e)
    # Create a minimal fallback config
    config = None
    logger.error("Global config is None - monitoring will not work correctly")
