"""
GDPR Agent Monitoring Package

Provides comprehensive monitoring for the GDPR agent serving endpoint including:
- Quality evaluation (LLM-as-a-judge)
- Performance metrics (latency, throughput)
- Error tracking and analysis
- Cost monitoring
- Query distribution and drift detection
- Dashboard data preparation
"""

__version__ = "1.0.0"

from monitoring.config import config, MonitoringConfig
from monitoring.main import run_monitoring
from monitoring.utils.request_logger import RequestLogger

__all__ = [
    "config",
    "MonitoringConfig",
    "run_monitoring",
    "RequestLogger"
]