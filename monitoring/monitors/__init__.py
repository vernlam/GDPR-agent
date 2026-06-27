"""
Monitor modules for different aspects of the serving endpoint.

This package provides monitoring capabilities for GDPR Agent serving endpoints,
including quality evaluation, performance tracking, error analysis, cost estimation,
and drift detection.

Available monitors:
    - QualityMonitor: Evaluate response quality using LLM-as-a-judge
    - PerformanceMonitor: Track latency, throughput, and success rates
    - ErrorMonitor: Analyze errors and detect patterns
    - CostMonitor: Estimate and track OpenAI API costs
    - DriftMonitor: Detect query distribution changes and drift
"""

import logging
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Import monitor classes with error handling
logger.debug("Initializing monitoring.monitors package")

try:
    from monitoring.monitors.quality_monitor import QualityMonitor
    logger.debug("QualityMonitor imported successfully")
except ImportError as e:
    logger.exception("Failed to import QualityMonitor: %s", e)
    raise
except Exception as e:
    logger.exception("Unexpected error importing QualityMonitor: %s", e)
    raise

try:
    from monitoring.monitors.performance_monitor import PerformanceMonitor
    logger.debug("PerformanceMonitor imported successfully")
except ImportError as e:
    logger.exception("Failed to import PerformanceMonitor: %s", e)
    raise
except Exception as e:
    logger.exception("Unexpected error importing PerformanceMonitor: %s", e)
    raise

try:
    from monitoring.monitors.error_monitor import ErrorMonitor
    logger.debug("ErrorMonitor imported successfully")
except ImportError as e:
    logger.exception("Failed to import ErrorMonitor: %s", e)
    raise
except Exception as e:
    logger.exception("Unexpected error importing ErrorMonitor: %s", e)
    raise

try:
    from monitoring.monitors.cost_monitor import CostMonitor
    logger.debug("CostMonitor imported successfully")
except ImportError as e:
    logger.exception("Failed to import CostMonitor: %s", e)
    raise
except Exception as e:
    logger.exception("Unexpected error importing CostMonitor: %s", e)
    raise

try:
    from monitoring.monitors.drift_monitor import DriftMonitor
    logger.debug("DriftMonitor imported successfully")
except ImportError as e:
    logger.exception("Failed to import DriftMonitor: %s", e)
    raise
except Exception as e:
    logger.exception("Unexpected error importing DriftMonitor: %s", e)
    raise

__all__: List[str] = [
    "QualityMonitor",
    "PerformanceMonitor",
    "ErrorMonitor",
    "CostMonitor",
    "DriftMonitor",
]

logger.info("monitoring.monitors package initialized successfully with %d monitors", len(__all__))
logger.debug("Available monitors: %s", ", ".join(__all__))
