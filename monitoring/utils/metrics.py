"""
Metric calculation utilities.

Provides statistical calculations, success rate computation, token estimation,
and cost estimation for monitoring agent performance and operational metrics.
"""

import logging
import statistics
from typing import Dict, List, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_summary_stats(values: List[float]) -> Dict[str, Any]:
    """
    Calculate summary statistics for a list of values.
    
    Computes mean, median, min, max, and count for numeric data.
    Returns zeros for all metrics if the input list is empty.
    
    Args:
        values: List of numeric values to analyze
    
    Returns:
        Dict containing:
        - mean: Arithmetic mean of values (or 0 if empty)
        - median: Median value (or 0 if empty)
        - min: Minimum value (or 0 if empty)
        - max: Maximum value (or 0 if empty)
        - count: Number of values (or 0 if empty)
        Returns dict with 0 values if calculation fails.
    
    Raises:
        Does not raise exceptions; returns safe defaults on error
    """
    logger.debug("Calculating summary statistics for %d values", len(values) if values else 0)
    
    try:
        if not values:
            logger.debug("Empty values list, returning zero statistics")
            return {"mean": 0, "median": 0, "min": 0, "max": 0, "count": 0}
        
        result = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "count": len(values)
        }
        
        logger.debug("Statistics calculated: mean=%.2f, median=%.2f, min=%.2f, max=%.2f, count=%d",
                    result["mean"], result["median"], result["min"], result["max"], result["count"])
        
        return result
        
    except statistics.StatisticsError as e:
        logger.exception("Statistics calculation error: %s", e)
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "count": 0}
    except (TypeError, ValueError) as e:
        logger.exception("Invalid value type in statistics calculation: %s", e)
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "count": 0}
    except Exception as e:
        logger.exception("Unexpected error calculating summary statistics: %s", e)
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "count": 0}


def calculate_success_rate(total: int, successful: int) -> float:
    """
    Calculate success rate percentage.
    
    Computes the percentage of successful operations out of total operations.
    Handles division by zero and invalid inputs gracefully.
    
    Args:
        total: Total number of operations
        successful: Number of successful operations
    
    Returns:
        Success rate as a percentage (0.0 to 100.0).
        Returns 0.0 if total is 0, negative values detected, or calculation fails.
        Returns 100.0 if successful > total.
    
    Raises:
        Does not raise exceptions; returns safe value on error
    """
    logger.debug("Calculating success rate: %d successful out of %d total", successful, total)
    
    try:
        if total == 0:
            logger.debug("Total is zero, returning 0.0 success rate")
            return 0.0
        
        if successful < 0 or total < 0:
            logger.warning("Negative values detected: total=%d, successful=%d. Returning 0.0", total, successful)
            return 0.0
        
        if successful > total:
            logger.warning("Successful count (%d) exceeds total (%d). Returning 100.0", successful, total)
            return 100.0
        
        rate = (successful / total) * 100
        logger.debug("Success rate calculated: %.2f%%", rate)
        
        return rate
        
    except (TypeError, ZeroDivisionError) as e:
        logger.exception("Error calculating success rate: %s", e)
        return 0.0
    except Exception as e:
        logger.exception("Unexpected error calculating success rate: %s", e)
        return 0.0


def estimate_tokens(text: str) -> int:
    """
    Rough token estimation for LLM API calls.
    
    Uses a simple heuristic: approximately 4 characters per token.
    This is a rough estimate and may vary by model and tokenizer.
    
    Args:
        text: Input text to estimate tokens for
    
    Returns:
        Estimated number of tokens (or 0 on error)
    
    Raises:
        Does not raise exceptions; returns 0 on error
    """
    try:
        if not text:
            logger.debug("Empty or None text provided, returning 0 tokens")
            return 0
        
        token_count = len(text) // 4
        logger.debug("Estimated %d tokens for text of length %d", token_count, len(text))
        
        return token_count
        
    except (TypeError, AttributeError) as e:
        logger.exception("Error estimating tokens (invalid text type): %s", e)
        return 0
    except Exception as e:
        logger.exception("Unexpected error estimating tokens: %s", e)
        return 0


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """
    Estimate OpenAI API cost based on token usage.
    
    Calculates total cost using per-million-token pricing from config.
    Cost formula: (tokens / 1M) × price_per_million for both input and output.
    
    Args:
        input_tokens: Number of input tokens consumed
        output_tokens: Number of output tokens generated
    
    Returns:
        Estimated cost in USD (or 0.0 on error)
    
    Raises:
        Does not raise exceptions; returns 0.0 on error
    """
    logger.debug("Estimating cost: %d input tokens, %d output tokens", input_tokens, output_tokens)
    
    try:
        from monitoring.config import config
        
        if input_tokens < 0 or output_tokens < 0:
            logger.warning("Negative token counts detected: input=%d, output=%d. Returning 0.0", 
                         input_tokens, output_tokens)
            return 0.0
        
        input_cost = (input_tokens / 1_000_000) * config.OPENAI_INPUT_COST
        output_cost = (output_tokens / 1_000_000) * config.OPENAI_OUTPUT_COST
        total_cost = input_cost + output_cost
        
        logger.debug("Cost estimated: input=$%.6f, output=$%.6f, total=$%.6f",
                    input_cost, output_cost, total_cost)
        
        return total_cost
        
    except (ImportError, AttributeError) as e:
        logger.exception("Failed to import config or access cost constants: %s", e)
        return 0.0
    except (TypeError, ZeroDivisionError) as e:
        logger.exception("Error calculating cost: %s", e)
        return 0.0
    except Exception as e:
        logger.exception("Unexpected error estimating cost: %s", e)
        return 0.0
