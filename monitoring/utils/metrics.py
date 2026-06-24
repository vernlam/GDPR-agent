"""
Metric calculation utilities.
"""
from typing import Dict, List
import statistics

def calculate_summary_stats(values: List[float]) -> Dict[str, float]:
    """Calculate summary statistics for a list of values"""
    if not values:
        return {"mean": 0, "median": 0, "min": 0, "max": 0, "count": 0}
    
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "count": len(values)
    }

def calculate_success_rate(total: int, successful: int) -> float:
    """Calculate success rate percentage"""
    if total == 0:
        return 0.0
    return (successful / total) * 100

def estimate_tokens(text: str) -> int:
    """Rough token estimation (4 chars ≈ 1 token)"""
    return len(text) // 4

def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate OpenAI API cost"""
    from monitoring.config import config
    input_cost = (input_tokens / 1_000_000) * config.OPENAI_INPUT_COST
    output_cost = (output_tokens / 1_000_000) * config.OPENAI_OUTPUT_COST
    return input_cost + output_cost