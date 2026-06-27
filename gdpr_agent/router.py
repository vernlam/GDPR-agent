"""
Query routing logic for GDPR Agent.
Determines which data sources (fines, legislation, policy) to query based on question semantics.
"""

import json
import logging
from typing import Dict
from . import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def route_query(question: str) -> Dict[str, bool]:
    """
    Analyze the question and determine which data sources to query.
    
    Uses LLM-based classification to identify whether the question requires:
    - Historical fines and enforcement examples
    - GDPR legislation and legal requirements
    - Internal policy and operational guidelines
    
    Args:
        question: The user's compliance question to analyze
        
    Returns:
        Dictionary with boolean flags:
            - query_fines: Whether to search enforcement history
            - query_legislation: Whether to search GDPR articles/regulations
            - query_policy: Whether to search internal policies
            
    Raises:
        Exception: If OpenAI API call fails (re-raised after logging)
        json.JSONDecodeError: If response is not valid JSON (re-raised after logging)
    """
    logger.info("Routing query to appropriate data sources")
    logger.debug("Question: %s", question[:100] + "..." if len(question) > 100 else question)
    
    prompt = f"""Analyse this GDPR compliance question and determine which data sources are needed.

    Question: {question}

    Return JSON with boolean flags:
    - "query_fines": true if question asks about enforcement examples, penalties, fines, or specific companies
    - "query_legislation": true if question asks about legal requirements, articles, regulations, or rights
    - "query_policy": true if question asks about internal procedures, retention periods, or operational guidelines

    Examples:
    - "What fines have companies received?" → {{"query_fines": true, "query_legislation": false, "query_policy": false}}
    - "What does Article 17 say?" → {{"query_fines": false, "query_legislation": true, "query_policy": false}}
    - "How long do we retain customer data?" → {{"query_fines": false, "query_legislation": false, "query_policy": true}}
    - "Right to be forgotten requirements and penalties" → {{"query_fines": true, "query_legislation": true, "query_policy": true}}

    JSON Output:"""

    try:
        logger.debug("Calling OpenAI API for query routing classification")
        response = config.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.0,
            seed=42
        )
        logger.debug("OpenAI API call completed successfully")
    except Exception as e:
        logger.exception("Failed to call OpenAI API for query routing: %s", e)
        raise
    
    try:
        routing_decision = json.loads(response.choices[0].message.content)
        logger.info("Query routing completed: fines=%s, legislation=%s, policy=%s",
                   routing_decision.get('query_fines', False),
                   routing_decision.get('query_legislation', False),
                   routing_decision.get('query_policy', False))
        return routing_decision
    except json.JSONDecodeError as e:
        logger.exception("Failed to parse routing response as JSON: %s", e)
        logger.debug("Response content: %s", response.choices[0].message.content)
        raise
