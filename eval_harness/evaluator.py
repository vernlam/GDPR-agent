"""
Evaluation logic for GDPR Agent test cases.
Scores agent responses against expected behavior with source verification and content matching.
"""

import logging
from typing import Dict, List, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def evaluate_case(test_case: Dict, agent_response: Dict) -> Dict:
    """
    Score agent response against expected behavior with source and content validation.
    
    Evaluates agent responses on two dimensions: (1) source correctness - whether the
    agent retrieved from expected knowledge sources, and (2) content match - whether
    the response includes required information items. Combines scores and determines
    pass/fail with 0.7 threshold.
    
    Args:
        test_case: Test case dictionary containing "expected_behavior" with:
            - sources (list): Expected knowledge source identifiers
            - must_retrieve_from_articles (list): Required article references
            - must_include_in_answer (list): Required content items in answer
        agent_response: Agent output dictionary containing:
            - answer (str): Generated response text
            - context (str or list): Retrieved context documents
            
    Returns:
        Dictionary containing:
            - scores (dict): Numeric scores (source_correct, content_match, total)
            - feedback (list): Human-readable evaluation messages
            - passed (bool): Whether evaluation passed (total >= 0.7)
            
    Raises:
        Exception: If scoring logic fails (logged but not re-raised, returns error result)
    """
    logger.debug("Starting evaluation for test case")
    
    try:
        # Extract expected behavior and agent outputs
        expected = test_case.get("expected_behavior", {})
        answer = agent_response.get("answer", "")
        context_raw = agent_response.get("context", "")
        
        # Handle context as string or list
        if isinstance(context_raw, list):
            context = " ".join(str(item) for item in context_raw)
            logger.debug("Converted context list to string (%d items)", len(context_raw))
        else:
            context = str(context_raw)
            logger.debug("Using context as string")
        
        logger.debug("Answer length: %d chars", len(answer))
        logger.debug("Context length: %d chars", len(context))
        
    except Exception as e:
        logger.exception("Failed to extract test case data: %s", e)
        return {
            "scores": {"source_correct": 0, "content_match": 0, "total": 0},
            "feedback": [f"Error: Failed to parse test case - {str(e)}"],
            "passed": False
        }
    
    # Initialize scores and feedback
    scores = {"source_correct": 0, "content_match": 0, "total": 0}
    feedback: List[str] = []
    
    # Source checking logic
    try:
        logger.debug("Checking source correctness")
        expected_sources = expected.get("sources", [])
        logger.debug("Expected sources: %s", expected_sources)
        
        if expected_sources:
            source_found = any(src.lower() in context.lower() for src in expected_sources)
            
            if source_found:
                scores["source_correct"] = 1
                feedback.append(f"PASS: Retrieved from expected sources: {expected_sources}")
                logger.debug("Source check passed: found expected source")
            else:
                scores["source_correct"] = 0
                feedback.append(f"FAIL: Expected sources {expected_sources} not found")
                logger.debug("Source check failed: no expected sources found")
        else:
            # No sources specified, default to pass
            scores["source_correct"] = 1
            logger.debug("No expected sources specified, defaulting to pass")
            
    except Exception as e:
        logger.exception("Failed during source checking: %s", e)
        scores["source_correct"] = 0
        feedback.append(f"Error: Source checking failed - {str(e)}")
    
    # Content matching logic
    try:
        logger.debug("Checking content match")
        must_retrieve = expected.get("must_retrieve_from_articles", [])
        must_include = expected.get("must_include_in_answer", [])
        all_required_items = must_retrieve + must_include
        
        logger.debug("Required items to find: %d", len(all_required_items))
        
        if all_required_items:
            combined_text = (answer + " " + context).lower()
            found = [item for item in all_required_items if item.lower() in combined_text]
            coverage = len(found) / len(all_required_items) if all_required_items else 0
            scores["content_match"] = coverage
            
            logger.debug("Content coverage: %.2f (%d/%d items found)", 
                        coverage, len(found), len(all_required_items))
            
            if coverage >= 0.7:
                feedback.append(f"PASS: Found {len(found)}/{len(all_required_items)} expected items")
                logger.debug("Content match passed: coverage >= 0.7")
            else:
                missing: Set[str] = set(all_required_items) - set(found)
                feedback.append(f"FAIL: Missing {len(missing)} items: {missing}")
                logger.debug("Content match failed: coverage < 0.7, missing %d items", len(missing))
        else:
            # No required items, default to pass
            scores["content_match"] = 1
            logger.debug("No required content items specified, defaulting to pass")
            
    except Exception as e:
        logger.exception("Failed during content matching: %s", e)
        scores["content_match"] = 0
        feedback.append(f"Error: Content matching failed - {str(e)}")
    
    # Calculate total score
    try:
        scores["total"] = (scores["source_correct"] + scores["content_match"]) / 2
        passed = scores["total"] >= 0.7
        
        logger.info("Evaluation complete: total=%.2f, passed=%s", scores["total"], passed)
        logger.debug("Final scores: source_correct=%.2f, content_match=%.2f", 
                    scores["source_correct"], scores["content_match"])
        
    except Exception as e:
        logger.exception("Failed to calculate total score: %s", e)
        scores["total"] = 0
        passed = False
        feedback.append(f"Error: Score calculation failed - {str(e)}")
    
    return {
        "scores": scores,
        "feedback": feedback,
        "passed": passed
    }
