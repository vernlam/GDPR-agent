"""
LangGraph conditional edge functions for GDPR Agent routing logic.
Implements context evaluation, output verification, and source expansion routing.
"""

import logging
from .state import AgentState
from .graders import grade_retrieved_context, grade_answer_groundedness

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# LANGGRAPH CONDITIONAL EDGES (The Crossroads)
# ============================================================================

def edge_evaluate_context(state: AgentState) -> str:
    """
    Evaluates the context quality. Directs to generation or rewrite loop.
    
    Args:
        state: Current agent state with retrieval context and loop count
        
    Returns:
        Next node name: 'generate_response' or 'rewrite_query'
    """
    # Enforce loop safety limit right at the gate
    if state["retrieval_loop_count"] >= 3:
        logger.warning("Max search attempts reached (count=%d). Forcing transition to final answer generation.", 
                      state["retrieval_loop_count"])
        return "generate_response"
    
    logger.debug("Evaluating retrieved context quality for question: %s", 
                state["original_question"][:100] + "..." if len(state["original_question"]) > 100 else state["original_question"])
    
    try:
        context_is_valid = grade_retrieved_context(
            user_question=state["original_question"], 
            retrieved_context=state["retrieved_context"]
        )
    except Exception as e:
        logger.exception("Failed to grade retrieved context: %s", e)
        raise
    
    if context_is_valid:
        logger.debug("Context validation passed. Routing to response generation.")
        return "generate_response"
    else:
        logger.debug("Context validation failed. Routing to query rewrite. Loop count: %d", 
                    state["retrieval_loop_count"])
        return "rewrite_query"


def edge_verify_output(state: AgentState) -> str:
    """
    Evaluates hallucination risks. If groundedness fails, regenerate instead of re-retrieving.
    
    Args:
        state: Current agent state with generated answer and loop count
        
    Returns:
        Next node name: 'end' or 'regenerate_strict'
    """
    if state["generation_loop_count"] >= 3:
        logger.warning("Max generation loops reached (count=%d). Returning fallback with groundedness warning.",
                      state["generation_loop_count"])
        return "return_fallback"
    
    logger.debug("Verifying answer groundedness. Generation loop count: %d", 
                state["generation_loop_count"])
    
    try:
        answer_is_safe = grade_answer_groundedness(
            generated_answer=state["generated_answer"], 
            retrieved_context=state["retrieved_context"]
        )
    except Exception as e:
        logger.exception("Failed to grade answer groundedness: %s", e)
        raise
    
    if answer_is_safe:
        logger.info("Answer groundedness validation passed. Completing agent workflow.")
        return "end"  # This tells LangGraph the process is successfully complete
    else:
        logger.warning("Groundedness validation failed. Regenerating with stricter prompt.")
        return "regenerate_strict"


def edge_route_after_source_check(state: AgentState) -> str:
    """
    Route based on whether the question needs real-world examples (fines).
    
    Args:
        state: Current agent state with expansion flags
        
    Returns:
        Next node name: 'check_completeness' or 'expand_all_sources'
    """
    needs_expansion = state.get("needs_example_expansion", False)
    expanded_used = state.get("expanded_search_used", False)
    
    logger.debug("Routing after source check: needs_expansion=%s, expanded_search_used=%s", 
                needs_expansion, expanded_used)
    
    # Prevent infinite loops - only expand once
    if expanded_used:
        logger.debug("Search already expanded. Proceeding to completeness check.")
        return "check_completeness"
    
    # If needs examples and haven't expanded yet
    if needs_expansion:
        logger.debug("Question needs example expansion. Routing to expand all sources.")
        return "expand_all_sources"
    
    # Otherwise proceed normally
    logger.debug("Source coverage adequate. Proceeding to completeness check.")
    return "check_completeness"


def edge_route_after_completeness(state: AgentState) -> str:
    """
    Route based on answer completeness.
    If answer is incomplete (e.g., 'context does not specify'), expand search to all sources.
    
    Args:
        state: Current agent state with completeness flag and retry count
        
    Returns:
        Next node name: 'regenerate_strict' or 'expand_all_sources'
    """
    is_complete = state.get("is_answer_complete", True)
    retry_count = state.get("retrieval_loop_count", 0)
    expanded_used = state.get("expanded_search_used", False)
    
    logger.debug("Routing after completeness check: is_complete=%s, retry_count=%d, expanded_search_used=%s", 
                is_complete, retry_count, expanded_used)
    
    # Prevent infinite loops - only expand once
    if retry_count >= 2 or expanded_used:
        logger.debug("Proceeding to output verification (retry_count=%d or already expanded=%s)",
                    retry_count, expanded_used)
        return "verify_output"

    # If incomplete and haven't expanded yet, expand search to all sources
    if not is_complete:
        logger.debug("Answer incomplete. Expanding search to all sources.")
        return "expand_all_sources"

    # Otherwise proceed to verification
    logger.debug("Answer complete. Proceeding to verification.")
    return "verify_output"
