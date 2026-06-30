"""
LangGraph node implementations for GDPR Agent workflow.
Handles retrieval, generation, query rewriting, and quality control operations.
"""

import logging
from typing import Dict, Any
import mlflow
from .state import AgentState
from .tools import tool_search_retail_policy, tool_search_gdpr_legislation, tool_search_historical_fines
from . import config
from .router import route_query

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# RETRIEVAL NODE: Cross-Index Retrieval
# ============================================================================
def node_route_and_retrieve(state: AgentState) -> Dict[str, Any]:
    """
    Routes query to appropriate indices based on question type.
    
    Args:
        state: Current agent state with query and loop count
        
    Returns:
        Dictionary with retrieved_context and updated retrieval_loop_count
    """
    query_to_search = state["current_query"]
    
    logger.info("Starting query routing and retrieval")
    logger.debug("Query to search: %s", query_to_search[:100] + "..." if len(query_to_search) > 100 else query_to_search)
    
    # Determine which sources to query
    try:
        routing = route_query(state["original_question"])
        logger.info("Routing decision: Fines=%s, Legislation=%s, Policy=%s",
                   routing.get('query_fines'), routing.get('query_legislation'), routing.get('query_policy'))
    except Exception as e:
        logger.exception("Failed to determine query routing: %s", e)
        raise
    
    retrieved_contexts = []

    # Query policy index
    if routing.get("query_policy", False):
        try:
            policy_results = tool_search_retail_policy(query_text=query_to_search, top_k=3)
            policy_rows = policy_results.get('result', {}).get('data_array', [])
            policy_count = 0
            for row in policy_rows:
                if row[-1] > 0.35:
                    retrieved_contexts.append(f"[SOURCE: Internal Retail Policy | Section: {row[0]}]\nContent: {row[1]}")
                    policy_count += 1
            logger.debug("Retrieved %d policy chunks above confidence threshold", policy_count)
        except Exception as e:
            logger.warning("Policy search failed: %s", e)

    # Query legislation index
    if routing.get("query_legislation", False):
        try:
            law_results = tool_search_gdpr_legislation(query_text=query_to_search, top_k=3)
            law_rows = law_results.get('result', {}).get('data_array', [])
            law_count = 0
            for row in law_rows:
                if row[-1] > 0.35:
                    retrieved_contexts.append(f"[SOURCE: GDPR Legislation | Article: {row[0]}]\nContent: {row[1]}")
                    law_count += 1
            logger.debug("Retrieved %d legislation chunks above confidence threshold", law_count)
        except Exception as e:
            logger.warning("Legislation search failed: %s", e)

    # Query fines index
    if routing.get("query_fines", False):
        try:
            fine_results = tool_search_historical_fines(query_text=query_to_search, top_k=3)
            fine_rows = fine_results.get('result', {}).get('data_array', [])
            fine_count = 0
            for row in fine_rows:
                if row[-1] > 0.35:
                    retrieved_contexts.append(f"[SOURCE: Enforcement History & Fines Precedent]\nContent: {row[1]}")
                    fine_count += 1
            logger.debug("Retrieved %d enforcement chunks above confidence threshold", fine_count)
        except Exception as e:
            logger.warning("Enforcement search failed: %s", e)

    # Combine all retrieved contexts
    combined_text = "\n\n---\n\n".join(retrieved_contexts)

    if not combined_text.strip():
        logger.warning("Search yielded zero results above confidence baseline (0.35)")
    else:
        logger.info("Aggregated %d cross-reference chunks for grading", len(retrieved_contexts))
    
    queried = []
    if routing.get("query_policy", False):
        queried.append("policy")
    if routing.get("query_legislation", False):
        queried.append("legislation")
    if routing.get("query_fines", False):
        queried.append("fines")


    return {
        "retrieved_context": combined_text,
        "retrieval_loop_count": state["retrieval_loop_count"] + 1,
        "sources_queried": queried
    }


# ============================================================================
# LANGGRAPH NODE: Generate Answer
# ============================================================================
@mlflow.trace(name="generate_answer", span_type="LLM")
def node_generate_answer(state: AgentState) -> Dict[str, Any]:
    """
    Generate compliance answer from validated context.
    
    Args:
        state: Current agent state with retrieved context and question
        
    Returns:
        Dictionary with generated_answer and updated generation_loop_count
        
    Raises:
        Exception: If OpenAI API call fails (re-raised after logging)
    """
    logger.info("Synthesizing answer from verified context")
    logger.debug("Context length: %d characters", len(state["retrieved_context"]))
    
    prompt = f"""You are an elite GDPR compliance expert. Answer the user's question accurately using ONLY the provided context. Do not assume facts, extrapolate, or use outside knowledge.

    CRITICAL GROUNDING RULES:
    1. Distinguish between Entities: Ensure you accurately identify who is being discussed. For example, if a case involves an accommodation/hotel using Booking.com, do not misattribute the violation or actions to Booking.com itself.
    2. Zero Knowledge Architecture: Do not state that a company was fined or found in violation of a specific article unless that specific legal conclusion is explicitly written in the text.
    3. Internal vs. Statutory: Clearly distinguish between a specific company's internal policy constraints (e.g., internal retention periods) and universal GDPR statutory mandates.

    IMPORTANT CITATION REQUIREMENTS (IF AVAILABLE IN CONTEXT):
    - For internal policies: Include specific retention periods, cooling-off windows, legal bases, and which data can/cannot be deleted.
    - For historical fines/enforcement: ONLY if explicitly stated in the context, cite:
      * Company/Entity name exactly as written (e.g., do not mistake a platform for the actual defendant)
      * Fine amount in EUR 
      * Year or date of enforcement action
      * Specific violation cited (e.g., "Article 5.1(f)")
      * Source document name if available

    STRICT CONSTRAINT: When discussing penalties, if specific company names, dates, or fine amounts are not explicitly detailed in the provided context, DO NOT invent them or use outside historical knowledge. Instead, state exactly what the context provides regarding the potential risks or ongoing proceedings.

    Validated Context:
    {state["retrieved_context"]}

    Question: {state["original_question"]}

    Answer:"""

    try:
        logger.debug("Calling OpenAI API for answer generation")
        response = config.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        logger.info("Answer generation completed successfully")
    except Exception as e:
        logger.exception("Failed to generate answer via OpenAI API: %s", e)
        raise
    
    generated_answer = response.choices[0].message.content.strip()
    logger.debug("Generated answer length: %d characters", len(generated_answer))
    
    return {
        "generated_answer": generated_answer,
        "generation_loop_count": state.get("generation_loop_count", 0) + 1
    }


@mlflow.trace(name="regenerate_answer_strict", span_type="LLM")
def node_regenerate_strict(state: AgentState) -> Dict[str, Any]:
    """
    Regenerate answer with stricter citation requirements when groundedness fails.
    
    Args:
        state: Current agent state with context and question
        
    Returns:
        Dictionary with regenerated answer and updated generation_loop_count
        
    Raises:
        Exception: If OpenAI API call fails (re-raised after logging)
    """
    logger.warning("Regenerating answer with stricter grounding requirements")
    logger.debug("Generation loop count: %d", state.get("generation_loop_count", 0))
    
    prompt = f"""You are an elite GDPR compliance expert. Answer the question using ONLY the provided context.

    CRITICAL: You FAILED the groundedness check on your previous attempt. Be EXTREMELY careful about attribution:
    - Only cite companies, amounts, and dates that are EXPLICITLY paired together in the context
    - If a fine amount is mentioned but the company is not clearly stated, say "an unnamed company"
    - If unsure about ANY detail, say "the context does not specify"
    - Double-check every fact against the context before including it

    Validated Context:
    {state["retrieved_context"]}

    Question: {state["original_question"]}

    Answer:"""
    
    try:
        logger.debug("Calling OpenAI API for strict regeneration")
        response = config.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        logger.info("Strict regeneration completed successfully")
    except Exception as e:
        logger.exception("Failed to regenerate answer via OpenAI API: %s", e)
        raise
    
    return {
        "generated_answer": response.choices[0].message.content.strip(),
        "generation_loop_count": state["generation_loop_count"] + 1
    }


# ============================================================================
# LANGGRAPH NODE: Rewrite Query
# ============================================================================
@mlflow.trace(name="rewrite_query", span_type="LLM")
def node_rewrite_query(state: AgentState) -> Dict[str, Any]:
    """
    Reformulate query to optimize search characteristics for vector lookup.
    
    Args:
        state: Current agent state with original question
        
    Returns:
        Dictionary with optimized current_query
        
    Raises:
        Exception: If OpenAI API call fails (re-raised after logging)
    """
    logger.info("Rewriting query to optimize search intent")
    logger.debug("Original question: %s", 
                state["original_question"][:100] + "..." if len(state["original_question"]) > 100 else state["original_question"])
    
    prompt = f"""Analyze the user's question and rewrite it into an optimized, concise keyword search query for a vector database. Focus purely on compliance terminology, core legal topics, or document sections. Do not include conversational text or conversational phrasing.

Original Question: {state["original_question"]}
Optimized Search Query:"""

    try:
        logger.debug("Calling OpenAI API for query rewrite")
        response = config.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        optimized_query = response.choices[0].message.content.strip()
        logger.info("Query rewrite completed successfully")
        logger.debug("Optimized query: %s", optimized_query)
    except Exception as e:
        logger.exception("Failed to rewrite query via OpenAI API: %s", e)
        raise
    
    return {
        "current_query": optimized_query
    }


# ============================================================================
# COMPLETENESS CHECK NODE
# ============================================================================
@mlflow.trace(name="check_completeness", span_type="CHAIN")
def node_check_completeness(state: AgentState) -> Dict[str, Any]:
    """
    Detect if answer is grounded but incomplete.
    
    This happens when we retrieved from the wrong source and the answer contains
    phrases like "context does not specify".
    
    Args:
        state: Current agent state with generated answer and context
        
    Returns:
        Dictionary with is_answer_complete boolean flag
    """
    answer = state.get("generated_answer", "")
    context = state.get("retrieved_context", "")
    
    logger.info("Evaluating answer completeness")
    logger.debug("Answer length: %d words", len(answer.split()))
    logger.debug("Context length: %d characters", len(context))
    
    # Detection patterns for incomplete answers
    incomplete_patterns = [
        "does not specify",
        "unclear from the context",
        "not provided in",
        "no information",
        "context does not",
        "insufficient information",
        "does not contain",
        "not mentioned",
        "no details"
    ]
    
    answer_lower = answer.lower()
    has_incomplete_language = any(pattern in answer_lower for pattern in incomplete_patterns)
    is_short = len(answer.split()) < 20
    weak_context = len(context) < 200
    
    is_incomplete = has_incomplete_language or (is_short and weak_context)
    
    if is_incomplete:
        logger.warning("Answer appears incomplete. Will expand search to all sources.")
        logger.debug("Incomplete indicators: incomplete_language=%s, short=%s, weak_context=%s",
                    has_incomplete_language, is_short, weak_context)
    else:
        logger.info("Answer completeness check passed")
    
    return {
        "is_answer_complete": not is_incomplete
    }


# ============================================================================
# CHECK SOURCE COVERAGE NODE
# ============================================================================
@mlflow.trace(name="check_source_coverage", span_type="CHAIN")
def node_check_source_coverage(state: AgentState) -> Dict[str, Any]:
    """
    Check if question semantically requires real-world examples (fines).
    
    Args:
        state: Current agent state with question and context
        
    Returns:
        Dictionary with needs_example_expansion boolean flag
    """
    question = state["original_question"]
    context = state["retrieved_context"]
    
    logger.debug("Checking source coverage for semantic requirements")
    
    # If question is about "handling situations" and context has no real examples
    needs_examples = any(kw in question.lower() for kw in 
                        ["handle", "approach", "deal with", "respond to"])
    has_examples = "SOURCE: Enforcement" in context
    
    if needs_examples and not has_examples:
        logger.info("Question requires real-world examples but none found in context")
        return {"needs_example_expansion": True}
    
    logger.debug("Source coverage is adequate for question type")
    return {"needs_example_expansion": False}


# ============================================================================
# EXPAND RETRIEVAL NODE (All Sources)
# ============================================================================
@mlflow.trace(name="expand_all_sources", span_type="RETRIEVER")
def node_expand_all_sources(state: AgentState) -> Dict[str, Any]:
    """
    Search ALL sources when primary routing didn't yield complete results.
    
    Args:
        state: Current agent state with query and loop count
        
    Returns:
        Dictionary with retrieved_context, updated loop count, and expansion flag
    """
    query_to_search = state["current_query"]
    
    logger.info("Expanding search to all sources due to insufficient primary results")
    logger.debug("Query: %s", query_to_search[:100] + "..." if len(query_to_search) > 100 else query_to_search)
    
    already_queried = state.get("sources_queried", [])
    newly_queried = []
    retrieved_contexts = []

    # Policy search
    if "policy" in already_queried:
        logger.debug("Skipping policy search — already queried in primary retrieval")
    else:
        newly_queried.append("policy")
        try:
            policy_results = tool_search_retail_policy(query_text=query_to_search, top_k=5)
            policy_rows = policy_results.get('result', {}).get('data_array', [])
            policy_count = 0
            for row in policy_rows:
                if row[-1] > 0.35:
                    retrieved_contexts.append(f"[SOURCE: Internal Retail Policy | Section: {row[0]}]\nContent: {row[1]}")
                    policy_count += 1
            logger.debug("Expanded search retrieved %d policy chunks", policy_count)
        except Exception as e:
            logger.warning("Policy search failed during expansion: %s", e)

    # Legislation search
    if "legislation" in already_queried:
        logger.debug("Skipping legislation search — already queried in primary retrieval")
    else:
        newly_queried.append("legislation")
        try:
            law_results = tool_search_gdpr_legislation(query_text=query_to_search, top_k=5)
            law_rows = law_results.get('result', {}).get('data_array', [])
            law_count = 0
            for row in law_rows:
                if row[-1] > 0.35:
                    retrieved_contexts.append(f"[SOURCE: GDPR Legislation | Article: {row[0]}]\nContent: {row[1]}")
                    law_count += 1
            logger.debug("Expanded search retrieved %d legislation chunks", law_count)
        except Exception as e:
            logger.warning("Legislation search failed during expansion: %s", e)

    # Fines search
    if "fines" in already_queried:
        logger.debug("Skipping fines search — already queried in primary retrieval")
    else:
        newly_queried.append("fines")
        try:
            fine_results = tool_search_historical_fines(query_text=query_to_search, top_k=5)
            fine_rows = fine_results.get('result', {}).get('data_array', [])
            fine_count = 0
            for row in fine_rows:
                if row[-1] > 0.35:
                    retrieved_contexts.append(f"[SOURCE: Enforcement History & Fines Precedent]\nContent: {row[1]}")
                    fine_count += 1
            logger.debug("Expanded search retrieved %d enforcement chunks", fine_count)
        except Exception as e:
            logger.warning("Enforcement search failed during expansion: %s", e)
    
    new_text = "\n\n---\n\n".join(retrieved_contexts)
    existing_context = state.get("retrieved_context", "")

    if new_text.strip():
        logger.info("Expanded search retrieved %d total chunks from new sources", len(retrieved_contexts))
        combined_text = (existing_context + "\n\n---\n\n" + new_text).strip() if existing_context.strip() else new_text
    else:
        logger.warning("Expanded search yielded no results above confidence threshold — keeping existing context")
        combined_text = existing_context

    return {
        "retrieved_context": combined_text,
        "sources_queried": already_queried + newly_queried,
        "retrieval_loop_count": state["retrieval_loop_count"] + 1,
        "expanded_search_used": True
    }


# ============================================================================
# VERIFY OUTPUT NODE (Passthrough — routing logic lives in edge_verify_output)
# ============================================================================
def node_verify_output(state: AgentState) -> Dict[str, Any]:
    logger.info("Entering output verification")
    return {}


# ============================================================================
# FALLBACK NODE
# ============================================================================
def node_return_fallback(state: AgentState) -> Dict[str, Any]:
    """
    Add groundedness warning to answer when verification fails.
    
    Args:
        state: Current agent state with generated answer
        
    Returns:
        Dictionary with warning-prefixed generated_answer
    """
    logger.warning("Returning fallback answer with groundedness warning")
    
    warning = """**GROUNDEDNESS WARNING**: This answer may contain information not fully supported by the retrieved documents. The system attempted multiple times to generate a fully grounded response but was unable to do so. Please verify critical details with a GDPR compliance expert before taking action.

    ---

    """
    
    return {
        "generated_answer": warning + state["generated_answer"]
    }
