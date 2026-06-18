from .state import AgentState
from .graders import grade_retrieved_context, grade_answer_groundedness

# ============================================================================
# LANGGRAPH CONDITIONAL EDGES (The Crossroads)
# ============================================================================

def edge_evaluate_context(state: AgentState) -> str:
    """
    Evaluates the context quality. Directs to generation or rewrite loop.
    """
    # Enforce loop safety limit right at the gate
    if state["retrieval_loop_count"] >= 3:
        print("🛑 Max search attempts reached. Forced transition to final answer.")
        return "generate_response"
        
    context_is_valid = grade_retrieved_context(
        user_question=state["original_question"], 
        retrieved_context=state["retrieved_context"]
    )
    
    if context_is_valid:
        return "generate_response"
    else:
        return "rewrite_query"


def edge_verify_output(state: AgentState) -> str:
    """
    Evaluates hallucination risks. If groundedness fails, regenerate instead of re-retrieving.
    """
    if state["generation_loop_count"] >= 3:
        print("🛑 Max loops reached during verification. Ending loop to prevent infinite run.")
        return "end"
        
    answer_is_safe = grade_answer_groundedness(
        generated_answer=state["generated_answer"], 
        retrieved_context=state["retrieved_context"]
    )
    
    if answer_is_safe:
        return "end"  # This tells LangGraph the process is successfully complete
    else:
        print("⚠️  Groundedness failed - regenerating with stricter prompt")
        return "regenerate_strict"
