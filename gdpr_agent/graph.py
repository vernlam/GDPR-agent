"""LangGraph workflow builder"""
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import node_route_and_retrieve, node_generate_answer, node_rewrite_query, node_regenerate_strict, node_return_fallback, node_check_source_coverage, node_check_completeness, node_expand_all_sources
from .edges import edge_evaluate_context, edge_verify_output, edge_route_after_completeness, edge_route_after_source_check

def build_agent():
    """Build and compile the GDPR compliance agent graph"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve_docs", node_route_and_retrieve)
    workflow.add_node("generate_response", node_generate_answer)
    workflow.add_node("check_source_coverage", node_check_source_coverage)
    workflow.add_node("rewrite_query", node_rewrite_query)
    workflow.add_node("regenerate_strict",node_regenerate_strict)
    workflow.add_node("return_fallback",node_return_fallback)
    workflow.add_node("check_completeness",node_check_completeness)
    workflow.add_node("expand_all_sources",node_expand_all_sources)
    
    # Set entry point
    workflow.set_entry_point("retrieve_docs")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "retrieve_docs",
        edge_evaluate_context,
        {
            "generate_response": "generate_response",
            "rewrite_query": "rewrite_query"
        }
    )

    workflow.add_edge("generate_response","check_source_coverage")

    workflow.add_conditional_edges(
        "check_source_coverage",
        edge_route_after_source_check,
        {
            "expand_all_sources":"expand_all_sources",
            "check_completeness":"check_completeness"
        }
    )
        
    workflow.add_edge("expand_all_sources","generate_response")

    workflow.add_conditional_edges(
    "check_completeness",
    edge_route_after_completeness,
        {
            "expand_all_sources":"expand_all_sources",
            "regenerate_strict":"regenerate_strict"
        }
    )

    workflow.add_conditional_edges(
        "regenerate_strict",
        edge_verify_output,
        {
            "end" : END,
            "regenerate_strict": "regenerate_strict",
            "return_fallback": "return_fallback"
        }
    )

    workflow.add_edge("return_fallback", END)
    
    # Rewrite loops back to retrieval
    workflow.add_edge("rewrite_query", "retrieve_docs")
    
    return workflow.compile()