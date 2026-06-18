"""LangGraph workflow builder"""
from langgraph.graph import StateGraph, END
from .state import AgentState
from .nodes import node_route_and_retrieve, node_generate_answer, node_rewrite_query, node_regenerate_strict, node_return_fallback
from .edges import edge_evaluate_context, edge_verify_output

def build_agent():
    """Build and compile the GDPR compliance agent graph"""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve_docs", node_route_and_retrieve)
    workflow.add_node("generate_response", node_generate_answer)
    workflow.add_node("rewrite_query", node_rewrite_query)
    workflow.add_node("regenerate_strict",node_regenerate_strict)
    workflow.add_node("return_fallback",node_return_fallback)
    
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
    
    workflow.add_conditional_edges(
        "generate_response",
        edge_verify_output,
        {
            "end" : END,
            "regenerate_strict": "regenerate_strict",
            "return_fallback": "return_fallback"
        }
    )

    workflow.add_conditional_edges(
        "regenerate_strict",
        edge_verify_output,
        {
            "end" : END,
            "regenerate_strict": "regenerate_strict"
        }
    )

    workflow.add_edge("return_fallback", END)
    
    # Rewrite loops back to retrieval
    workflow.add_edge("rewrite_query", "retrieve_docs")
    
    return workflow.compile()