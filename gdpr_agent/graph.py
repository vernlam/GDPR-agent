"""
LangGraph workflow builder for GDPR Agent.
Constructs the agentic graph with nodes, edges, and conditional routing logic.
"""

import logging
from langgraph.graph import StateGraph, END, CompiledGraph
from .state import AgentState
from .nodes import (
    node_route_and_retrieve,
    node_generate_answer,
    node_rewrite_query,
    node_regenerate_strict,
    node_return_fallback,
    node_check_source_coverage,
    node_check_completeness,
    node_expand_all_sources
)
from .edges import (
    edge_evaluate_context,
    edge_verify_output,
    edge_route_after_completeness,
    edge_route_after_source_check
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


def build_agent() -> CompiledGraph:
    """
    Build and compile the GDPR compliance agent graph.
    
    The graph consists of:
    - Retrieval and generation nodes
    - Context evaluation and quality control nodes
    - Conditional routing based on relevance, completeness, and groundedness
    - Loop prevention mechanisms with fallback handling
    
    Returns:
        Compiled LangGraph workflow ready for invocation
        
    Raises:
        Exception: If graph compilation fails
    """
    logger.info("Building GDPR Agent workflow graph")
    
    try:
        workflow = StateGraph(AgentState)
        logger.debug("StateGraph initialized with AgentState")
    except Exception as e:
        logger.exception("Failed to initialize StateGraph: %s", e)
        raise
    
    # Add nodes
    logger.debug("Adding nodes to workflow graph")
    workflow.add_node("retrieve_docs", node_route_and_retrieve)
    workflow.add_node("generate_response", node_generate_answer)
    workflow.add_node("check_source_coverage", node_check_source_coverage)
    workflow.add_node("rewrite_query", node_rewrite_query)
    workflow.add_node("regenerate_strict", node_regenerate_strict)
    workflow.add_node("return_fallback", node_return_fallback)
    workflow.add_node("check_completeness", node_check_completeness)
    workflow.add_node("expand_all_sources", node_expand_all_sources)
    logger.debug("Added 8 nodes to workflow graph")
    
    # Set entry point
    workflow.set_entry_point("retrieve_docs")
    logger.debug("Entry point set to 'retrieve_docs'")
    
    # Add conditional edges
    logger.debug("Adding conditional edges and routing logic")
    
    # Retrieval evaluation: relevant context -> generation, irrelevant -> rewrite
    workflow.add_conditional_edges(
        "retrieve_docs",
        edge_evaluate_context,
        {
            "generate_response": "generate_response",
            "rewrite_query": "rewrite_query"
        }
    )
    logger.debug("Added conditional edge: retrieve_docs -> [generate_response | rewrite_query]")

    # After generation, check source coverage
    workflow.add_edge("generate_response", "check_source_coverage")
    logger.debug("Added edge: generate_response -> check_source_coverage")

    # Source coverage check: needs examples -> expand, adequate -> completeness check
    workflow.add_conditional_edges(
        "check_source_coverage",
        edge_route_after_source_check,
        {
            "expand_all_sources": "expand_all_sources",
            "check_completeness": "check_completeness"
        }
    )
    logger.debug("Added conditional edge: check_source_coverage -> [expand_all_sources | check_completeness]")
        
    # After expanding sources, regenerate response
    workflow.add_edge("expand_all_sources", "generate_response")
    logger.debug("Added edge: expand_all_sources -> generate_response")

    # Completeness check: incomplete -> expand, complete -> verify
    workflow.add_conditional_edges(
        "check_completeness",
        edge_route_after_completeness,
        {
            "expand_all_sources": "expand_all_sources",
            "regenerate_strict": "regenerate_strict"
        }
    )
    logger.debug("Added conditional edge: check_completeness -> [expand_all_sources | regenerate_strict]")

    # Output verification: grounded -> end, hallucination -> regenerate or fallback
    workflow.add_conditional_edges(
        "regenerate_strict",
        edge_verify_output,
        {
            "end": END,
            "regenerate_strict": "regenerate_strict",
            "return_fallback": "return_fallback"
        }
    )
    logger.debug("Added conditional edge: regenerate_strict -> [end | regenerate_strict | return_fallback]")

    # Fallback terminates the workflow
    workflow.add_edge("return_fallback", END)
    logger.debug("Added edge: return_fallback -> END")
    
    # Rewrite loops back to retrieval
    workflow.add_edge("rewrite_query", "retrieve_docs")
    logger.debug("Added edge: rewrite_query -> retrieve_docs")
    
    # Compile the workflow
    logger.info("Compiling workflow graph")
    try:
        compiled_graph = workflow.compile()
        logger.info("GDPR Agent workflow graph compiled successfully")
        return compiled_graph
    except Exception as e:
        logger.exception("Failed to compile workflow graph: %s", e)
        raise
