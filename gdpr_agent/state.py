"""
LangGraph state schema for GDPR Agent workflow.
Defines the shared state structure passed between all graph nodes.
"""

from typing import TypedDict, List


class AgentState(TypedDict):
    """
    Shared state for the GDPR Agent LangGraph workflow.
    
    This TypedDict defines all fields that flow through the agent's graph nodes.
    Each node can read any field and update specific fields by returning a dict
    with the fields to update.
    
    Fields:
        original_question: The user's original compliance question (immutable).
        current_query: The current search query, potentially rewritten for optimization.
        sources_queried: The sources that the agent has already queried in.
        retrieved_context: Combined text from all retrieved document chunks with source tags.
        generated_answer: The agent's compliance answer generated from context.
        loop_count: Total number of graph iterations (safeguard against infinite loops).
        retrieval_loop_count: Number of retrieval attempts made (tracks re-retrieval cycles).
        generation_loop_count: Number of answer generation attempts (tracks regeneration cycles).
        is_answer_complete: Whether the generated answer fully addresses the question.
        expanded_search_used: Whether all-source search expansion was triggered.
        needs_example_expansion: Whether question semantics require real-world examples.
    """
    
    # Core question and query
    original_question: str
    current_query: str
    
    # Retrieved content
    sources_queried: List[str]
    retrieved_context: str
    
    # Generated output
    generated_answer: str
    
    # Loop counters (safeguards)
    loop_count: int
    retrieval_loop_count: int
    generation_loop_count: int
    
    # Quality control flags
    is_answer_complete: bool
    expanded_search_used: bool
    needs_example_expansion: bool
