from typing import TypedDict
from langgraph.graph import StateGraph, END

# 1. Define your Graph's State
class AgentState(TypedDict):
    original_question: str
    current_query: str
    retrieved_context: str
    generated_answer: str
    loop_count: int
    retrieval_loop_count: int
    generation_loop_count: int
    is_answer_complete: bool
    expanded_search_used: bool
    needs_example_expansion: bool
