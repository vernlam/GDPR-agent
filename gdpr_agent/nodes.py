
from .state import AgentState
from .tools import tool_search_retail_policy, tool_search_gdpr_legislation, tool_search_historical_fines
from . import config
from .router import route_query
import mlflow


# ============================================================================
# RETRIEVAL NODE: Cross-Index Retrieval
# ============================================================================
def node_route_and_retrieve(state: AgentState) -> dict:
    """
    Routes query to appropriate indices based on question type.
    """
    query_to_search = state["current_query"]
    
    print(f"🧠 [Node: Retrieve] Analyzing query routing...")
    
    # Determine which sources to query
    routing = route_query(state["original_question"])
    print(f"📍 Routing: Fines={routing.get('query_fines')}, "
          f"Legislation={routing.get('query_legislation')}, "
          f"Policy={routing.get('query_policy')}")
    
    retrieved_contexts = []
    
    if routing.get("query_policy",False):
        try:
            policy_results = tool_search_retail_policy(query_text=query_to_search, top_k=3)
            policy_rows = policy_results.get('result', {}).get('data_array', [])
            for row in policy_rows:
                if row[-1] > 0.35:  
                    retrieved_contexts.append(f"[SOURCE: Internal Retail Policy | Section: {row[0]}]\nContent: {row[1]}")
        except Exception as e:
            print(f"⚠️ Policy search warning: {e}")

    if routing.get("query_legislation",False):
        try:
            law_results = tool_search_gdpr_legislation(query_text=query_to_search, top_k=3)
            law_rows = law_results.get('result', {}).get('data_array', [])
            for row in law_rows:
                if row[-1] > 0.35:
                    retrieved_contexts.append(f"[SOURCE: GDPR Legislation | Article: {row[0]}]\nContent: {row[1]}")
        except Exception as e:
            print(f"⚠️ Legislation search warning: {e}")

    if routing.get("query_fines", False):
        try:
            fine_results = tool_search_historical_fines(query_text=query_to_search, top_k=3)
            fine_rows = fine_results.get('result', {}).get('data_array', [])
            for row in fine_rows:
                if row[-1] > 0.35:
                    retrieved_contexts.append(f"[SOURCE: Enforcement History & Fines Precedent]\nContent: {row[1]}")
        except Exception as e:
            print(f"⚠️ Enforcement search warning: {e}")

    # Combine everything gathered across your entire compliance ecosystem
    combined_text = "\n\n---\n\n".join(retrieved_contexts)
    
    if not combined_text.strip():
        print("⚠️ Parallel search yielded zero results above confidence baseline.")
    else:
        print(f"✅ Aggregated {len(retrieved_contexts)} cross-reference chunks for grading.")

    return {
        "retrieved_context": combined_text,
        "retrieval_loop_count": state["retrieval_loop_count"] + 1
    }

# ============================================================================
# LANGGRAPH NODE: Generate Answer
# ============================================================================
@mlflow.trace(name="generate_answer",span_type="LLM")
def node_generate_answer(state: AgentState) -> dict:
    """
    LangGraph Node: Takes the validated context from the state 
    and synthesizes the final compliance response.
    """
    print("🤖 [Node: Generate] Synthesizing final answer from verified context...")
    
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

    response = config.openai_client.chat.completions.create(  
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    return {
        "generated_answer": response.choices[0].message.content.strip(),
        "generation_loop_count": state.get("generation_loop_count", 0) + 1
    }
    
@mlflow.trace(name="regenerate_answer_strict",span_type="LLM")
def node_regenerate_strict(state: AgentState) -> dict:
    """
    Regenerates answer with stricter citation requirements when groundedness fails.
    Uses the SAME context but with a more careful prompt.
    """
    print("🔄 [Node: Regenerate] Creating stricter answer from existing context...")
    
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
    
    response = config.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0 
    )
    
    return {
        "generated_answer": response.choices[0].message.content.strip(),
        "generation_loop_count": state["generation_loop_count"] + 1
    }

# ============================================================================
# LANGGRAPH NODE: Rewrite Query
# ============================================================================
mlflow.trace(name="rewrite_query",span_type="LLM")
def node_rewrite_query(state: AgentState) -> dict:
    """
    LangGraph Node: If a check fails, reformulates the current query 
    to optimize its search characteristics for the next vector lookup.
    """
    print("🔄 [Node: Rewrite] Optimizing search query intent...")
    
    prompt = f"""Analyze the user's question and rewrite it into an optimized, concise keyword search query for a vector database. Focus purely on compliance terminology, core legal topics, or document sections. Do not include conversational text or conversational phrasing.

Original Question: {state["original_question"]}
Optimized Search Query:"""

    response = config.openai_client.chat.completions.create( 
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    return {
        "current_query": response.choices[0].message.content.strip()
    }


# ============================================================================
# COMPLETENESS CHECK NODE
# ============================================================================
@mlflow.trace(name="check_completeness", span_type="CHAIN")
def node_check_completeness(state: AgentState) -> dict:
    """
    Detect if answer is grounded but incomplete (e.g., 'context does not specify').
    This happens when we retrieved from the wrong source.
    """
    answer = state.get("generated_answer", "")
    context = state.get("retrieved_context", "")
    
    print("🔍 [Node: Check Completeness] Evaluating answer completeness...")
    
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
        print("⚠️  Answer appears incomplete. Will expand search to all sources.")
    else:
        print("✅ Answer appears complete.")
    
    return {
        "is_answer_complete": not is_incomplete
    }


# ============================================================================
# EXPAND RETRIEVAL NODE (All Sources)
# ============================================================================
@mlflow.trace(name="expand_all_sources", span_type="RETRIEVER")
def node_expand_all_sources(state: AgentState) -> dict:
    """
    When primary routing didn't yield complete results, search ALL sources.
    """
    query_to_search = state["current_query"]
    
    print("🔄 [Node: Expand] Primary source insufficient. Searching ALL sources...")
    
    retrieved_contexts = []
    
    # Search ALL 3 sources regardless of routing
    try:
        policy_results = tool_search_retail_policy(query_text=query_to_search, top_k=5)
        policy_rows = policy_results.get('result', {}).get('data_array', [])
        for row in policy_rows:
            if row[-1] > 0.35:
                retrieved_contexts.append(f"[SOURCE: Internal Retail Policy | Section: {row[0]}]\nContent: {row[1]}")
        print(f"  ✓ Retrieved {len([r for r in policy_rows if r[-1] > 0.35])} policy chunks")
    except Exception as e:
        print(f"⚠️ Policy search warning: {e}")
    
    try:
        law_results = tool_search_gdpr_legislation(query_text=query_to_search, top_k=5)
        law_rows = law_results.get('result', {}).get('data_array', [])
        for row in law_rows:
            if row[-1] > 0.35:
                retrieved_contexts.append(f"[SOURCE: GDPR Legislation | Article: {row[0]}]\nContent: {row[1]}")
        print(f"  ✓ Retrieved {len([r for r in law_rows if r[-1] > 0.35])} legislation chunks")
    except Exception as e:
        print(f"⚠️ Legislation search warning: {e}")
    
    try:
        fine_results = tool_search_historical_fines(query_text=query_to_search, top_k=5)
        fine_rows = fine_results.get('result', {}).get('data_array', [])
        for row in fine_rows:
            if row[-1] > 0.35:
                retrieved_contexts.append(f"[SOURCE: Enforcement History & Fines Precedent]\nContent: {row[1]}")
        print(f"  ✓ Retrieved {len([r for r in fine_rows if r[-1] > 0.35])} fines chunks")
    except Exception as e:
        print(f"⚠️ Enforcement search warning: {e}")
    
    combined_text = "\n\n---\n\n".join(retrieved_contexts)
    
    if not combined_text.strip():
        print("⚠️ Even expanded search yielded no results.")
    else:
        print(f"✅ Expanded search retrieved {len(retrieved_contexts)} total chunks from all sources.")
    
    return {
        "retrieved_context": combined_text,
        "retrieval_loop_count": state["retrieval_loop_count"] + 1,
        "expanded_search_used": True
    }

# ============================================================================
# FALLBACK NODE
# ============================================================================

def node_return_fallback(state: AgentState) -> dict:
    """Add warning to last answer when groundedness fails"""
    warning = """⚠️ **GROUNDEDNESS WARNING**: This answer may contain information not fully supported by the retrieved documents. The system attempted multiple times to generate a fully grounded response but was unable to do so. Please verify critical details with a GDPR compliance expert before taking action.

    ---

    """
    
    return {
        "generated_answer": warning + state["generated_answer"]
    }