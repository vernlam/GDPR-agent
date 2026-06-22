
from .state import AgentState
from .tools import tool_search_retail_policy, tool_search_gdpr_legislation, tool_search_historical_fines
from . import config
from .router import route_query


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
                # row[-1] is similarity score, row[0] is title, row[1] is text
                if row[-1] > 0.35:  # Soft math filter to eliminate pure noise
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
def node_generate_answer(state: AgentState) -> dict:
    """
    LangGraph Node: Takes the validated context from the state 
    and synthesizes the final compliance response.
    """
    print("🤖 [Node: Generate] Synthesizing final answer from verified context...")
    
    prompt = f"""You are an elite GDPR compliance expert. Answer the user's question accurately using the provided context.

    IMPORTANT CITATION REQUIREMENTS:
    - For internal policies: Include specific retention periods, cooling-off windows, legal bases, and which data can/cannot be deleted
    - For historical fines/enforcement: ALWAYS cite:
    * Company name (e.g., "Google Ireland", "British Airways")
    * Fine amount in EUR (e.g., "€50,000,000")
    * Year or date of enforcement action
    * Specific violation cited (e.g., "Article 17(1)(c)")
    * Source document name if available

    When discussing penalties, provide concrete examples with company names and amounts. Do not use generic phrases like "companies have faced fines" - name the specific companies and amounts from the context.

    Validated Context:
    {state["retrieved_context"]}

    Question: {state["original_question"]}

    Answer:"""

    response = config.openai_client.chat.completions.create(  # ✅ Using your initialized openai_client
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    return {
        "generated_answer": response.choices[0].message.content.strip(),
        "generation_loop_count": state.get("generation_loop_count", 0) + 1
    }
    

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
        temperature=0.0  # ✅ Lower temperature for more careful generation
    )
    
    return {
        "generated_answer": response.choices[0].message.content.strip(),
        "generation_loop_count": state["generation_loop_count"] + 1
    }

# ============================================================================
# LANGGRAPH NODE: Rewrite Query
# ============================================================================
def node_rewrite_query(state: AgentState) -> dict:
    """
    LangGraph Node: If a check fails, reformulates the current query 
    to optimize its search characteristics for the next vector lookup.
    """
    print("🔄 [Node: Rewrite] Optimizing search query intent...")
    
    prompt = f"""Analyze the user's question and rewrite it into an optimized, concise keyword search query for a vector database. Focus purely on compliance terminology, core legal topics, or document sections. Do not include conversational text or conversational phrasing.

Original Question: {state["original_question"]}
Optimized Search Query:"""

    response = config.openai_client.chat.completions.create(  # ✅ Using your initialized openai_client
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    
    return {
        "current_query": response.choices[0].message.content.strip()
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