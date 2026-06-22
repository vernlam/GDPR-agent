from pydantic import BaseModel, Field
import json
from . import config

class RelevanceGrade(BaseModel):
    is_relevant: bool

class GroundednessGrade(BaseModel):
    is_grounded: bool
    reason: str = Field(default='')


# ============================================================================
# UTILITY HELPER 1: Retrieval Grader
# ============================================================================
def grade_retrieved_context(user_question: str, retrieved_context: str) -> bool:
    """
    Evaluates whether the retrieved context is relevant and contains useful info
    to help answer the user's question.
    """
    if not retrieved_context.strip():
        return False
        
    print("🧐 Grading retrieved context relevance...")
    
    prompt = f"""You are a quality control auditor. Evaluate if the provided retrieved context contains background information, direct evidence, or context that is relevant to answering the user's question.

    Retrieved Context:
    {retrieved_context}

    User Question: {user_question}

    Provide your assessment in a strict JSON format with a single boolean key 'is_relevant' (true or false).
    JSON Output:"""

    response = config.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        seed=42
    )
    
    try:
        result = RelevanceGrade(**json.loads(response.choices[0].message.content))
        return result.is_relevant
    except Exception as e:
        print(f"⚠️ Grader response validation error: {e}")
        return False

# ============================================================================
# UTILITY HELPER 2: Hallucination & Groundedness Grader
# ============================================================================
def grade_answer_groundedness(generated_answer: str, retrieved_context: str) -> bool:
    """
    Verifies that the generated answer is strictly supported by the context, 
    preventing hallucinations or outside assumptions.
    """
    print("⚖️ Verifying answer groundedness against source documents...")

    print(f"\n📄 CONTEXT LENGTH: {len(retrieved_context)} chars")
    print(f"📝 ANSWER LENGTH: {len(generated_answer)} chars")
    
    prompt = f"""You are an expert compliance risk assessor. Compare the generated answer against the approved retrieved context. Determine if the generated answer contains ANY claims, assumptions, or facts that are NOT explicitly stated or directly backed by the retrieved context.

    Retrieved Context:
    {retrieved_context}

    Generated Answer:
    {generated_answer}

    Determine if the generated answer is substantially grounded in the context. Minor reasonable inferences for readability are acceptable.

    Provide your assessment in JSON format with:
    - "is_grounded": boolean (true if substantially supported by context, false if it contains major unsupported claims)
    - "reason": string (brief explanation of your decision)
    JSON Output:"""

    response = config.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        seed=42
    )

    try:
        result = GroundednessGrade(**json.loads(response.choices[0].message.content))
        print(f"🔍 GRADER REASONING: {result.reason}")
        return result.is_grounded
    except Exception as e:
        print(f"⚠️ Groundedness validation error: {e}")
        return False
    