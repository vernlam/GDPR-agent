def evaluate_case(test_case: dict, agent_response: dict) -> dict:
    """Score agent response against expected behavior"""
    expected = test_case.get("expected_behavior", {})
    answer = agent_response.get("answer", "")
    context = agent_response.get("context", "")
    
    scores = {"source_correct": 0, "content_match": 0, "total": 0}
    feedback = []
    
    # Source checking logic
    expected_sources = expected.get("sources", [])
    source_found = any(src.lower() in context.lower() for src in expected_sources)
    
    if source_found:
        scores["source_correct"] = 1
        feedback.append(f"✅ Retrieved from expected sources: {expected_sources}")
    else:
        feedback.append(f"❌ Expected sources {expected_sources} not found")
    
    # Content matching logic
    must_include = expected.get("must_retrieve_from_articles", []) + expected.get("must_include_in_answer", [])
    
    if must_include:
        found = [item for item in must_include if item.lower() in (answer + context).lower()]
        coverage = len(found) / len(must_include)
        scores["content_match"] = coverage
        
        if coverage >= 0.7:
            feedback.append(f"✅ Found {len(found)}/{len(must_include)} expected items")
        else:
            missing = set(must_include) - set(found)
            feedback.append(f"❌ Missing: {missing}")
    else:
        scores["content_match"] = 1
    
    scores["total"] = (scores["source_correct"] + scores["content_match"]) / 2
    
    return {
        "scores": scores,
        "feedback": feedback,
        "passed": scores["total"] >= 0.7
    }