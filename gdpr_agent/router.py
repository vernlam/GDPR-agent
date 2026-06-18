from . import config

def route_query(question: str) -> dict:
    """
    Analyses the question and determines which data sources to query.
    Returns a dict with boolean flags for each source.
    """
    
    prompt = f"""Analyse this GDPR compliance question and determine which data sources are needed.

    Question: {question}

    Return JSON with boolean flags:
    - "query_fines": true if question asks about enforcement examples, penalties, fines, or specific companies
    - "query_legislation": true if question asks about legal requirements, articles, regulations, or rights
    - "query_policy": true if question asks about internal procedures, retention periods, or operational guidelines

    Examples:
    - "What fines have companies received?" → {{"query_fines": true, "query_legislation": false, "query_policy": false}}
    - "What does Article 17 say?" → {{"query_fines": false, "query_legislation": true, "query_policy": false}}
    - "How long do we retain customer data?" → {{"query_fines": false, "query_legislation": false, "query_policy": true}}
    - "Right to be forgotten requirements and penalties" → {{"query_fines": true, "query_legislation": true, "query_policy": true}}

    JSON Output:"""

    response = config.openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        seed = 42
    )
    
    import json
    return json.loads(response.choices[0].message.content)