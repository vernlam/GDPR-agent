"""
LLM-as-a-judge implementation for quality evaluation.
"""
from openai import OpenAI
import json
from typing import Dict
from monitoring.config import config

class LLMJudge:
    """Evaluate agent responses using GPT-4 as a judge"""
    
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
        self.model = config.LLM_JUDGE_MODEL
    
    def evaluate(self, question: str, answer: str, context: str = "") -> Dict[str, float]:
        """
        Evaluate answer quality on multiple dimensions.
        
        Returns:
            Dict with scores (1-5) for relevance, accuracy, completeness, 
            citation, clarity, overall, and explanation
        """
        
        prompt = self._build_evaluation_prompt(question, answer, context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Validate result has required fields
            required_fields = ['relevance', 'accuracy', 'completeness', 
                             'citation', 'clarity', 'overall', 'explanation']
            if not all(field in result for field in required_fields):
                return self._get_error_result("Missing required fields in response")
            
            return result
            
        except Exception as e:
            return self._get_error_result(str(e))
    
    def _build_evaluation_prompt(self, question: str, answer: str, context: str) -> str:
        """Build the evaluation prompt"""
        return f"""You are evaluating a GDPR compliance chatbot's response quality.

Question: {question}

Answer: {answer}

Context Retrieved: {context[:500] if context else "Not provided"}

Rate the answer on these criteria (1-5 scale, where 5 is best):

1. **Relevance** (1-5): Does the answer directly address the question?
2. **Accuracy** (1-5): Is the information factually correct for GDPR?
3. **Completeness** (1-5): Does it cover all important aspects?
4. **Citation** (1-5): Does it reference specific GDPR articles/sources?
5. **Clarity** (1-5): Is it clear and easy to understand?

Also provide:
- Overall score (1-5)
- Brief explanation (1 sentence)

Return ONLY valid JSON in this exact format:
{{
  "relevance": <score>,
  "accuracy": <score>,
  "completeness": <score>,
  "citation": <score>,
  "clarity": <score>,
  "overall": <score>,
  "explanation": "<brief explanation>"
}}"""
    
    def _get_error_result(self, error_message: str) -> Dict[str, float]:
        """Return error result structure"""
        return {
            "relevance": 0,
            "accuracy": 0,
            "completeness": 0,
            "citation": 0,
            "clarity": 0,
            "overall": 0,
            "explanation": f"Error: {error_message}"
        }