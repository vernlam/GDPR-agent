"""
LLM-as-a-judge implementation for quality evaluation.

Provides GPT-4 based evaluation of agent responses across multiple
quality dimensions including relevance, accuracy, completeness,
citation quality, and clarity.
"""

import logging
import json
from typing import Dict, Any, Optional

from openai import OpenAI

from monitoring.config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class LLMJudge:
    """
    Evaluate agent responses using GPT-4 as a judge.
    
    Provides multi-dimensional scoring (1-5 scale) across relevance,
    accuracy, completeness, citation quality, and clarity dimensions
    for GDPR compliance chatbot responses.
    """
    
    def __init__(self, api_key: Optional[str] = None) -> None:
        """
        Initialize LLM judge with OpenAI client.
        
        Args:
            api_key: Optional OpenAI API key. Falls back to config if not provided.
        
        Raises:
            Exception: If OpenAI client initialization fails (logged and re-raised)
        """
        logger.debug("Initializing LLMJudge")
        
        try:
            self.client = OpenAI(api_key=api_key or config.OPENAI_API_KEY)
            logger.debug("OpenAI client initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize OpenAI client: %s", e)
            raise
        
        self.model = config.LLM_JUDGE_MODEL
        logger.info("LLMJudge initialized with model: %s", self.model)
    
    def evaluate(self, question: str, answer: str, context: str = "") -> Dict[str, Any]:
        """
        Evaluate answer quality on multiple dimensions.
        
        Calls GPT-4 to score the answer on relevance, accuracy, completeness,
        citation quality, and clarity. Returns structured scores (1-5 scale)
        with an explanation.
        
        Args:
            question: User's original question
            answer: Agent's response to evaluate
            context: Optional context retrieved from RAG system
        
        Returns:
            Dict containing:
            - relevance: Score 1-5 (or 0 on error)
            - accuracy: Score 1-5 (or 0 on error)
            - completeness: Score 1-5 (or 0 on error)
            - citation: Score 1-5 (or 0 on error)
            - clarity: Score 1-5 (or 0 on error)
            - overall: Overall score 1-5 (or 0 on error)
            - explanation: Brief explanation or error message
            Returns error structure with 0 scores if evaluation fails.
        
        Raises:
            Does not raise exceptions; returns error structure instead
        """
        question_preview = question[:80] if len(question) > 80 else question
        logger.debug("Evaluating response for question: %s", question_preview)
        
        try:
            prompt = self._build_evaluation_prompt(question, answer, context)
            logger.debug("Evaluation prompt built (length: %d chars)", len(prompt))
            
            logger.debug("Calling OpenAI API with model: %s", self.model)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=300
            )
            
            logger.debug("OpenAI API call successful")
            
            # Parse JSON response
            response_content = response.choices[0].message.content
            logger.debug("Parsing JSON response (length: %d chars)", len(response_content))
            
            try:
                result = json.loads(response_content)
                logger.debug("JSON parsing successful")
            except json.JSONDecodeError as e:
                logger.exception("Failed to parse JSON response: %s", e)
                return self._get_error_result(f"JSON parsing error: {str(e)}")
            
            # Validate result has required fields
            required_fields = ['relevance', 'accuracy', 'completeness', 
                             'citation', 'clarity', 'overall', 'explanation']
            missing_fields = [f for f in required_fields if f not in result]
            
            if missing_fields:
                logger.warning("Missing required fields in evaluation response: %s", missing_fields)
                return self._get_error_result(f"Missing required fields: {', '.join(missing_fields)}")
            
            # Log scores
            logger.info("Evaluation complete - Overall: %.1f, Relevance: %.1f, Accuracy: %.1f, Completeness: %.1f, Citation: %.1f, Clarity: %.1f",
                       result['overall'], result['relevance'], result['accuracy'],
                       result['completeness'], result['citation'], result['clarity'])
            
            logger.debug("Explanation: %s", result['explanation'])
            
            return result
            
        except Exception as e:
            logger.exception("Failed to evaluate response: %s", e)
            return self._get_error_result(f"Evaluation error: {str(e)}")
    
    def _build_evaluation_prompt(self, question: str, answer: str, context: str) -> str:
        """
        Build the evaluation prompt for GPT-4 judge.
        
        Args:
            question: User's original question
            answer: Agent's response to evaluate
            context: Retrieved context from RAG system
        
        Returns:
            Formatted prompt string with evaluation instructions
        """
        logger.debug("Building evaluation prompt")
        
        context_preview = context[:500] if context else "Not provided"
        prompt_length = len(question) + len(answer) + len(context_preview)
        logger.debug("Prompt components: question=%d chars, answer=%d chars, context=%d chars",
                    len(question), len(answer), len(context_preview))
        
        return f"""You are evaluating a GDPR compliance chatbot's response quality.

Question: {question}

Answer: {answer}

Context Retrieved: {context_preview}

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
    
    def _get_error_result(self, error_message: str) -> Dict[str, Any]:
        """
        Return error result structure with zero scores.
        
        Args:
            error_message: Description of the error that occurred
        
        Returns:
            Dict with all scores set to 0 and error message in explanation
        """
        logger.debug("Returning error result: %s", error_message)
        
        return {
            "relevance": 0,
            "accuracy": 0,
            "completeness": 0,
            "citation": 0,
            "clarity": 0,
            "overall": 0,
            "explanation": f"Error: {error_message}"
        }
