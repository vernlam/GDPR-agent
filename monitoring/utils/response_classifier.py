"""
Response classifier for detecting non-answers and refusal types.
"""
import re
from typing import Tuple, Optional, List, Dict


class ResponseClassifier:
    """Classify agent responses to detect refusals and non-answers"""
    
    # Pattern groups for different refusal types
    INSUFFICIENT_CONTEXT_PATTERNS = [
        r"no\s+(?:sufficient|enough)\s+(?:information|evidence|context)",
        r"(?:the\s+)?context\s+(?:does\s+not|doesn't|has\s+no)\s+(?:contain|provide|include)",
        r"not\s+(?:mentioned|specified|provided|included)\s+in\s+(?:the\s+)?(?:context|document)",
        r"(?:the\s+)?(?:provided|given)\s+context\s+(?:does\s+not|doesn't)",
        r"insufficient\s+(?:information|context|data)",
        r"(?:the\s+)?context\s+(?:is\s+)?(?:unclear|incomplete|lacks)",
        r"no\s+relevant\s+(?:information|context|data)",
        r"based\s+on\s+(?:the\s+)?(?:provided|given)\s+context,?\s+(?:i\s+)?(?:cannot|can't)",
        r"(?:the\s+)?document\s+(?:does\s+not|doesn't)\s+(?:mention|specify|provide)",
        r"not\s+enough\s+(?:information|context|evidence)",
    ]
    
    AMBIGUOUS_QUESTION_PATTERNS = [
        r"(?:your\s+)?question\s+(?:is\s+)?(?:unclear|ambiguous|vague)",
        r"(?:could\s+you|please)\s+(?:clarify|rephrase|be\s+more\s+specific)",
        r"(?:i\s+)?(?:need|require)\s+(?:more\s+)?(?:clarification|details)",
        r"what\s+(?:do\s+you\s+mean|are\s+you\s+asking)",
        r"(?:not\s+)?(?:sure|certain)\s+(?:what\s+you're|what\s+you\s+are)\s+asking",
    ]
    
    OUT_OF_SCOPE_PATTERNS = [
        r"(?:that's|that\s+is)\s+(?:outside|beyond)\s+(?:my\s+)?(?:scope|expertise|knowledge)",
        r"(?:i\s+)?(?:cannot|can't)\s+(?:help|assist)\s+with\s+(?:that|this)",
        r"(?:i'm|i\s+am)\s+(?:only|specifically)\s+(?:designed|trained)\s+(?:for|to)",
        r"(?:not\s+)?(?:related\s+to|about)\s+(?:GDPR|data\s+protection)",
        r"(?:outside|beyond)\s+(?:the\s+)?(?:scope\s+of|domain\s+of)",
    ]
    
    CAPABILITY_LIMITATION_PATTERNS = [
        r"(?:i\s+)?(?:cannot|can't|unable\s+to)\s+(?:access|retrieve|search)",
        r"(?:i\s+)?(?:don't|do\s+not)\s+have\s+(?:access|permission|ability)",
        r"(?:that\s+)?(?:requires|needs)\s+(?:external|real-time)\s+(?:access|data)",
        r"(?:i\s+)?(?:cannot|can't)\s+(?:browse|fetch|query)\s+(?:external|live)",
    ]
    
    VAGUE_PATTERNS = [
        r"^(?:yes|no|maybe|unclear|unknown)\.?$",
        r"^(?:i\s+)?(?:don't|do\s+not)\s+know\.?$",
        r"^not\s+(?:sure|certain|available)\.?$",
    ]
    
    def __init__(self, min_answer_length: int = 20):
        """
        Initialize classifier.
        
        Args:
            min_answer_length: Minimum character count for a valid answer
        """
        self.min_answer_length = min_answer_length
        
        # Compile all pattern groups
        self.refusal_patterns = {
            'insufficient_context': re.compile('|'.join(self.INSUFFICIENT_CONTEXT_PATTERNS), re.IGNORECASE),
            'ambiguous_question': re.compile('|'.join(self.AMBIGUOUS_QUESTION_PATTERNS), re.IGNORECASE),
            'out_of_scope': re.compile('|'.join(self.OUT_OF_SCOPE_PATTERNS), re.IGNORECASE),
            'capability_limitation': re.compile('|'.join(self.CAPABILITY_LIMITATION_PATTERNS), re.IGNORECASE),
        }
        
        self.vague_regex = re.compile('|'.join(self.VAGUE_PATTERNS), re.IGNORECASE)
    
    def classify(self, response: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Classify a response with refusal type.
        
        Args:
            response: The agent's response text
            
        Returns:
            Tuple of (classification, refusal_type, reason)
            - classification: "answer", "refusal", "vague", or "empty"
            - refusal_type: Specific type if refusal ("insufficient_context", "ambiguous_question", 
                           "out_of_scope", "capability_limitation", or None)
            - reason: Explanation of classification
        """
        if not response or not response.strip():
            return ("empty", None, "Response is empty or whitespace")
        
        response_clean = response.strip()
        
        # Check each refusal type in priority order
        for refusal_type, pattern in self.refusal_patterns.items():
            match = pattern.search(response)
            if match:
                reason = f"Contains {refusal_type.replace('_', ' ')} pattern: '{match.group()}'"
                return ("refusal", refusal_type, reason)
        
        # Check for vague responses
        if len(response_clean) < self.min_answer_length:
            vague_match = self.vague_regex.search(response)
            if vague_match:
                return ("vague", None, f"Too short and vague: '{response_clean}'")
            return ("vague", None, f"Response too short ({len(response_clean)} chars)")
        
        return ("answer", None, None)
    
    def is_valid_answer(self, response: str) -> bool:
        """
        Check if response is a valid answer (not a refusal/vague/empty).
        
        Args:
            response: The agent's response text
            
        Returns:
            True if it's a valid answer, False otherwise
        """
        classification, _, _ = self.classify(response)
        return classification == "answer"
    
    def get_refusal_type(self, response: str) -> Optional[str]:
        """
        Get just the refusal type (or None if it's a valid answer).
        
        Args:
            response: The agent's response text
        
        Returns:
            One of: 'insufficient_context', 'ambiguous_question', 'out_of_scope',
                   'capability_limitation', or None
        """
        classification, refusal_type, _ = self.classify(response)
        if classification == "refusal":
            return refusal_type
        return None
    
    def get_statistics(self, responses: List[str]) -> Dict[str, float]:
        """
        Get classification statistics for a list of responses.
        
        Args:
            responses: List of response texts
            
        Returns:
            Dict with counts and percentages for each classification type
        """
        if not responses:
            return {
                'total': 0,
                'answer_count': 0,
                'refusal_count': 0,
                'vague_count': 0,
                'empty_count': 0,
                'answer_rate': 0.0,
                'refusal_rate': 0.0,
                'vague_rate': 0.0,
                'empty_rate': 0.0,
            }
        
        classifications = [self.classify(r)[0] for r in responses]
        total = len(classifications)
        
        stats = {
            'total': total,
            'answer_count': classifications.count('answer'),
            'refusal_count': classifications.count('refusal'),
            'vague_count': classifications.count('vague'),
            'empty_count': classifications.count('empty'),
        }
        
        # Add percentages
        stats['answer_rate'] = stats['answer_count'] / total
        stats['refusal_rate'] = stats['refusal_count'] / total
        stats['vague_rate'] = stats['vague_count'] / total
        stats['empty_rate'] = stats['empty_count'] / total
        
        return stats


# Convenience functions for quick use
def is_refusal(response: str) -> bool:
    """Quick check if response is a refusal"""
    classifier = ResponseClassifier()
    classification, _, _ = classifier.classify(response)
    return classification in ('refusal', 'vague', 'empty')


def classify_response(response: str) -> Tuple[str, Optional[str], Optional[str]]:
    """Quick classification of a response"""
    classifier = ResponseClassifier()
    return classifier.classify(response)