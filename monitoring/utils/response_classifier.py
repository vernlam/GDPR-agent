"""
Response classifier for detecting non-answers and refusal types.

Provides pattern-based classification of agent responses to identify refusals,
vague responses, and valid answers. Supports multiple refusal types including
insufficient context, ambiguous questions, out-of-scope requests, and capability
limitations.
"""

import logging
import re
from typing import Tuple, Optional, List, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class ResponseClassifier:
    """
    Classify agent responses to detect refusals and non-answers.
    
    Uses regex pattern matching to categorize responses into:
    - Valid answers
    - Refusals (with subtypes: insufficient_context, ambiguous_question, 
      out_of_scope, capability_limitation)
    - Vague responses (too short or non-committal)
    - Empty responses
    """
    
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
    
    def __init__(self, min_answer_length: int = 20) -> None:
        """
        Initialize classifier with configurable minimum answer length.
        
        Args:
            min_answer_length: Minimum character count for a valid answer
                              (responses shorter than this may be classified as vague)
        
        Raises:
            Does not raise exceptions; logs initialization errors
        """
        logger.debug("Initializing ResponseClassifier with min_answer_length=%d", min_answer_length)
        
        self.min_answer_length = min_answer_length
        
        try:
            # Compile all pattern groups
            self.refusal_patterns = {
                'insufficient_context': re.compile('|'.join(self.INSUFFICIENT_CONTEXT_PATTERNS), re.IGNORECASE),
                'ambiguous_question': re.compile('|'.join(self.AMBIGUOUS_QUESTION_PATTERNS), re.IGNORECASE),
                'out_of_scope': re.compile('|'.join(self.OUT_OF_SCOPE_PATTERNS), re.IGNORECASE),
                'capability_limitation': re.compile('|'.join(self.CAPABILITY_LIMITATION_PATTERNS), re.IGNORECASE),
            }
            logger.debug("Compiled %d refusal pattern groups", len(self.refusal_patterns))
            
            self.vague_regex = re.compile('|'.join(self.VAGUE_PATTERNS), re.IGNORECASE)
            logger.debug("Compiled vague response patterns")
            
            logger.info("ResponseClassifier initialized successfully with %d refusal types", 
                       len(self.refusal_patterns))
            
        except re.error as e:
            logger.exception("Failed to compile regex patterns: %s", e)
            # Set empty patterns as fallback
            self.refusal_patterns = {}
            self.vague_regex = re.compile(r'(?!.*)')  # Never matches
        except Exception as e:
            logger.exception("Unexpected error during ResponseClassifier initialization: %s", e)
            self.refusal_patterns = {}
            self.vague_regex = re.compile(r'(?!.*)')
    
    def classify(self, response: str) -> Tuple[str, Optional[str], Optional[str]]:
        """
        Classify a response with refusal type and reason.
        
        Checks response against refusal patterns in priority order:
        1. Insufficient context
        2. Ambiguous question
        3. Out of scope
        4. Capability limitation
        Then checks for vague/short responses, and finally classifies as valid answer.
        
        Args:
            response: The agent's response text to classify
            
        Returns:
            Tuple of (classification, refusal_type, reason):
            - classification: "answer", "refusal", "vague", or "empty"
            - refusal_type: Specific type if refusal ("insufficient_context", 
              "ambiguous_question", "out_of_scope", "capability_limitation"), or None
            - reason: Human-readable explanation of classification, or None for valid answers
        
        Raises:
            Does not raise exceptions; returns ("answer", None, None) on error
        """
        try:
            if not response or not response.strip():
                logger.debug("Response is empty or whitespace")
                return ("empty", None, "Response is empty or whitespace")
            
            response_clean = response.strip()
            logger.debug("Classifying response (length=%d chars)", len(response_clean))
            
            # Check each refusal type in priority order
            for refusal_type, pattern in self.refusal_patterns.items():
                try:
                    match = pattern.search(response)
                    if match:
                        matched_text = match.group()[:50]  # Truncate for logging
                        reason = f"Contains {refusal_type.replace('_', ' ')} pattern: '{match.group()}'"
                        logger.debug("Classified as refusal (type=%s, match='%s...')", 
                                   refusal_type, matched_text)
                        return ("refusal", refusal_type, reason)
                except Exception as e:
                    logger.exception("Error checking refusal pattern %s: %s", refusal_type, e)
                    continue
            
            # Check for vague responses
            if len(response_clean) < self.min_answer_length:
                try:
                    vague_match = self.vague_regex.search(response)
                    if vague_match:
                        logger.debug("Classified as vague (short and matches vague pattern)")
                        return ("vague", None, f"Too short and vague: '{response_clean}'")
                    logger.debug("Classified as vague (too short: %d chars)", len(response_clean))
                    return ("vague", None, f"Response too short ({len(response_clean)} chars)")
                except Exception as e:
                    logger.exception("Error checking vague pattern: %s", e)
                    # Continue to classify as answer if vague check fails
            
            logger.debug("Classified as valid answer")
            return ("answer", None, None)
            
        except AttributeError as e:
            logger.exception("AttributeError during classification (invalid response type?): %s", e)
            return ("answer", None, None)
        except Exception as e:
            logger.exception("Unexpected error during classification: %s", e)
            return ("answer", None, None)
    
    def is_valid_answer(self, response: str) -> bool:
        """
        Check if response is a valid answer (not a refusal/vague/empty).
        
        Args:
            response: The agent's response text
            
        Returns:
            True if it's a valid answer, False otherwise
            (returns False on error)
        
        Raises:
            Does not raise exceptions; returns False on error
        """
        try:
            classification, _, _ = self.classify(response)
            is_valid = classification == "answer"
            logger.debug("is_valid_answer check: %s (classification=%s)", is_valid, classification)
            return is_valid
        except Exception as e:
            logger.exception("Error in is_valid_answer: %s", e)
            return False
    
    def get_refusal_type(self, response: str) -> Optional[str]:
        """
        Get just the refusal type (or None if it's a valid answer).
        
        Args:
            response: The agent's response text
        
        Returns:
            One of: 'insufficient_context', 'ambiguous_question', 'out_of_scope',
            'capability_limitation', or None if not a refusal
            (returns None on error)
        
        Raises:
            Does not raise exceptions; returns None on error
        """
        try:
            classification, refusal_type, _ = self.classify(response)
            if classification == "refusal":
                logger.debug("get_refusal_type: %s", refusal_type)
                return refusal_type
            logger.debug("get_refusal_type: None (classification=%s)", classification)
            return None
        except Exception as e:
            logger.exception("Error in get_refusal_type: %s", e)
            return None
    
    def get_statistics(self, responses: List[str]) -> Dict[str, Any]:
        """
        Get classification statistics for a list of responses.
        
        Classifies each response and computes counts and percentages for
        each classification type.
        
        Args:
            responses: List of response texts to analyze
            
        Returns:
            Dict containing:
            - total: Total number of responses
            - answer_count: Number of valid answers
            - refusal_count: Number of refusals
            - vague_count: Number of vague responses
            - empty_count: Number of empty responses
            - answer_rate: Percentage of valid answers (0.0-1.0)
            - refusal_rate: Percentage of refusals (0.0-1.0)
            - vague_rate: Percentage of vague responses (0.0-1.0)
            - empty_rate: Percentage of empty responses (0.0-1.0)
            Returns all zeros if input is empty or on error.
        
        Raises:
            Does not raise exceptions; returns empty stats dict on error
        """
        try:
            if not responses:
                logger.debug("Empty responses list provided to get_statistics")
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
            
            logger.debug("Computing statistics for %d responses", len(responses))
            
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
            
            logger.info("Classification statistics: total=%d, valid=%d (%.1f%%), refusals=%d (%.1f%%), vague=%d (%.1f%%)",
                       total, stats['answer_count'], stats['answer_rate'] * 100,
                       stats['refusal_count'], stats['refusal_rate'] * 100,
                       stats['vague_count'], stats['vague_rate'] * 100)
            
            return stats
            
        except (TypeError, ZeroDivisionError) as e:
            logger.exception("Error calculating statistics (type/division error): %s", e)
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
        except Exception as e:
            logger.exception("Unexpected error calculating statistics: %s", e)
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


# Convenience functions for quick use
def is_refusal(response: str) -> bool:
    """
    Quick check if response is a refusal, vague, or empty.
    
    Creates a classifier instance and checks if the response is anything
    other than a valid answer.
    
    Args:
        response: The agent's response text
    
    Returns:
        True if response is refusal/vague/empty, False if it's a valid answer
        (returns False on error)
    
    Raises:
        Does not raise exceptions; returns False on error
    """
    try:
        logger.debug("is_refusal convenience function called")
        classifier = ResponseClassifier()
        classification, _, _ = classifier.classify(response)
        result = classification in ('refusal', 'vague', 'empty')
        logger.debug("is_refusal result: %s (classification=%s)", result, classification)
        return result
    except Exception as e:
        logger.exception("Error in is_refusal convenience function: %s", e)
        return False


def classify_response(response: str) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Quick classification of a response.
    
    Creates a classifier instance and returns the full classification tuple.
    
    Args:
        response: The agent's response text
    
    Returns:
        Tuple of (classification, refusal_type, reason)
        (returns ("answer", None, None) on error)
    
    Raises:
        Does not raise exceptions; returns default tuple on error
    """
    try:
        logger.debug("classify_response convenience function called")
        classifier = ResponseClassifier()
        result = classifier.classify(response)
        logger.debug("classify_response result: classification=%s, refusal_type=%s", 
                   result[0], result[1])
        return result
    except Exception as e:
        logger.exception("Error in classify_response convenience function: %s", e)
        return ("answer", None, None)
