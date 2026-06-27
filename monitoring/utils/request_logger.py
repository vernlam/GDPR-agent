"""
Request logger for endpoint traffic simulation.

Logs inference requests to Delta table with response classification.
Provides buffered logging for batch writes, automatic response classification,
and convenience methods for logging successes, errors, and exceptions.
"""

import logging
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField, StringType, TimestampType, 
    DoubleType, DateType, BooleanType
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the response classifier
try:
    from monitoring.utils.response_classifier import ResponseClassifier
    logger.debug("ResponseClassifier imported successfully")
except ImportError:
    logger.warning("Failed to import ResponseClassifier, using fallback implementation")
    # Fallback if import fails
    import re
    
    class ResponseClassifier:
        """Fallback response classifier implementation."""
        
        def __init__(self, min_answer_length: int = 20) -> None:
            self.min_answer_length = min_answer_length
            self.insufficient_context_regex = re.compile(
                r"no\s+(?:sufficient|enough)\s+(?:information|evidence|context)|"
                r"(?:the\s+)?context\s+(?:does\s+not|doesn't|has\s+no)",
                re.IGNORECASE
            )
        
        def classify(self, response: str) -> Tuple[str, Optional[str], Optional[str]]:
            """
            Classify response type.
            
            Args:
                response: Response text to classify
            
            Returns:
                Tuple of (classification, refusal_type, reason)
            """
            if not response or not response.strip():
                return ("empty", None, "Response is empty")
            if self.insufficient_context_regex.search(response):
                return ("refusal", "insufficient_context", "Contains refusal pattern")
            if len(response.strip()) < self.min_answer_length:
                return ("vague", None, f"Too short ({len(response)} chars)")
            return ("answer", None, None)


class RequestLogger:
    """
    Log endpoint requests to Delta table with automatic response classification.
    
    Provides buffered logging for batch writes to Delta tables, automatic
    response classification using keyword patterns, and convenience methods
    for logging different request outcomes.
    """
    
    def __init__(self, log_table: str = "main.default.gdpr_agent_inference_logs") -> None:
        """
        Initialize request logger.
        
        Args:
            log_table: Fully qualified Delta table name for logging
        
        Raises:
            Exception: If SparkSession or ResponseClassifier initialization fails
                      (logged and re-raised)
        """
        logger.debug("Initializing RequestLogger with table: %s", log_table)
        
        self.log_table = log_table
        
        try:
            self.spark = SparkSession.builder.getOrCreate()
            logger.debug("SparkSession obtained successfully")
        except Exception as e:
            logger.exception("Failed to obtain SparkSession: %s", e)
            raise
        
        self.logs_buffer = []
        
        try:
            self.classifier = ResponseClassifier()
            logger.debug("ResponseClassifier initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize ResponseClassifier: %s", e)
            raise
        
        # Enhanced log schema with classification fields
        self.log_schema = StructType([
            StructField("timestamp", TimestampType(), False),
            StructField("request_id", StringType(), False),
            StructField("question", StringType(), False),
            StructField("answer", StringType(), True),
            StructField("context", StringType(), True),
            StructField("latency_ms", DoubleType(), True),
            StructField("status", StringType(), False),
            StructField("error_message", StringType(), True),
            StructField("date", DateType(), False),
            # New classification fields
            StructField("response_classification", StringType(), True),
            StructField("refusal_type", StringType(), True),
            StructField("classification_reason", StringType(), True),
            StructField("is_valid_answer", BooleanType(), True)
        ])
        
        logger.info("RequestLogger initialized successfully with table: %s", log_table)
    
    def log_request(self, request_data: Dict[str, Any]) -> None:
        """
        Add a request to the buffer for batch writing.
        
        Automatically classifies the response if present using the
        ResponseClassifier. Adds the log entry to the internal buffer
        for later batch writing via flush().
        
        Args:
            request_data: Dict containing request details:
                - request_id (str, required): Unique request identifier
                - question (str, required): User's question
                - status (str, required): Request status ('success', 'error', 'exception')
                - answer (str, optional): Agent's response
                - context (str, optional): Retrieved context
                - latency_ms (float, optional): Request latency in milliseconds
                - error_message (str, optional): Error message if status != 'success'
                - timestamp (datetime, optional): Request timestamp (defaults to now)
        
        Returns:
            None
        
        Raises:
            Does not raise exceptions; logs errors internally
        """
        try:
            # Classify the response if available
            answer = request_data.get('answer')
            if answer and request_data.get('status') == 'success':
                logger.debug("Classifying response for request: %s", request_data.get('request_id', 'unknown'))
                classification, refusal_type, reason = self.classifier.classify(answer)
                is_valid = classification == "answer"
                logger.debug("Classification result: %s (refusal_type=%s, is_valid=%s)", 
                           classification, refusal_type, is_valid)
            else:
                classification = None
                refusal_type = None
                reason = None
                is_valid = False
                logger.debug("No answer to classify for request: %s (status=%s)", 
                           request_data.get('request_id', 'unknown'), request_data.get('status'))
            
            log_entry = {
                'timestamp': request_data.get('timestamp', datetime.now()),
                'request_id': request_data['request_id'],
                'question': request_data['question'],
                'answer': answer,
                'context': request_data.get('context'),
                'latency_ms': request_data.get('latency_ms', 0.0),
                'status': request_data['status'],
                'error_message': request_data.get('error_message'),
                'date': request_data.get('timestamp', datetime.now()).date(),
                'response_classification': classification,
                'refusal_type': refusal_type,
                'classification_reason': reason,
                'is_valid_answer': is_valid
            }
            
            self.logs_buffer.append(log_entry)
            logger.debug("Added log entry to buffer (buffer size: %d)", len(self.logs_buffer))
            
        except KeyError as e:
            logger.exception("Missing required field in request_data: %s", e)
            # Do not re-raise - log and continue
        except Exception as e:
            logger.exception("Failed to log request: %s", e)
            # Do not re-raise - log and continue
    
    def flush(self) -> int:
        """
        Write all buffered logs to Delta table.
        
        Creates a Spark DataFrame from the buffered logs and appends
        them to the configured Delta table. Clears the buffer after
        successful write.
        
        Returns:
            Number of logs written (or 0 if no logs or write failed)
        
        Raises:
            Does not raise exceptions; returns 0 on error
        """
        if not self.logs_buffer:
            logger.debug("No logs in buffer to write")
            return 0
        
        logger.info("Flushing %d logs to table: %s", len(self.logs_buffer), self.log_table)
        
        try:
            logs_df = self.spark.createDataFrame(self.logs_buffer, self.log_schema)
            logger.debug("Created DataFrame with %d rows", len(self.logs_buffer))
            
            logs_df.write.mode("append").saveAsTable(self.log_table)
            logger.debug("DataFrame written to table successfully")
            
            count = len(self.logs_buffer)
            
            # Show classification breakdown
            valid_count = sum(1 for log in self.logs_buffer if log.get('is_valid_answer'))
            refusal_count = sum(1 for log in self.logs_buffer if log.get('response_classification') == 'refusal')
            
            logger.info("Successfully wrote %d logs to %s (valid: %d, refusals: %d)",
                       count, self.log_table, valid_count, refusal_count)
            
            # Clear buffer
            self.logs_buffer = []
            logger.debug("Buffer cleared")
            
            return count
            
        except Exception as e:
            logger.exception("Failed to flush logs to table %s: %s", self.log_table, e)
            return 0
    
    def log_success(self, request_id: str, question: str, answer: str, 
                    context: str, latency_ms: float, timestamp: datetime = None) -> None:
        """
        Convenience method to log successful request.
        
        Automatically classifies the response using ResponseClassifier.
        
        Args:
            request_id: Unique request identifier
            question: User's question
            answer: Agent's response
            context: Retrieved context used for the response
            latency_ms: Request latency in milliseconds
            timestamp: Request timestamp (defaults to now if not provided)
        
        Returns:
            None
        
        Raises:
            Does not raise exceptions; errors logged internally
        """
        logger.debug("Logging successful request: %s", request_id)
        
        self.log_request({
            'timestamp': timestamp or datetime.now(),
            'request_id': request_id,
            'question': question,
            'answer': answer,
            'context': context,
            'latency_ms': latency_ms,
            'status': 'success',
            'error_message': None
        })
    
    def log_error(self, request_id: str, question: str, error_message: str,
                  latency_ms: float = 0.0, timestamp: datetime = None) -> None:
        """
        Convenience method to log error.
        
        Args:
            request_id: Unique request identifier
            question: User's question
            error_message: Error message (truncated to 500 chars)
            latency_ms: Request latency in milliseconds (defaults to 0.0)
            timestamp: Request timestamp (defaults to now if not provided)
        
        Returns:
            None
        
        Raises:
            Does not raise exceptions; errors logged internally
        """
        logger.debug("Logging error for request: %s", request_id)
        
        self.log_request({
            'timestamp': timestamp or datetime.now(),
            'request_id': request_id,
            'question': question,
            'answer': None,
            'context': None,
            'latency_ms': latency_ms,
            'status': 'error',
            'error_message': error_message[:500]
        })
    
    def log_exception(self, request_id: str, question: str, exception: Exception,
                      timestamp: datetime = None) -> None:
        """
        Convenience method to log exception.
        
        Args:
            request_id: Unique request identifier
            question: User's question
            exception: Exception object (converted to string, truncated to 500 chars)
            timestamp: Request timestamp (defaults to now if not provided)
        
        Returns:
            None
        
        Raises:
            Does not raise exceptions; errors logged internally
        """
        logger.debug("Logging exception for request: %s", request_id)
        
        self.log_request({
            'timestamp': timestamp or datetime.now(),
            'request_id': request_id,
            'question': question,
            'answer': None,
            'context': None,
            'latency_ms': 0.0,
            'status': 'exception',
            'error_message': str(exception)[:500]
        })
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """
        Get classification statistics from current buffer.
        
        Computes statistics about the classification distribution in
        the current buffer (not persisted logs).
        
        Returns:
            Dict containing:
            - total: Total number of logs in buffer
            - valid_answer_count: Number of valid answers
            - refusal_count: Number of refusals
            - vague_count: Number of vague responses
            - valid_answer_rate: Percentage of valid answers
            - refusal_rate: Percentage of refusals
            Returns empty dict if buffer is empty.
        
        Raises:
            Does not raise exceptions; returns empty dict on error
        """
        try:
            if not self.logs_buffer:
                logger.debug("Buffer is empty, returning empty stats")
                return {}
            
            total = len(self.logs_buffer)
            valid = sum(1 for log in self.logs_buffer if log.get('is_valid_answer'))
            refusals = sum(1 for log in self.logs_buffer if log.get('response_classification') == 'refusal')
            vague = sum(1 for log in self.logs_buffer if log.get('response_classification') == 'vague')
            
            stats = {
                'total': total,
                'valid_answer_count': valid,
                'refusal_count': refusals,
                'vague_count': vague,
                'valid_answer_rate': valid / total if total > 0 else 0,
                'refusal_rate': refusals / total if total > 0 else 0
            }
            
            logger.debug("Classification stats: total=%d, valid=%d, refusals=%d, vague=%d",
                        total, valid, refusals, vague)
            
            return stats
            
        except (TypeError, ZeroDivisionError) as e:
            logger.exception("Error calculating classification stats: %s", e)
            return {}
        except Exception as e:
            logger.exception("Unexpected error calculating classification stats: %s", e)
            return {}
