"""
Request logger for endpoint traffic simulation.
Logs inference requests to Delta table with response classification.
"""
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, TimestampType, DoubleType, DateType, BooleanType

# Import the response classifier
try:
    from monitoring.utils.response_classifier import ResponseClassifier
except ImportError:
    # Fallback if import fails
    import re
    class ResponseClassifier:
        def __init__(self, min_answer_length: int = 20):
            self.min_answer_length = min_answer_length
            self.insufficient_context_regex = re.compile(
                r"no\s+(?:sufficient|enough)\s+(?:information|evidence|context)|"
                r"(?:the\s+)?context\s+(?:does\s+not|doesn't|has\s+no)",
                re.IGNORECASE
            )
        
        def classify(self, response: str) -> Tuple[str, Optional[str], Optional[str]]:
            if not response or not response.strip():
                return ("empty", None, "Response is empty")
            if self.insufficient_context_regex.search(response):
                return ("refusal", "insufficient_context", "Contains refusal pattern")
            if len(response.strip()) < self.min_answer_length:
                return ("vague", None, f"Too short ({len(response)} chars)")
            return ("answer", None, None)


class RequestLogger:
    """Log endpoint requests to Delta table with automatic response classification"""
    
    def __init__(self, log_table: str = "main.default.gdpr_agent_inference_logs"):
        self.log_table = log_table
        self.spark = SparkSession.builder.getOrCreate()
        self.logs_buffer = []
        self.classifier = ResponseClassifier()
        
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
    
    def log_request(self, request_data: Dict):
        """
        Add a request to the buffer for batch writing.
        Automatically classifies the response if present.
        """
        # Classify the response if available
        answer = request_data.get('answer')
        if answer and request_data['status'] == 'success':
            classification, refusal_type, reason = self.classifier.classify(answer)
            is_valid = classification == "answer"
        else:
            classification = None
            refusal_type = None
            reason = None
            is_valid = False
        
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
    
    def flush(self) -> int:
        """Write all buffered logs to Delta table."""
        if not self.logs_buffer:
            print("⚠️  No logs to write")
            return 0
        
        try:
            logs_df = self.spark.createDataFrame(self.logs_buffer, self.log_schema)
            logs_df.write.mode("append").saveAsTable(self.log_table)
            
            count = len(self.logs_buffer)
            
            # Show classification breakdown
            valid_count = sum(1 for log in self.logs_buffer if log.get('is_valid_answer'))
            refusal_count = sum(1 for log in self.logs_buffer if log.get('response_classification') == 'refusal')
            
            print(f"✅ Wrote {count} logs to {self.log_table}")
            print(f"   📊 Valid answers: {valid_count}, Refusals: {refusal_count}")
            
            # Clear buffer
            self.logs_buffer = []
            
            return count
            
        except Exception as e:
            print(f"❌ Failed to write logs: {e}")
            return 0
    
    def log_success(self, request_id: str, question: str, answer: str, 
                    context: str, latency_ms: float, timestamp: datetime = None):
        """Convenience method to log successful request (auto-classifies response)"""
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
                  latency_ms: float = 0.0, timestamp: datetime = None):
        """Convenience method to log error"""
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
                      timestamp: datetime = None):
        """Convenience method to log exception"""
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
    
    def get_classification_stats(self) -> Dict:
        """Get classification statistics from current buffer"""
        if not self.logs_buffer:
            return {}
        
        total = len(self.logs_buffer)
        valid = sum(1 for log in self.logs_buffer if log.get('is_valid_answer'))
        refusals = sum(1 for log in self.logs_buffer if log.get('response_classification') == 'refusal')
        vague = sum(1 for log in self.logs_buffer if log.get('response_classification') == 'vague')
        
        return {
            'total': total,
            'valid_answer_count': valid,
            'refusal_count': refusals,
            'vague_count': vague,
            'valid_answer_rate': valid / total if total > 0 else 0,
            'refusal_rate': refusals / total if total > 0 else 0
        }