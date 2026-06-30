# gdpr_agent/agent.py
"""
GDPR Agent wrapper for LangGraph-based conversational AI.
Provides a simple interface for instantiating and invoking the agent.
"""

import os
import logging
import mlflow
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


class GDPRAgent:
    """
    Wrapper class for GDPR Agent that can be instantiated and invoked directly.
    This class wraps the LangGraph app built by build_agent().
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        """
        Initialize the GDPR Agent.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to read from OPENAI_API_KEY env var.
        """
        logger.info("Initializing GDPR Agent")
        
        try:
            from gdpr_agent import config, build_agent
        except ImportError as e:
            logger.exception("Failed to import required modules: %s", e)
            raise
        
        # Get API key from parameter or environment
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            logger.debug("OpenAI API key retrieved from environment variable")
        else:
            logger.debug("OpenAI API key provided as parameter")
        
        if not openai_api_key:
            error_msg = "OpenAI API key is required. Pass it to __init__ or set OPENAI_API_KEY environment variable."
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Setup config and build the agent
        logger.info("Setting up agent configuration")
        try:
            config.setup(openai_api_key)
            logger.info("Building LangGraph agent application")
            self.app = build_agent()
            logger.info("GDPR Agent initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize GDPR Agent: %s", e)
            raise
    
    @mlflow.trace(name="gdpr_agent", span_type="CHAIN")
    def invoke(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Invoke the agent with a question.
        
        Args:
            input_data: Dictionary with either:
                - {"question": "your question"} - simple format
                - Full state dict with all fields
        
        Returns:
            Dictionary with:
                - "answer": The generated answer
                - "context": The retrieved context
                - "original_question": The original question
                - "current_query": The current query (may be refined)
        """
        logger.debug("Agent invocation requested with input keys: %s", list(input_data.keys()))
        
        # Handle simple question format
        if "question" in input_data and "original_question" not in input_data:
            question = input_data["question"]
            logger.info("Processing question: %s", question[:100] + "..." if len(question) > 100 else question)
            state = {
                "original_question": question,
                "current_query": question,
                "retrieved_context": "",
                "generated_answer": "",
                "sources_queried": [],
                "retrieval_loop_count": 0,
                "generation_loop_count": 0,
                "is_answer_complete": True,
                "expanded_search_used": False,
                "needs_example_expansion": False
            }
            logger.debug("Initialized agent state with default values")
        else:
            # Use provided state directly
            state = input_data
            logger.debug("Using provided state directly with %d fields", len(state))
        
        # Invoke the LangGraph app
        try:
            logger.info("Invoking LangGraph application")
            final_state = self.app.invoke(state)
            logger.info("Agent invocation completed successfully")
        except Exception as e:
            logger.exception("Agent invocation failed: %s", e)
            raise
        
        # Return standardized output
        result = {
            "answer": final_state.get("generated_answer", ""),
            "context": final_state.get("retrieved_context", ""),
            "original_question": final_state.get("original_question", ""),
            "current_query": final_state.get("current_query", "")
        }
        
        logger.debug("Returning result with answer length: %d characters", len(result["answer"]))
        return result
    
    def predict(self, question: str) -> Dict[str, str]:
        """
        Simplified predict method that takes a string question.
        
        Args:
            question: The question to ask the agent
        
        Returns:
            Dictionary with "answer" and "context"
        """
        logger.info("Predict method called with question: %s", question[:100] + "..." if len(question) > 100 else question)
        
        try:
            result = self.invoke({"question": question})
            return {
                "answer": result["answer"],
                "context": result["context"]
            }
        except Exception as e:
            logger.exception("Predict method failed: %s", e)
            raise


# Convenience function for direct instantiation
def create_agent(openai_api_key: Optional[str] = None) -> GDPRAgent:
    """
    Create and return a GDPRAgent instance.
    
    Args:
        openai_api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
    
    Returns:
        Initialized GDPRAgent instance
    """
    logger.info("Creating GDPR Agent instance via create_agent function")
    try:
        agent = GDPRAgent(openai_api_key=openai_api_key)
        logger.info("GDPR Agent instance created successfully")
        return agent
    except Exception as e:
        logger.exception("Failed to create GDPR Agent instance: %s", e)
        raise
