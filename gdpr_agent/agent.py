# gdpr_agent/agent.py
import os
from typing import Dict, Any

class GDPRAgent:
    """
    Wrapper class for GDPR Agent that can be instantiated and invoked directly.
    This class wraps the LangGraph app built by build_agent().
    """
    
    def __init__(self, openai_api_key: str = None):
        """
        Initialize the GDPR Agent.
        
        Args:
            openai_api_key: OpenAI API key. If None, will try to read from OPENAI_API_KEY env var.
        """
        from gdpr_agent import config, build_agent
        
        # Get API key from parameter or environment
        if openai_api_key is None:
            openai_api_key = os.environ.get("OPENAI_API_KEY")
        
        if not openai_api_key:
            raise ValueError("OpenAI API key is required. Pass it to __init__ or set OPENAI_API_KEY environment variable.")
        
        # Setup config and build the agent
        config.setup(openai_api_key)
        self.app = build_agent()
    
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
        # Handle simple question format
        if "question" in input_data and "original_question" not in input_data:
            question = input_data["question"]
            state = {
                "original_question": question,
                "current_query": question,
                "retrieved_context": "",
                "generated_answer": "",
                "retrieval_loop_count": 0,
                "generation_loop_count": 0,
                "is_answer_complete": True,
                "expanded_search_used": False,
                "needs_example_expansion": False
            }
        else:
            # Use provided state directly
            state = input_data
        
        # Invoke the LangGraph app
        final_state = self.app.invoke(state)
        
        # Return standardized output
        return {
            "answer": final_state.get("generated_answer", ""),
            "context": final_state.get("retrieved_context", ""),
            "original_question": final_state.get("original_question", ""),
            "current_query": final_state.get("current_query", "")
        }
    
    def predict(self, question: str) -> Dict[str, str]:
        """
        Simplified predict method that takes a string question.
        
        Args:
            question: The question to ask the agent
        
        Returns:
            Dictionary with "answer" and "context"
        """
        result = self.invoke({"question": question})
        return {
            "answer": result["answer"],
            "context": result["context"]
        }


# Convenience function for direct instantiation
def create_agent(openai_api_key: str = None) -> GDPRAgent:
    """
    Create and return a GDPRAgent instance.
    
    Args:
        openai_api_key: OpenAI API key. If None, reads from OPENAI_API_KEY env var.
    
    Returns:
        Initialized GDPRAgent instance
    """
    return GDPRAgent(openai_api_key=openai_api_key)