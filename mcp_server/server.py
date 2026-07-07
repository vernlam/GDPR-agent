from fastmcp import FastMCP
import logging
from dotenv import load_dotenv
import os
import time
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


mcp = FastMCP("GDPR Agent MCP Server")

agent = None


def get_agent():
    global agent
    if agent is None:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from gdpr_agent.agent import GDPRAgent
        try:
            agent = GDPRAgent()
            logger.info("GDPR Agent instance created successfully")
            return agent
        except Exception as e:
            logger.exception("Failed to create GDPR Agent instance: %s", e)
            raise
    else:
        return agent

def preload_agent():
    get_agent()
    logger.info("Agent preloaded and ready.")

def invoke_with_retry(question: str = None,max_retries=3):

    if not question or question.strip() == "":
        raise ValueError("No question was inputted.")
    tries = 0
    base_delay = 2

    while tries < max_retries:
        try:
            response = get_agent().invoke({"question": question})
            return response
        except Exception as e:
            logger.warning("Error invoking. retrying attempt number: %s | %s",tries,e)
            tries += 1
            if tries == max_retries:
                logger.exception("Max retries attempted reached %s",e)
                raise
            else:
                time.sleep(base_delay * (2**tries))


@mcp.tool()
def ask_gdpr_question(question: str):
    """Ask the GDPR Agent a question about GDPR compliance, regulations, or enforcement cases."""
    response = invoke_with_retry(question)
    return response["answer"]

@mcp.tool()
def get_gdpr_sources(question: str):
    """Retrieve the sources used in the answer from the GDPR Agent from compliance, regulations or enforcement cases."""
    response = invoke_with_retry(question)
    return response["context"]

if __name__ == "__main__":
    preload_agent()
    mcp.run()
