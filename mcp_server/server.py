from fastmcp import FastMCP
import logging
from dotenv import load_dotenv
import os
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
        import os
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

@mcp.tool()
def ask_gdpr_question(question: str):
    """Ask the GDPR Agent a question about GDPR compliance, regulations, or enforcement cases."""
    response = get_agent().invoke({"question": question})
    return response["answer"]

@mcp.tool()
def get_gdpr_sources(question: str):
    """Retrieve the sources used in the answer from the GDPR Agent from compliance, regulations or enforcement cases."""
    response = get_agent().invoke({"question": question})
    return response["context"]

if __name__ == "__main__":
    preload_agent()
    mcp.run()
