# GDPR Agent MCP Server

This server exposes the GDPR agent as a tool for other LLMs to call via MCP. Rather than answering from training data alone, it retrieves grounded context from real GDPR legislation, case law, and policy documents — so any MCP-compatible client (Claude Desktop, etc.) can get sourced, accurate answers to GDPR compliance questions.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create `.env` in this folder:
   ```
   OPENAI_API_KEY=your-key-here
   ```

3. Add this server to your Claude Desktop config (Settings → Developer → Edit Config):
   ```json
   {
     "mcpServers": {
       "gdpr-agent": {
         "command": "path-to-python",
         "args": [
           "path-to-repo\\mcp_server\\server.py"
         ]
       }
     }
   }
   ```
   The API key is picked up from `.env` automatically — no need to duplicate it in the Claude Desktop config.

## Known Issues / Gotchas

- **The agent must be preloaded at startup, not lazy-loaded on first tool call.** Importing `mlflow` inside the thread FastMCP uses to execute tool calls causes a deadlock — the call hangs indefinitely with no error. Loading the agent once on the main thread, before `mcp.run()`, avoids this.
- **The working directory must be set explicitly.** MLflow creates its local tracking database using a relative path. Running the script from a terminal at the repo root works fine, but Claude Desktop spawns the process from a different working directory, so the relative path fails (`sqlite3.OperationalError: unable to open database file`). `os.chdir()` at the top of `server.py` forces the working directory to the repo root regardless of who launches it.
- **Transient Databricks Vector Search failures are retried automatically.** `invoke_with_retry()` wraps the agent call with up to 3 attempts before raising, so a brief network blip doesn't fail the whole request.

## Example Usage

Once connected, ask Claude Desktop a GDPR question directly in chat:

> "What are the GDPR requirements for responding to a data deletion request?"

Claude will call the `ask_gdpr_question` tool, which retrieves relevant legislation/case law and returns a grounded answer — citing specific articles (e.g. Article 17) rather than relying on general knowledge alone. Use `get_gdpr_sources` if you want the raw retrieved context instead of a synthesised answer.

