%md 
What did I learn?

### Vector Search
By using Databricks vector search, querying similarity search is much faster. It allows direct integration with delta lake and unity catalog so there isn't a need for an external ETL pipeline. The Change Data Feed enables auto-sync with the delta tables, so whenever that table changes, Vector Search automatically updates its search index using this. It also runs on severless compute that autoscales dynamically, using an endpoint.

### Langgraph Set-up
The GDPR agent has three separate vector libraries to look at in order to generate its query. The historical fines database, the GDPR statute as well as the companie's privacy policy. It needs to synthetise all three pieces of evidence together before returning an answer to the end user. Therefore, there should be three separate vector searches happening in parallel. Once queried, it aggregates the selected chunks into a unified context pool which, as a whole, has its quality evaluated. If the grade is False, the agent determines that it cannot confidently answer the question. If the question cannot be answered, then the agent will attempt at a query re-write, to again search. If the grade eventually passes, then it moves onto generation. Finally, once generated, there is a groundedness and safety verification. If the answer references facts not explicitly contained in the documents, it fails the groundedness grade and triggers a correction loop.

### Pydantic Set-up
I implemented Pydantic schemas for grading functions to validate the LLM response before processing. This is because the functions expect a structured JSON format with fields `is_relevant`, `is_grounded` and `reason`. Without this, malformed JSON responses could crash the agent and produce unreliable results. Pydantic ensures type is adhered to, validates response structure and provides clear error messages when the LLM returns unexpected output. This makes the agent more robust and easier to debug. The validation happens immediately after each grader call, catching issues before they propagate through the workflow.

### Included BM25 Search for Historical Fines Retrieval
I implemented BM25 Search in order to capture the actual case law titles, company names, exact fine amounts for specificity. This was done only on the historical fines, while regular dense search was used for the privacy policy and GDPR documents. Dense retrievl was also used for Historical Fines, and combined together with BM25 in a hybrid search.

### Groundedness Check
It is preferntial for the groundedness check used an ugpraded model `gpt-4o` as opposed to `gpt-4o-mini` as its the step that requires greater accuracy, to eliminate hallucination risk. If the generation fails the groundedness check, the agent routes to a stricter generation node `node_regenerate_strict` which has a more strict prompt. However, for the purposes of limiting the cost, I have left it as a smaller model.

### Evaluation Harness & CI/CD Pipeline
I built a standalone evaluation harness as a Python package (`eval_harness/`) to automate pre-production validation of the GDPR agent. The harness runs test cases from a golden dataset against the serving endpoint, scoring responses on source accuracy and content match. Each test case validates that the agent retrieves information from the correct documents (GDPR statutes, historical fines, or privacy policy) and includes expected content in its answers.

The evaluation harness is integrated into a GitHub Actions CI/CD pipeline that automatically runs on every push to `main` or `develop` when code in `gdpr_agent/`, `eval_harness/`, or `evaluation_data/` changes. The workflow gates deployments with a 90% pass rate threshold - if evaluation scores fall below this, the pipeline fails and blocks the merge. This prevents regressions from reaching production.

All evaluation runs are logged to MLflow experiments, creating a historical record of agent performance over time. Each run captures metrics like pass rate, average score, source accuracy, and category-level breakdowns, along with artifacts including full result CSVs and failed test cases for debugging. This enables tracking performance trends, comparing runs, and identifying degradation patterns across different agent versions.

The modular structure (`evaluator.py`, `runner.py`, `utils.py`, `cli.py`) separates scoring logic, execution orchestration, reporting utilities, and the CLI entry point, making the harness reusable across notebooks, CI/CD, and manual evaluation workflows. This approach ensures consistent validation whether testing locally in Databricks or automatically through GitHub Actions.