# Learning Reflections: GDPR Agent Project

*A retrospective on building an AI-powered compliance agent with RAG, vector search, and production monitoring.* \
\
I established this project mainly for the purposes of learning how to implement a fully productionalised agentic RAG architecture, with monitoring and evaluation. Additionally, this was built on databricks to also learn how to use the platform and understand its capabilities for end-to-end LLM engineering.

**Note**: This project is not designed to be cloned and run independently. The GDPR enforcement data, internal policy documents, and vector indexes are stored in my private Databricks workspace and are not publicly available. The code serves as a demonstration of production ML architecture patterns.

---

## 🎯 Learning Objectives

### What I Set Out to Learn

* Agentic AI architectures (LangGraph, tool-calling, routing)
* RAG pipelines and vector search
* Production ML monitoring and evaluation
* MLOps on Databricks (Unity Catalog, Model Serving)
* End-to-end AI system design

### Why This Project?

To develop a deeper understanding of production ML/AI systems, software engineering design patterns and better understanding Databricks as a unified governance platform. I also wanted to develop an Agentic RAG system, after having spent some time learning about how RAG is implemented to solidify my understanding of the subject. I chose to create a GDPR compliance chatbot through exposure to the value of GDPR compliance in my previous professional experiences.



---

## 💡 Key Learnings

### Technical Skills Gained

**Data Engineering and Ingestion:**
* Uploading files to Volumes and writing ETL pipeline to ingest into structured Delta tables
* Using LLM to auto-translate to English before vectorisation
* Chunking based on headers, chunking to ensure sentences aren't cut-off
* Vectorisation, generating embeddings and adding metadata
* Automating vector syncing to include new embeddings if Volumes are updated
* Creating endpoint for Vector search

**Agentic AI:**
* Orchestrating multi-step agentic workflow using LangGraph with multiple LLM calls per query
* State management using LangGraph, updating a `TypedDict` as the graph is traversed. Edges read state to make routing decisions
* Router-based tool section to determine retrieval sources, developing tools as python functions
* Building self-correction loops to cover quality, completeness and groundedness

**Databricks Ecosystem:**
* Enabling Change Data Feed with Delta Tables to process only changed rows whenever new data gets updated
* Vector Search index management and sync
* Model Serving and endpoint configuration
* Dashboard to for monitoring and evaluation

**MLOps & Monitoring:**
* CI/CD using GitHub actions to automate evaluation harness and model serving
* Setting up LLM-as-judge to capture evaluation patterns Relevance, Accuracy, Completeness, and Clarity
* Setting up Cost monitoring

**Simulations**
* Simulated real users by using LLM to generate questions and run on a scheduled basis.
* Simulated a set of golden questions to evaluate against, using raw chunk data from Volumes.

### Conceptual Insights

* Building the production system is harder than building the model
* Evaluation is harder than building the system. Creating golden question set, and wiring the CI/CD pipeline was not that straight forward
* How to balance latency, cost and accuracy trade-offs

## 🏗️ Design Decisions

### Why LangGraph?

I chose LangGraph over plain LangChain because the agent needed explicit statement management to coordinate the multi-step workflow with self-correction loops.

The agent had to:

1. Route queries to the right data source (GDPR law, internal policy documents and/or case precedents)
2. Retrieve and grade relevance
3. Generate an answer
4. Check for completeness, quality, and groundedness
5. Self-correct by re-retrieving or regenerating if quality checks failed

With LangChain, tracking what the agent has tried, what failed or what to re-try across these steps would be difficult to debug.

* LangGraph enabled `TypedDict` state where every node was able to read/write from the same state object.
* Edge functions read state for routing, and the self-correction logic could inspect quality scores and route back to retrieval or generation nodes

### Why Multi-Source RAG?

I chose RAG to look at multiple sources because different sources answer different questions:

1. Statutory Law (GDPR Articles): Provides the legal framework and requirements
* Example query: "What are the lawful bases for processing personal data?"
* Needs: Official regulation text from GDPR articles

2. Internal Policy Documents: Shows how your organisation implements GDPR
* Example query: "What's our process for handling data deletion requests?"
* Needs: Company-specific procedures and workflows

3. Case Precedents (Enforcement Decisions): Provides real-world interpretation and consequences
* Example query: "Have companies been fined for similar violations?"
* Needs: Historical enforcement examples and regulatory interpretation

This approach was better that a unified index as mixing document types creates retrieval noise, and different sources should be weighted differently depending on the ask. 

Trade-off: Multi-source RAG added routing complexity, however it improves the answer quality and source attribution significantly, justifying the engineering overhead.


### Monitoring Architecture

Before I could monitor anything, I had to enable logging at the model serving endpoint. Having to set-up the endpoint to capture request/response data was critical. Then, I needed to create a `RequestLogger` wrapper that wraps around the endpoint to capture:

* Timestamp, request_id, question, answer, context
* Latency
* Status
* Automatic response classification (answer vs. refusal vs. vague vs. empty)

The inference logs table captures all the data writes, becoming the single source of truth for all the monitoring modules. In addition to this, the logger buffers requests in-memory and writes in batches, to avoid write overhead per request.

---

## 🚧 Challenges & Solutions

### Challenge 1: *Translating GDPR enforcement cases into English before Vectorisation*

**Problem:** \
When using `ai_translate()` over large documents, I noticed that the text was cutting off after translation, leaving a significant amount of the document un-translated. It was important to capture the full context of each text, before vectorisation.

**Debugging Process:** \
To solve this, instead of translating a giant text block and then chunking the English output, the operational sequence was inverted to first explode and chunk the raw native text into paragraphs, then pass the small, individual strings to the translation model.

**Solution:** \
This solved the problem as each text sequence was between 100-300 words, ensuring that the output never came close to hitting the translation model's generation limit. Once translated, the translated chunks were combined back into the full English text before embeddings were created.

**What I Learned:** \
Always chunk data to the lowest logical granularity as early in the pipeline as possible to guarantee deterministic, un-truncated outputs.

### Challenge 2: *Initial version of the agent were not retrieving from multiple sources when needed*

**Problem:** \
Early testing showed that the router was only selecting ONE data source per query, even when questions required information from multiple sources. For example, a question like *"What does GDPR Article 17 say about the right to erasure, and what's our company's process for handling deletion requests?"* would only retrieve from statutory law OR policy docs, not both - resulting in incomplete answers.

**Debugging Process:** \
I analysed weaker queries and noticed a pattern: the router prompt asked *"Which sources should you use?"* (singular), and the LangGraph routing logic used `if/elif` conditions that short-circuited after the first match. The tool selection function returned a single string `("gdpr_law")` instead of a list of sources.

**Solution:**
* I redesigned the router prompt to `"...determine which data sources are needed"` and provided clear examples with JSON strings
* I added a parallel retrieval node that iterates over the source list and calls each retrieval tool, before concatenating the results into a single context.
* In a later step, I added in the completeness check node that evaluates the generated answer against the original question, where if missing information is detected, it routes back to the router to trigger retrieval from additional sources.

**What I Learned:** 
* Prompt Engineering can change the LLM routing behaviour
* Self-correction loops are essential. Including the completeness check as a safety net catches edge cases
* By storing `sources_used` in State, it guided the router towards unexplored sources

### Challenge 3: *Evaluation Harness Design and CI/CD Integration*

**Problem:** 
Setting up the design of the evaluation harness was quite abstract having never used GitHub actions before. Additionally, I wasn't sure how to best approach this despite understanding that I needed to develop an automated pipeline new code gets committed, a script will automatically run to check the updates against a golden set of questions, and blocks the commit if it doesn't. I knew my first step was to generate a golden set and then figure out how to trigger a pipeline to check it against.

**Solution:** 
* I used an LLM call to generate 30 representative questions from raw document chunks in Volumes, covering simple lookups, multi-source queries and edge cases. These were stored in a Delta table with expected answers in json format, with expected keywords, and source documents.
* I enabled MLflow tracing on all steps in the agent.
* I created a GitHub Actions workflow which triggers on every push to `main` which spins up a Databricks job to run the evaluation pipeline. This runs the agent against all golden questions, and invokes LLM-as-judge to write the results as an artifact. MLFlow tracing enables me to look at the steps in which the agent took too long, or did not pass a step. Using this golden set evaluation, I ended up iterating on the agent itself a few times.

**What I Learned:** 
* Thinking of evaluation as a part of the infrastructure made the process of improving the agent infinitely less tedious. MLFlow tracing enabled me to easily drill down into specific test cases and see exactly which node was having issues and causing the problems. The artifact of the evaluation of the dataset also enabled me to see whether there were similar issues across test cases.
* Github Actions syntax took me multiple iterations to get right. Genie was very helpful in figuring out the right syntax.

---

## ✅ What Worked Well

**Not using OpenAI Function calling to invoke functions and instead doing so using python.** 

I deliberately chose to implement routing logic in Python rather than relying on OpenAI's function calling API, even though this is a common pattern for agentic systems. I did this because I only have three fixed data sources with predictable routing patterns. The routing logic is deterministic based on query classification, not arbitrary tool selection. Additionally, there is no LLM API cost for function calling with metadata. The trade-off, however is that it is less flexible if I wanted to add dynamic tools, but for a fixed multi-source RAG system, explicit routing was the right choice.

### Best Practices Adopted

* Structuring the project as independent modules rather than a monolithic codebase to allow independent changes. It is a lot more organised and therefore easy to keep a track of changes and updates.
* Building the CI/CD evaluation harness to run via GitHub Actions upon every push request.

---

## 🔄 What I'd Do Differently

### Mistakes & Missteps

**Not writing unit tests while developing**

I didn't write unit tests because this was a personal learning project on Databricks and I figured I was the only person who'd touch the code. This turned out to be a mistake that cost me debugging time later. Whenever I changed the router logic to support multi-source retrieval, I had no way to verify whether each component worked without running the entire agent end-to-end.

I should've written unit tests as I was developing the code base as I went through it. However, I added unit tests retrospectively, but coverage is intentionally limited to core utility functions. Infrastructure-heavy code (Spark pipelines, LangGraph nodes) was excluded as the integration tests and CI/CD eval harness provide better signal for those components.

**What I should have tested:**

Retrieval tools: Each source-specific retrieval function (GDPR law, policy, precedents) with mock vector search responses
Router logic: Given a question, does it correctly identify which sources are needed?
Response classifier: Does the regex correctly classify answers vs. refusals vs. vague responses?
State transitions: Do LangGraph edges route correctly based on state values (quality scores, loop counts)?
Monitoring queries: Do the SQL queries in monitoring modules return expected aggregations?


### Performance/Architecture Improvements

**Latency Reduction Techniques** 

Using MLflow tracing to drill into individual node execution times, average end-to-end latency was reduced from ~20s to ~11s through three targeted changes:

1. **Removed the LLM relevance grader (–2–4s per query).** `edge_evaluate_context` previously called `gpt-4o-mini` to grade whether the retrieved context was relevant to the question. This was replaced with a simple emptiness check — if vector search returned any content above the 0.35 confidence threshold, it is treated as valid. The LLM call was redundant because the vector search already filters by semantic similarity at retrieval time.

2. **Added a `verify_output` passthrough node (–3–5s on complete answers).** Previously, all answers — including complete, well-grounded ones — were unconditionally routed through `regenerate_strict` before the graph could terminate. A lightweight passthrough node was inserted so that answers only enter `regenerate_strict` if the groundedness check actually fails.

3. **Removed a `max_tokens` cap from the groundedness grader.** A `max_tokens=50` limit was briefly applied to reduce generation time on the groundedness grader. This caused the JSON response to be truncated before the closing brace, making `json.loads` fail on every call. The grader then defaulted to `False` (not grounded) on every query, forcing `regenerate_strict` to run on every response and hitting the generation loop limit. Removing the cap restored correct behaviour and eliminated the redundant regeneration cycles.

---

## 🛠️ Technical Details

### Tech Stack

**Core Framework:**
* **LangGraph** - Agentic workflow orchestration with state management
* **OpenAI GPT-4** - Language model for routing, generation, and evaluation
* **LangChain** - RAG components and retrieval tools

**Databricks Platform:**
* **Vector Search** - Managed vector store with auto-sync
* **Model Serving** - Serverless REST API endpoint with scale-to-zero
* **Unity Catalog** - Data governance and model registry
* **Delta Lake** - Storage layer for tables and inference logs
* **MLflow** - Experiment tracking and model versioning

**Infrastructure:**
* **GitHub Actions** - CI/CD pipeline for automated evaluation and FastAPI Deployment
* **Databricks Jobs** - Scheduled query simulation and monitoring
* **Python** - Primary language for agent logic and monitoring
* **FastAPI** - REST API framework wrapping the Databricks agent endpoint with auth and request validation
* **Docker** - Containerising the FastAPI app for deployment
* **GCP Cloud Run** - Severless container hosting that scales to zero when idle
* **Terraform** - IaaC managing Cloud Run, Artifact Registry, and Secret Manager resources
* **GCP Secret Manager** - Secure storage for Databricks credentials and API Keys

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Client / User                          │
└──────────────────────────┬──────────────────────────────────┘
                           │  POST /query + X-API-Key
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              GCP Cloud Run (FastAPI)                         │
│              API key auth · error handling · logging         │
└──────────────────────────┬──────────────────────────────────┘
                           │  Bearer token (from Secret Manager)
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Databricks Model Serving REST API               │
│              (gdpr-agent-staging endpoint)                   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                   LangGraph Agent                            │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  1. Router → Identify data sources needed            │   │
│  │  2. Sequential Retrieval → Query vector indexes       │   │
│  │  3. Generation → Create answer                       │   │
│  │  4. Quality Checks → Completeness & Groundedness     │   │
│  │  5. Self-Correction Loop (if needed)                 │   │
│  └──────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│              Vector Search (3 indexes)                       │
│  • GDPR Law (Articles & Recitals)                           │
│  • Internal Policy Documents                                │
│  • Case Precedents (Enforcement Decisions)                  │
└─────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│           RequestLogger → Delta Table Logging                │
│  (Automatic response classification & metrics capture)      │
└─────────────────────────────────────────────────────────────┘

```

### Key Unity Catalog Tables

| Table | Purpose |
|-------|---------|
| `main.default.gdpr_agent_inference_logs` | Production inference logs with automatic response classification |
| `main.default.gdpr_agent_question_pool` | Golden question set for evaluation |
| `main.default.gdpr_agent_eval_results` | Evaluation results from CI/CD runs |

### Project Structure

```
GDPR-agent/
├── gdpr_agent/          # Core agent logic
│   ├── agent.py         # Wrapper class for LangGraph app
│   ├── graph.py         # LangGraph workflow definition
│   ├── nodes.py         # Agent nodes (router, retrieval, generation)
│   ├── edges.py         # Routing logic based on state
│   ├── tools.py         # Vector search retrieval tools
│   └── state.py         # TypedDict state schema
├── api/                 # FastAPI service
│   ├── main.py          # API endpoints with auth, logging, error handling
│   ├── Dockerfile       # Container definition
│   └── requirements.txt # API dependencies
├── terraform/           # Infrastructure as code
│   ├── main.tf          # GCP resources (Cloud Run, Artifact Registry, Secret Manager)
│   ├── variables.tf     # Input variables
│   ├── outputs.tf       # Output values
│   └── README.md        # Deployment instructions
├── tests/               # Unit and integration tests
│   ├── test_chunk_text.py
│   ├── test_spark_helpers.py
│   ├── test_router.py
│   └── integration/
│       └── test_databricks_connection.py
├── deploy/              # Model deployment scripts
│   ├── deploy_endpoint.py       # Databricks Model Serving setup
│   ├── register_staging.py      # MLflow model registration
│   └── promote_to_production.py # Promotion workflow
├── monitoring/          # Production monitoring modules
│   ├── utils/
│   │   ├── request_logger.py       # RequestLogger class
│   │   ├── response_classifier.py  # Automatic response classification
│   │   └── llm_judge.py           # LLM-as-judge evaluation
│   └── monitors/        # Alert monitors (refusal, error tracking)
├── notebooks/           # Databricks notebooks
│   ├── Query Simulator  # Simulated user traffic generator
│   └── Bootstrap Golden Set  # Golden question generation
└── .github/workflows/   # CI/CD pipelines
    ├── eval.yml          # Automated evaluation on push to main
    ├── staging-deploy.yml # Databricks staging deployment
    └── deploy-api.yml    # Cloud Run API deployment

```

### REST API Implementation

**API Endpoint:**
```
https://gdpr-api-nkfxshjbea-ts.a.run.app
```

**Request Format:**
```python
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "your_api_key"
}

payload = {"question": "What are the requirements for GDPR Article 17?"}

response = requests.post("https://gdpr-api-nkfxshjbea-ts.a.run.app/query", headers=headers, json=payload)
```

**Response Format:**
```json
{
    "answer": "Article 17 establishes the right to erasure..."
}
```

---

## 🚀 Future Improvements

* Multi-turn conversation support with session memory
* Fine-tuned retrieval model on GDPR domain
* Real-time streaming evaluation
* User feedback loop for continuous improvement


---

**Updated:** *30 June 2026*  
