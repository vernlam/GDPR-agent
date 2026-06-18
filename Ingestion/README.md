
%md
# GDPR Compliance Document Ingestion Pipeline

Automated pipeline for ingesting, translating, vectorizing, and indexing GDPR compliance documents for RAG-based retrieval.

## Architecture

```text
┌─────────────────────┐
│      1. INGEST      │ Parse markdown & PDFs → Delta tables
│ ├─ GDPR Legislation │
│ ├─ Privacy Policy   │
│ └─ Enforcement PDFs │
└──────┬──────────────┘
       │
       ▼ (enforcement only)
┌─────────────────────┐
│    2. TRANSLATE     │ AI translation (multilingual → English)
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│    3. VECTORIZE     │ Generate embeddings via Foundation Models
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│   4. SYNC INDICES   │ Update Vector Search indices
└─────────────────────┘
```
## Project Structure
```text
compliance_pipeline/
├── config.py                 # Configuration (catalogs, tables, models)
├── setup_vector_indices.py   # One-time: Create vector search infrastructure
├── ingest_documents.py       # Stage 1: Parse & load documents
├── translate_enforcement.py  # Stage 2: Translate enforcement PDFs
├── vectorize_documents.py    # Stage 3: Generate embeddings
├── sync_vector_indices.py    # Stage 4: Sync vector indices
├── run_full_pipeline.py      # Orchestrator: Run all stages
└── utils/
    ├── file_parser.py        # Markdown & PDF parsing logic
    ├── translation_utils.py  # Translation chunking & AI functions
    └── spark_helpers.py      # DataFrame & Delta table utilities
```
## Data Sources

| Source | Format | Volume Path | Output Table |
|--------|--------|-------------|--------------|
| **GDPR Chapter 3** | Markdown | `/Volumes/main/default/compliance_sources/GDPR_Chapter_3.md` | `main.default.gdpr_statutory_legislation` |
| **Privacy Policy** | Markdown | `/Volumes/main/default/compliance_sources/retail_privacy_policy.md` | `main.default.retail_corporate_policy` |
| **Enforcement Tracker** | PDF | `/Volumes/main/default/enforcement_tracker/*.pdf` | `main.default.gdpr_raw_multilingual_precedents` |

## Prerequisites

### 1. Unity Catalog Setup
- Catalog: `main`
- Schema: `default`
- Volumes created at paths above

### 2. Vector Search (One-time setup)
```python
# Run once to create endpoint and indices
python setup_vector_indices.py
```
### 3. Upload Source Files
Upload documents to Unity Catalog Volumes (via UI, CLI, or dbutils):
```python
# Via dbutils in notebook
dbutils.fs.cp("file:/local/path/GDPR_Chapter_3.md", 
              "/Volumes/main/default/compliance_sources/GDPR_Chapter_3.md")
```

## Usage

### Full Pipeline (All Sources)
```python
from run_full_pipeline import run_full_pipeline
# Process all sources: GDPR + Policy + Enforcement
run_full_pipeline()
```
### Update Single Source
```python
# Only update privacy policy (after editing the markdown file)
run_full_pipeline(sources=["policy"])
```
### Skip Stages
```python
# Skip translation (use raw enforcement docs)
run_full_pipeline(skip_translation=True)

# Skip vector index sync
run_full_pipeline(skip_sync=True)
```
### Individual Stages

**Ingestion only:**
```python
from ingest_documents import ingest_all_sources
ingest_all_sources(sources=["gdpr", "policy"])
```
**Vectorization only:**
```python
from vectorize_documents import vectorize_all_sources
vectorize_all_sources(sources=["policy"])
```
**Index sync only:**
```python
from sync_vector_indices import sync_all_indices
sync_all_indices(sources=["policy"])
```

## Workflow for Updates
### Scenario 1: Privacy Policy Updated
```bash
# 1. Upload new policy file to Volume
# 2. Run pipeline for policy only
python run_full_pipeline.py --sources policy
```
**What happens:**

- ✅ Policy table overwritten with new data
- ✅ Policy embeddings regenerated
- ✅ Policy vector index synced
- ✅ GDPR and enforcement remain unchanged

### Scenario 2: New Enforcement PDFs Added
```bash
# 1. Upload new PDFs to enforcement_tracker volume
# 2. Run full pipeline for enforcement
python run_full_pipeline.py --sources enforcement
```
**Pipeline stages:**

1. Ingests new PDFs (overwrites table)
2. Translates documents to English
3. Generates embeddings
4. Syncs vector index

### Scenario 3: First-time Setup
```bash
# 1. Upload all source files
# 2. Create vector search infrastructure
python setup_vector_indices.py

# 3. Run full pipeline
python run_full_pipeline.py
```

## Configuration
Edit `config.py` to change:

- Catalog/schema names
- Volume paths
- Table names
- Embedding model (`databricks-gte-large-en`)
- Vector search endpoint name
- Translation settings
## Key Features
### Incremental Processing
- Only process changed sources
- Other tables remain untouched
- No unnecessary recomputation
### Idempotent Operations
- Safe to re-run pipelines
- Overwrite mode ensures consistency
- Change Data Feed tracks history
### Error Handling
- Each source processes independently
- Failures don't crash entire pipeline
- Detailed logging at each stage
## Output Tables

|Stage|Source|Table|Schema|
|:---|:---|:---|:---|
|Ingestion|GDPR|`gdpr_statutory_legislation`|article_id, text_content, ...|
|Ingestion|Policy|`retail_corporate_policy`|section_id, text_content, ...|
|Ingestion|Enforcement|`gdpr_raw_multilingual_precedents`|source_file_name, full_document_text, ...|
|Translation|Enforcement|`gdpr_translated_precedents`|..., full_document_text_translated|
|Vectorization|All|`*_embeddings`|..., embedding (ARRAY<FLOAT>)|
## Vector Search Indices

|Index Name|Source Table|Primary Key|
|:---|:---|:---|
|`gdpr_law_vector_index`|`gdpr_statutory_legislation_embeddings`|`chunk_id`|
|`privacy_policy_vector_index`|`retail_corporate_policy_embeddings`|`chunk_id`|
|`enforcement_tracker_vector_index`|`gdpr_precedents_embeddings`|`source_file_name`|

## Monitoring
### Check pipeline status:
```python
from sync_vector_indices import check_all_indices_status
check_all_indices_status()
```
### Verify table row counts:
```sql
SELECT COUNT(*) FROM main.default.gdpr_statutory_legislation;
SELECT COUNT(*) FROM main.default.retail_corporate_policy;
SELECT COUNT(*) FROM main.default.gdpr_translated_precedents;
```
### Check index health:
```python
from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient()
index = vsc.get_index("gdpr_rag_endpoint", "main.default.gdpr_law_vector_index")
print(index.describe())
```
## Troubleshooting
**Import errors in notebooks:**
- Make sure all files are in the same directory
- Verify `utils/__init__.py` exists

**AI function errors:**

- Verify `ai_generate_embedding()` is available in your workspace
- Check embedding model name in `config.py`

**File not found errors:**

- Verify files uploaded to correct Volume paths
- Use `dbutils.fs.ls("/Volumes/main/default/compliance_sources/")` to check

**Vector index not syncing:**
- Ensure Change Data Feed is enabled: `DESCRIBE DETAIL <table>`
- Check endpoint is ONLINE: `check_all_indices_status()`

## Notes
- **Setup scripts (98, 99)** were one-time bulk operations and don't need to be re-run
- **Translation** only applies to enforcement PDFs (GDPR/policy are already in English)
- **Embedding model** must support the `ai_generate_embedding()` SQL function
- **Vector indices** use Delta Sync with triggered pipeline type for manual control
---
**Last Updated**: June 2026

**Databricks Runtime**: Serverless (Python 3.10+)