# ARIA — Automated Resolution Intelligence Assistant

> An AI-powered SAP support ticket intelligence platform that automates consultant triage workflows using fine-tuned transformers, gradient boosted decision trees, semantic vector search, and a multi-agent reasoning architecture.

---

## What ARIA Does

ARIA automates the repetitive pre-work that SAP support consultants perform before diagnosing an issue: identifying ticket nature, classifying technical scope, assessing SLA breach risk, and locating relevant historical solutions. When a new ticket arrives, ARIA runs an automated inference pipeline to categorize the ticket, estimate escalation probability, retrieve semantically relevant past resolutions, and assemble a grounded investigation plan for the consultant.

---

## Current Status

| Component | Status | Implementation Details / Artifacts |
|---|---|---|
| Synthetic Data Generator | **Built** | Multi-pass LLM expansion (Gemini / Groq), 25,000 raw tickets |
| Data Cleaning & Verification | **Built** | `data/synthetic/inspect_data.ipynb` (24,252 clean incoming, 18,102 clean resolved) |
| DistilBERT Intake Router | **Built** | Binary classifier (`incident` vs `development`), weights in `backend/models/distilbert_intake/` |
| BERT Multi-Task Classifier | **Built** | Joint module (7), issue type (8), and complexity (3) heads in `backend/models/bert_classifier/` |
| LightGBM Escalation Predictor | **Built** | Binary SLA breach risk model with validation-tuned threshold in `backend/models/lightgbm/` |
| Embedding Pipeline | **Built** | `backend/pipeline/embedder.py` using `all-MiniLM-L6-v2` singleton |
| ChromaDB Vector Store & Ingestion | **Built** | `backend/pipeline/vector_store.py`, `backend/ingestion/pipeline.py`, full pytest suite |
| Preprocessing & Regex Extraction | *Planned* | Text cleaning and SAP error code pattern extraction |
| Model Inference Wrappers | *Planned* | `classifier.py` and `predictor.py` runtime inference services |
| RAG Retriever Service | *Planned* | `retriever.py` query interface wrapping vector search |
| LangGraph Resolution Agent | *Planned* | Multi-agent reasoning graph (Reviewer, Reasoner, T-Code Validator, Formatter) |
| FastAPI Application Server | *Planned* | REST endpoints, streaming responses, and health diagnostics |
| HRMS / Ticketing Connector | *Planned* | Abstract connector interface, mock adapter, and polling worker |
| Next.js Frontend Dashboard | *Planned* | Consultant interface with real-time risk badges and resolution view |

---

## How It Works

```
Incoming Ticket Text
        │
        ▼
Preprocessing (clean text, extract SAP error codes) [Planned]
        │
        ▼
Stage 0: DistilBERT Intake Routing
        │
   ┌────┴────────────────────────┐
   ▼                             ▼
Incident / Edge Case      Development Request
   │                             │
   │                             └─► Route out of support queue
   ▼
Stage 1: BERT Multi-Task Classifier
   ├── Module (FI, MM, SD, PP, BASIS, ABAP, HR)
   ├── Issue Type (Authorization, Configuration, Master Data, etc.)
   └── Complexity (Simple, Medium, Complex)
        │
        ▼
Stage 2: LightGBM Escalation Risk Predictor
   └── Features: BERT predictions + metadata (error codes, length, calendar, history)
   └── Outputs: Calibrated escalation probability + At Risk flag (threshold = 0.3769)
        │
        ▼
Stage 3: ChromaDB Semantic Retrieval
   └── Embeds problem text using sentence-transformers (all-MiniLM-L6-v2)
   └── Retrieves top 5 nearest historical resolved tickets using cosine distance
        │
        ▼
Stage 4: LangGraph Resolution Agent [Planned]
   └── CaseReviewer ──► Reasoner ──► TCodeValidator ──► ResponseFormatter
   └── Grounded resolution plan, verified T-codes, and investigation steps
        │
        ▼
Consultant Dashboard [Planned]
   └── Progressive display: classification & risk score, followed by resolution plan
```

*Note: Initial response within 2 seconds and complete resolution suggestion within 15 seconds are architectural design goals, not verified production benchmarks.*

---

## Experimental Results

The following evaluations were measured on held-out test sets (10% stratified splits, seed 42):

| Model | Task | Target Classes | Test Size | Key Metrics |
|---|---|---|---|---|
| **DistilBERT** | Intake Routing | Incident (0) vs Development (1) | 2,426 | Accuracy: 1.0000<br>Weighted F1: 1.0000<br>Macro F1: 1.0000 |
| **BERT Multi-Task** | Technical Triage | Module (7)<br>Issue Type (8)<br>Complexity (3) | 1,811 | Module F1: 0.9983<br>Issue Type F1: 0.9939<br>Complexity F1: 0.9956<br>Average F1: 0.9959 |
| **LightGBM** | Escalation Risk | No Risk (0) vs At Risk (1) | 1,811 | Accuracy: 0.9409<br>Precision (At Risk): 0.5556<br>Recall (At Risk): 0.6148<br>F1 (At Risk): 0.5837<br>ROC-AUC: 0.9620<br>PR-AUC: 0.6216 (baseline 0.0674)<br>5-Fold CV PR-AUC: 0.6023 ± 0.0346 |

> **Evaluation Caveat**: All metrics are evaluated on synthetic ticket data. The near-perfect transformer scores reflect distinct vocabulary patterns between seed categories. The LightGBM escalation label is derived from synthetic resolution times (`> 120` minutes). In enterprise deployment, all models must be retrained on real operational logs and SLA breach records.

---

## Dataset

Enterprise SAP support data contains confidential business operations and client credentials; no public corpus exists. ARIA generates realistic training tickets using template-driven LLM augmentation:

1. **Seed Engineering**: 58 incident templates across 7 SAP modules and 8 issue types, 18 development request templates, and 15 edge-case templates.
2. **LLM Augmentation**: Scripted multi-pass expansion using Google Gemini (`gemini-3.5-flash-lite`) via an OpenAI-compatible endpoint, requesting 20 variations per prompt.
3. **Structured Outputs**:
   - `incoming_tickets.csv` (25,000 raw tickets): Unstructured user problem reports.
   - `resolved_tickets.csv` (18,500 raw tickets): Structured consultant resolutions (root causes, fixes, T-codes, resolution times, escalation flags). Development requests have no resolved record.
4. **Data Cleaning Pipeline** (`data/synthetic/inspect_data.ipynb`):
   - Removes empty and whitespace-only text fields.
   - Deduplicates identical problem texts (`drop_duplicates(subset=['text'], keep='first')`).
   - Synchronizes resolved records to eliminate orphaned resolutions.
   - Reproducibly shuffles rows (`random_state=42`) while preserving `ticket_id` values.
5. **Clean Corpus Sizes**:
   - `incoming_tickets_clean.csv`: **24,252 rows** (14,702 incident, 6,150 development, 3,400 edge case).
   - `resolved_tickets_clean.csv`: **18,102 rows** (14,702 incident, 3,400 edge case).

---

## Tech Stack

- **Machine Learning & NLP**: PyTorch, HuggingFace Transformers (`distilbert-base-uncased`, `bert-base-uncased`), LightGBM, scikit-learn
- **Embeddings & Vector Store**: `sentence-transformers` (`all-MiniLM-L6-v2`), ChromaDB (in-process persistent client, cosine space)
- **Data Engineering**: pandas, NumPy, python-docx, pypdf
- **Agent Reasoning**: LangGraph, Groq API (`openai/gpt-oss-120b`) *(Planned)*
- **Backend Application**: FastAPI, Uvicorn, PostgreSQL, asyncpg, APScheduler *(Planned)*
- **Frontend Dashboard**: Next.js 14, React, Tailwind CSS *(Planned)*

---

## Repository Structure

```
ARIA/
├── .env.example                      # Template environment variables
├── .gitignore                        # Git ignore patterns
├── LICENSE                           # MIT License
├── README.md                         # Public documentation
├── backend/
│   ├── __init__.py
│   ├── requirements.txt              # Python project dependencies
│   ├── ingestion/
│   │   ├── __init__.py
│   │   └── pipeline.py               # Document & CSV vector ingestion CLI
│   ├── integration/                  # HRMS connector module (planned)
│   │   └── .gitkeep
│   ├── models/                       # Model checkpoints & metadata (gitignored)
│   │   ├── label_mappings.json       # Canonical categorical mappings
│   │   ├── distilbert_intake/        # DistilBERT model weights & configuration
│   │   ├── bert_classifier/          # Fine-tuned multi-task BERT model & tokenizer
│   │   └── lightgbm/                 # LightGBM escalation model & metadata
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── embedder.py               # Singleton embedding model utility
│   │   └── vector_store.py           # ChromaDB client & semantic search interface
│   └── tests/
│       ├── __init__.py
│       ├── conftest.py               # Pytest isolation fixtures
│       ├── test_vector_store.py      # Vector store unit tests
│       └── test_ingestion.py         # Ingestion pipeline unit tests
├── data/
│   └── synthetic/
│       ├── generate_tickets.py       # LLM ticket generator with resume support
│       ├── inspect_data.ipynb        # Data inspection, cleaning, & verification
│       ├── incoming_tickets_clean.csv # Clean incoming dataset (gitignored)
│       └── resolved_tickets_clean.csv # Clean resolved dataset (gitignored)
├── frontend/                         # Next.js web application (planned)
│   └── .gitkeep
└── notebooks/
    ├── .gitkeep
    ├── bert_training.ipynb           # Colab notebook for DistilBERT and BERT
    └── lightgbm_training.ipynb       # Local notebook for LightGBM escalation model
```

---

## Setup and Reproduction

### 1. Environment Setup

Clone the repository and create a Python 3.13 virtual environment from the project root:

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
Copy-Item .env.example .env
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
cp .env.example .env
```

### 2. Configure Environment Variables

Edit `.env` and provide your API keys:
- `GEMINI_API_KEY`: API key from Google AI Studio (for data generation).
- `CHROMA_PERSIST_PATH`: Path to vector database (default: `./chroma_db`).
- `EMBEDDING_MODEL_NAME`: Embedding model (default: `all-MiniLM-L6-v2`).

### 3. Generate and Clean Synthetic Data

Run data generation from the project root:
```bash
python data/synthetic/generate_tickets.py
```
*(If interrupted, the generator automatically resumes from `data/synthetic/generation_progress.json`).*

Execute `data/synthetic/inspect_data.ipynb` in Jupyter Notebook or your IDE to generate `incoming_tickets_clean.csv` and `resolved_tickets_clean.csv`.

### 4. Train Models

- **DistilBERT & BERT**: Upload `notebooks/bert_training.ipynb` to Google Colab (GPU runtime recommended). Execute all cells and download the generated artifacts into `backend/models/`.
- **LightGBM**: Open and execute `notebooks/lightgbm_training.ipynb` locally. Artifacts mirror into `backend/models/lightgbm/`.

### 5. Ingest Knowledge Base

Ingest the cleaned historical resolutions into ChromaDB:
```bash
python -m backend.ingestion.pipeline csv
```

Verify collection health and semantic retrieval:
```bash
python -m backend.ingestion.pipeline verify
```

### 6. Run Unit Tests

Execute the test suite from the project root:
```bash
python -m pytest backend/tests -q
```

---

## Environment Variables

Configured in `.env.example`:

| Variable | Default Value | Description |
|---|---|---|
| `GEMINI_API_KEY` | `your_gemini_api_key_here` | Google Gemini API key for ticket generation |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | Gemini model identifier |
| `GEMINI_BASE_URL` | `https://generativelanguage.googleapis.com/v1beta/openai` | OpenAI-compatible Gemini endpoint |
| `CHROMA_PERSIST_PATH` | `./chroma_db` | Persistent vector store directory (resolves to project root) |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | sentence-transformers embedding model |
| `HRMS_MODE` | `manual` | Ingestion mode (`manual` or `auto`) |
| `POLL_INTERVAL_MINUTES` | `15` | Polling interval for automated ticket ingestion |
| `DATABASE_URL` | `postgresql://aria:aria@localhost:5432/aria` | PostgreSQL connection string *(Planned)* |
| `LANGSMITH_API_KEY` | `your_langsmith_key_here` | Observability tracing key *(Planned)* |
| `LANGSMITH_PROJECT` | `aria` | LangSmith project name *(Planned)* |

---

## Limitations

1. **Synthetic Data**: All models are trained on synthetic tickets generated from seed templates. Real-world SAP support tickets contain messy abbreviations, client-specific jargon, and multi-language text not fully captured here.
2. **Near-Ceiling Classification**: High classification F1 scores reflect clear semantic separation in synthetic seeds rather than guaranteed production generalization.
3. **Escalation Label Construction**: The escalation target was synthetically generated using a threshold on resolution times (`> 120` minutes). In production, models must be trained against real SLA breach logs.
4. **Deterministic Complexity Mapping**: Technical complexity is mapped deterministically from issue types (`COMPLEXITY_MAP`). The complexity prediction head therefore acts as an issue-type proxy rather than an independent feature.

---

## Roadmap

- [x] **Step 1**: Project architecture and configuration setup
- [x] **Step 2**: Synthetic ticket generation (25,000 raw tickets)
- [x] **Step 3**: Data inspection, cleaning, and verification pipeline
- [x] **Step 4**: Fine-tune DistilBERT intake router and BERT multi-task classifier
- [x] **Step 5**: Train LightGBM escalation risk predictor with calibrated threshold
- [x] **Step 6**: SentenceTransformer embedding pipeline
- [x] **Step 7**: ChromaDB vector store and document ingestion pipeline
- [ ] **Step 8**: FastAPI core application server and endpoints
- [ ] **Step 9**: Preprocessing pipeline and SAP error code regex extractors
- [ ] **Step 10**: Classifier and predictor inference integration
- [ ] **Step 11**: RAG semantic retrieval service
- [ ] **Step 12**: LangGraph multi-agent resolution graph with Groq LLM
- [ ] **Step 13**: HRMS / ticketing system connector (abstract base, mock adapter, webhook)
- [ ] **Step 14**: Next.js consultant triage dashboard

---

## License

MIT License

Copyright (c) 2026 Deeksha Gupta

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Disclaimer

This is a portfolio project built to demonstrate end-to-end applied AI systems engineering for enterprise technical support. All ticket datasets, resolution steps, and consultant notes are synthetically generated. No proprietary client data, corporate confidential information, or active SAP system credentials are used.
