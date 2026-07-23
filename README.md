# ARIA — Automated Resolution Intelligence Assistant

> An AI-powered SAP support ticket intelligence platform that automates consultant triage workflows using fine-tuned transformers, semantic RAG retrieval, gradient boosting, and a LangGraph reasoning agent.

---

## What ARIA Does

SAP support consultants spend significant time on pre-work before any actual consulting begins — classifying incoming tickets, assessing real urgency, and searching for similar resolved cases. ARIA eliminates this pre-work entirely.

When a consultant receives a ticket, ARIA automatically:

1. **Classifies** whether it is an incident or a new development request (DistilBERT)
2. **Labels** the ticket by SAP module, issue type, and complexity (fine-tuned BERT)
3. **Scores** urgency, complexity, and escalation risk (LightGBM)
4. **Retrieves** the 5 most semantically similar resolved tickets from the knowledge base (RAG)
5. **Generates** a structured resolution suggestion with T-codes and investigation steps (LangGraph + LLM)

Results stream progressively — urgency scores appear within 2 seconds, full resolution suggestion within 15 seconds.

---

## Architecture

```
Incoming Ticket (plain text)
        │
        ▼
Preprocessing — clean text, extract SAP error codes if present
        │
        ▼
Step 0 — DistilBERT — Incident or Development Request?
        │                          │
   Incident                New Development
        │                    → Flag and Route Out
        ▼
BERT Multi-Label Classifier
→ Module (FI / MM / SD / PP / BASIS / ABAP / HR)
→ Issue Type (Authorization / Configuration / Master Data / etc.)
→ Complexity (Simple / Medium / Complex)
        │
        ▼
LightGBM Scorer (uses BERT output as features)
→ Urgency Score
→ Complexity Score
→ Escalation Risk (0 to 1)
        │
        ▼
RAG Retrieval — ChromaDB semantic search
→ Top 5 similar resolved tickets
        │
        ▼
LangGraph Agent — gpt-oss-120b via Groq
→ CaseReviewer → Reasoner → TCodeValidator → ResponseFormatter
→ Structured resolution suggestion
```

---

## AI/ML Components

| Component | Technique | Purpose |
|---|---|---|
| Intake Classifier | Fine-tuned DistilBERT | Incident vs development request |
| Ticket Classifier | Fine-tuned BERT | Module, issue type, complexity |
| Risk Scorer | LightGBM | Urgency, complexity, escalation risk |
| Case Retrieval | sentence-transformers + ChromaDB | Semantic similar ticket search |
| Resolution Agent | LangGraph + gpt-oss-120b | Grounded resolution suggestions |

---

## Tech Stack

**Backend** — FastAPI, PostgreSQL, asyncpg, APScheduler  
**ML/AI** — HuggingFace Transformers, LightGBM, sentence-transformers, ChromaDB, LangGraph, Groq API  
**Frontend** — Next.js 14, Tailwind CSS  
**Observability** — LangSmith, Prometheus, Loguru  
**Infrastructure** — Docker, Docker Compose, GitHub Actions  

---

## Project Structure

```
ARIA/
├── backend/
│   ├── pipeline/          # AI processing components
│   ├── ingestion/         # Document ingestion pipeline
│   ├── integration/       # HRMS connector (Nexus)
│   └── models/            # Saved model files (gitignored)
├── data/
│   └── synthetic/         # Synthetic ticket generator and datasets
├── frontend/              # Next.js dashboard
└── notebooks/             # Google Colab training notebooks
```

---

## Data Strategy

Training data is generated using LLM-assisted augmentation — seed examples written across 7 SAP modules and 8 issue types, expanded to 25,000 tickets using Groq API. This is standard practice for enterprise AI where domain-specific labeled data is confidential.

**25,000 synthetic tickets:**
- 15,000 incident tickets — vague non-technical language, some with SAP error codes, some without
- 6,500 new development requests — feature requests, enhancements, new reports
- 3,500 edge cases — ambiguous, cross-module, unusual formats

In production, the knowledge base is replaced by the company's real historical resolution documents via the ingestion pipeline.

---

## Setup

Clone the repository and navigate into it, then:

```
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
cp .env.example .env
```

Fill in your API keys in .env, then:

```
docker compose up
```

---

## Environment Variables

Copy .env.example to .env and fill in your values:

- GROQ_API_KEY — from console.groq.com
- GROQ_AGENT_MODEL — LLM for the resolution agent
- GROQ_FAST_MODEL — LLM for data generation
- DATABASE_URL — PostgreSQL connection string
- LANGSMITH_API_KEY — from smith.langchain.com

---

## HRMS Integration

ARIA runs in two modes:

- **Manual Entry (Demo)** — consultant pastes ticket text directly into ARIA
- **Nexus Auto-Pull (Production)** — ARIA connects to Nexus HRMS via configurable connector, polls for new tickets automatically

The abstract connector interface (backend/integration/base.py) allows any HRMS to integrate by implementing three methods: authenticate(), fetch_new_tickets(), update_ticket_status().

---

## Status

| Component | Status |
|---|---|
| Synthetic data generator | In Progress |
| DistilBERT intake classifier | Pending |
| BERT ticket classifier | Pending |
| LightGBM risk scorer | Pending |
| RAG retrieval pipeline | Pending |
| LangGraph resolution agent | Pending |
| FastAPI backend | Pending |
| Next.js frontend | Pending |
| Docker setup | Pending |

---

## Disclaimer

This project was built as a portfolio project during a SAP consultancy internship. All training data is synthetic — generated using LLM-assisted augmentation. No real client data, SAP credentials, or proprietary information is used anywhere in this project.
