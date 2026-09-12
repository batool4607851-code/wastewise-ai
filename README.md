[README.md](https://github.com/user-attachments/files/32139670/README.md)
# WasteWise AI

## Manufacturing Loss & Rework Intelligence

WasteWise AI is a factory-agnostic, evidence-based decision-support application for food manufacturing. It combines deterministic manufacturing analytics, uploaded SOP/document retrieval, optional visual evidence, and AI reasoning to help production and quality teams investigate material loss, quality deviations, rework, recovery, and final loss.

> **Hackathon MVP:** Upload → Analyze → Ask → Investigate

## Problem

Food-manufacturing loss information is often fragmented across spreadsheets, CSV files, PDFs, QC reports, waste logs, images, and employee knowledge. Spreadsheets can calculate numbers but still require interpretation; generic chatbots can explain documents but do not reliably perform deterministic operational calculations; document-only RAG cannot explain production patterns; image-only analysis cannot establish causality.

WasteWise AI brings these evidence types together while keeping deterministic calculations separate from AI interpretation.

## MVP Features

- CSV, XLSX, and XLS factory-data upload
- PDF/TXT factory-document upload
- JPG/JPEG/PNG/WebP visual-evidence upload
- Validation and dynamic data-field mapping
- Deterministic production, loss, loss-rate, cost, rework, and recovery analytics
- Product, line, shift, category/event-type, and reason analysis
- Simple explainable anomaly indicators
- SOP/QC/rework/specification RAG
- Multimodal Grok/xAI image assessment
- Natural-language AI Analyst
- Cross-source reasoning using data + documents + visual findings
- Investigation findings with evidence, hypotheses, confidence, and recommended next checks
- Human-verification guardrails

## Factory-Agnostic Design

The application is intentionally not hardcoded to a particular factory, product, production line, shift, waste reason, quality parameter, SOP, or customer dataset.

It is designed for food-manufacturing environments such as sauces, dairy, beverages, bakery, snacks, frozen foods, canned foods, and other food operations. Factory evidence is supplied at runtime.

## Core Data Model

The internal canonical model may include:

- Date
- Product
- Batch
- Production Line
- Shift
- Production Quantity
- Event Type
- Event Reason
- Affected Quantity
- Loss Quantity
- Rework Quantity
- Recovered Quantity
- Final Loss Quantity
- Disposition
- Unit Cost
- Rework Cost
- Loss Cost
- Quality Parameter
- Quality Value
- Quality Status

Fields may be unavailable in a factory dataset. WasteWise AI must report unavailable information rather than inventing values.

### Event-Type Distinction

WasteWise AI does not treat every quality deviation as waste.

The intended material-flow distinction is:

`Quality Deviation → Hold → Rework → Recovery → Potential Final Loss`

Where supported by uploaded data, the application separates Process Loss, Packaging Loss, Quality Deviation, Raw Material Issue, Equipment/Process Issue, Other, Rework, Recovery, and Final Loss.

## Architecture

```text
Factory User
    |
    +-- Factory Data
    +-- Factory Documents
    +-- Optional Images
    |
    v
Data Engine -----> Deterministic Analytics
    |
    +-------------> RAG / Document Evidence
    |
    +-------------> Vision / Visual Observations
    |
    v
Unified AI Context
    |
    v
Grok / xAI
    |
    v
WasteWise AI Analyst
    |
    v
Loss + Rework + Investigation Intelligence
    |
    v
Human Decision Maker
```

### Responsibility Boundaries

**Python/Pandas/NumPy** handles validation, mapping, normalization, totals, aggregations, rates, cost calculations, product/line/shift comparisons, trends, rework/recovery calculations, and explainable anomaly indicators.

**RAG** handles SOPs, QC procedures, rework procedures, specifications, operating procedures, and investigation procedures.

**Vision** handles observable visual issues, packaging/product defects, obvious visual abnormalities, and structured visual observations.

**Grok/xAI** handles natural-language questions, explanations, evidence synthesis, hypotheses, recommendations, and cross-source reasoning.

## Guardrails

WasteWise AI is decision support, not an autonomous factory-control system.

The application must:

- Separate calculated facts from AI inference.
- Separate visual observations from possible causes.
- Ground SOP claims in retrieved documents.
- State when evidence is insufficient.
- Never claim an image proves root cause.
- Never invent costs, procedures, measurements, statistics, or missing fields.
- Never autonomously approve release/reject/disposal/quarantine/rework.
- Never control machinery or factory settings.
- Require human verification for recommendations.

Preferred investigation language includes: “potential contributor”, “investigation lead”, “pattern suggests”, and “consider investigating”.

## Financial Impact

When required fields exist:

`Estimated Loss Cost = Final Loss Quantity × Unit Cost`

Illustrative scenarios:

`Potential Monthly Saving = Estimated Loss Cost × Illustrative Reduction`

`Potential Annual Saving = Potential Monthly Saving × 12`

Illustrative scenarios must be labelled as illustrative and are not verified savings.

## RAG Flow

```text
PDF/TXT → Text Extraction → Cleaning → Chunking → Embeddings → FAISS / Vector Store → Top Relevant Chunks → Grok → Answer + Source Metadata
```

Supporting source metadata should include document filename and page/section where available.

## Visual Analysis Flow

```text
Image → Grok Multimodal Assessment → Structured Finding
                         ├── visible_issue
                         ├── category
                         ├── severity
                         ├── possible_causes
                         ├── recommended_investigation
                         └── confidence
```

Visual findings are observations and investigation leads, not confirmed root causes.

## Repository Structure

```text
wastewise-ai/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── src/
│   └── __init__.py
└── tests/
    └── test_smoke.py
```

Later build stages add the data engine, analytics, anomaly detection, PDF/RAG pipeline, Grok client, image analyzer, reasoning layer, and their tests.

## Installation

### 1. Clone

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd wastewise-ai
```

### 2. Virtual environment

```bash
python -m venv .venv
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Environment Variables

Copy `.env.example` to `.env` and add your xAI key locally. Never commit `.env`.

```text
XAI_API_KEY=your_api_key_here
XAI_MODEL=grok-4.6
```

For Streamlit deployment, store the key in deployment secrets instead of source control.

## Run

```bash
streamlit run app.py
```

The app starts without a preloaded factory dataset.

## Testing

```bash
pytest -q
```

## Deployment

Recommended simple path: push to GitHub, deploy on Streamlit Community Cloud where practical, and configure `XAI_API_KEY` as a deployment secret. Never expose API keys in source, README files, logs, screenshots, or UI.

## Error Handling Requirements

The completed application should gracefully handle unsupported file types, missing required fields, ambiguous mappings, invalid numeric data, nulls, duplicates, unparseable PDFs, no relevant RAG result, invalid/oversized images, missing API keys, xAI failures, and unavailable cost/rework/recovery data.

Unavailable metrics must be reported as unavailable rather than silently shown as zero.

## Scope Protection

The MVP excludes IoT/live sensors, ERP integration, machine control, custom CV training, YOLO/custom CV pipelines, mobile apps, payments/marketplace, digital twins, advanced forecasting, autonomous factory agents, automated QC release, automated disposal decisions, and complex multi-factory administration/authentication.

## Hackathon Demo

```text
Upload factory data
      ↓
Validate + map
      ↓
Analyze loss/rework/recovery
      ↓
Upload SOP/QC/rework evidence
      ↓
Ask WasteWise
      ↓
Add optional image
      ↓
Cross-analyze evidence
      ↓
Identify investigation priority
      ↓
Review evidence + hypothesis + next check
```

Synthetic demonstration data must be clearly labelled as synthetic/demo data.

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- OpenPyXL
- xlrd
- Plotly
- PyPDF
- FAISS
- Sentence Transformers / configured embedding provider
- Pydantic
- python-dotenv
- Requests
- xAI/Grok
- pytest
- GitHub

## Roadmap

- V1 — Loss + Rework
- V2 — Predictive Loss
- V3 — Yield
- V4 — Inventory / Expiry
- V5 — Quality
- V6 — Computer Vision
- V7 — Production Optimization
- V8 — Digital Twin
- V9 — AI Factory Operations Agent

## Project Principle

**WORKING > SIMPLE > RELIABLE > DEMONSTRABLE > COMPLEX**

The project prioritizes a reliable end-to-end hackathon prototype over unnecessary infrastructure or features outside the MVP.
