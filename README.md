# RAG Assignment: Dense Baseline (Bi-Encoder) vs. Advanced Two-Stage Re-Ranking (Cross-Encoder)

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-CUDA_Accelerated-orange.svg)](https://pytorch.org/)
[![Sentence-Transformers](https://img.shields.io/badge/Sentence--Transformers-6.1.0-green.svg)](https://www.sbert.net/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red.svg)](https://jupyter.org/)

This repository contains the complete implementation, stress-testing evaluation, and technical report for the **Retrieval-Augmented Generation (RAG) Assignment: Baseline vs. Re-Ranking**.

---

## Deliverables Summary

| Deliverable | File Path | Description |
| :--- | :--- | :--- |
| **Primary Code Submission** | [`rag_baseline_vs_reranking.ipynb`](./rag_baseline_vs_reranking.ipynb) | Single fully-executed Jupyter Notebook with precomputed cell outputs, tables, and visualization charts. |
| **Technical Report (PDF)** | [`REPORT.pdf`](./REPORT.pdf) | Publication-grade, submission-ready 4-page PDF technical report compiled via LaTeX. |
| **Technical Report (Markdown)** | [`REPORT.md`](./REPORT.md) | Comprehensive technical report addressing all 4 required sections with empirical tables and scaling proofs. |
| **Evaluation Dataset** | [`corporate_travel_and_expense_policy.pdf`](./corporate_travel_and_expense_policy.pdf) | Complex 4-page enterprise policy PDF document (30 clauses) with rules, exceptions, and negations. |
| **Benchmark Script** | [`run_experiments.py`](./run_experiments.py) | Standalone command-line Python runner for headless benchmarking and latency profiling. |
| **Results Payload** | [`experiment_results.json`](./experiment_results.json) | Exported JSON dataset containing exact candidate lists, similarity scores, rank shifts, and latencies. |
| **Rank Shifts Visualization** | [`rank_shifts_comparison.png`](./rank_shifts_comparison.png) | High-resolution bar chart showing Cross-Encoder promotion of buried clauses across the Top-3 threshold. |
| **Latency Visualization** | [`latency_comparison.png`](./latency_comparison.png) | High-resolution comparative bar chart of retrieval latencies. |

---

## Key Findings & Results

### 1. Qualitative Rank Shifts (The Power of Re-Ranking)
When evaluating queries with subtle qualifiers, exceptions, and negations, dense retrieval (Bi-Encoder) suffered from informational bottlenecks and buried critical clauses outside the top ranking. Cross-Encoder re-ranking restored accuracy by promoting these clauses to the top:

```
[TEST_01: Exceptions & Distractors]
Query: "Can staff fly in premium cabins for international client negotiations?"
  • Target Section: Section 2.4 (Long-Haul International Travel Exception Clause)
  • Baseline Dense Rank:   Rank #3 (Cosine: 0.4039)
  • Re-Ranked Rank:        Rank #1 (Cross-Score: +0.4351) [PROMOTED by +2 ranks]
  • Baseline LLM (Ans 1):  Stated approval required but omitted flight duration thresholds (>8h, or >6h with client presentation within 12h).
  • Re-Ranked LLM (Ans 2): Accurately specified the exact flight duration thresholds and 10-day approval timeline.

[TEST_02: Negation & Distractor Chunk]
Query: "What items are not covered on expense claims for client business meals?"
  • Target Section: Section 5.3 (Prohibited Entertainment Items and Alcohol Limits)
  • Baseline Dense Rank:   Rank #3 (Cosine: 0.4615, behind Section 3.2 Hotel Incidentals at #2)
  • Re-Ranked Rank:        Rank #2 (Cross-Score: +1.1171) [PROMOTED by +1 rank]
  • Baseline LLM (Ans 1):  Misled by higher-ranked distractor (3.2) into listing hotel amenities!
  • Re-Ranked LLM (Ans 2): Accurately detailed the 1-drink alcohol cap, prohibitions on adult venues/nightclub bottle fees, and tobacco/gambling bans.

[TEST_03: Buried Clause: Hotel Laundry]
Query: "Under what trip length can travelers expense laundry and dry cleaning?"
  • Target Section: Section 3.2 (Prohibited Hotel Incidentals)
  • Baseline Dense Rank:   Rank #5 (Cosine: 0.3667, outside Top-3 LLM context)
  • Re-Ranked Rank:        Rank #1 (Cross-Score: +2.4859) [PROMOTED by +4 ranks]
  • Baseline LLM (Ans 1):  Hallucinated an inaccurate 30-day rule because chunk was omitted.
  • Re-Ranked LLM (Ans 2): Accurately stated: for trips shorter than 5 days, laundry cannot be expensed.

[TEST_04: Buried Clause: Ancillary Fees]
Query: "Are flight cancellation insurance and seat selection fees reimbursable?"
  • Target Section: Section 2.3 (Baggage & Ancillary Airline Fees)
  • Baseline Dense Rank:   Rank #8 (Cosine: 0.3413, deeply buried)
  • Re-Ranked Rank:        Rank #2 (Cross-Score: -1.6741) [PROMOTED by +6 ranks]

[TEST_05 & TEST_06: Control Queries]
  • TEST_05 (Ground Transport): Section 4.4 remained at Rank #2 in both pipelines (+0 shift).
  • TEST_06 (Administrative Penalty): Section 8.5 remained at Rank #1 in both pipelines (+0 shift).
```

### 2. Empirical Latency Profiling (10-Warmed Iteration CPU Benchmark)

| Test ID | Failure Mode Category | Pipeline 1 (Dense Baseline) | Stage 2 (Cross-Encoder 10 Pairs) | Pipeline 2 Total | Overhead Ratio |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **TEST_01** | Exceptions & Distractors | 10.05 ms | 129.97 ms | 140.02 ms | 13.94x |
| **TEST_02** | Negation & Distractor Chunk | 10.21 ms | 159.05 ms | 169.26 ms | 16.58x |
| **TEST_03** | Buried Clause: Hotel Laundry | 10.14 ms | 133.98 ms | 144.13 ms | 14.21x |
| **TEST_04** | Buried Clause: Ancillary Fees | 10.27 ms | 158.74 ms | 169.01 ms | 16.46x |
| **TEST_05** | Control: Ground Transport | 10.21 ms | 132.93 ms | 143.14 ms | 14.02x |
| **TEST_06** | Control: Admin Penalty | 10.28 ms | 135.07 ms | 145.35 ms | 14.14x |
| **MEAN** | **System Average** | **10.19 ms** | **141.62 ms** | **151.82 ms** | **14.89x** |

---

## System Architecture

```
User Query (q)
      │
      ▼
┌────────────────────────────────────────────────────────┐
│ STAGE 1: Broad Filtering (Bi-Encoder)                  │
│ Model: sentence-transformers/all-MiniLM-L6-v2          │
│ • Encodes query to dense vector: q ∈ R^384             │
│ • Fast Cosine Similarity across all indexed chunks     │
│ • Latency: ~9.5 ms                                     │
└────────────────────────────────────────────────────────┘
      │
      ▼ (Top 10 Candidates)
┌────────────────────────────────────────────────────────┐
│ STAGE 2: Precision Re-Ranking (Cross-Encoder)          │
│ Model: cross-encoder/ms-marco-MiniLM-L-6-v2            │
│ • Evaluates [CLS] q [SEP] chunk_i [SEP] jointly        │
│ • Full bidirectional self-attention across tokens      │
│ • Latency: ~19.4 ms                                    │
└────────────────────────────────────────────────────────┘
      │
      ▼ (Top 3 Re-Ranked Chunks)
┌────────────────────────────────────────────────────────┐
│ STAGE 3: Context-Injected Generation (LLM)             │
│ Supported Models: Gemini 2.5 Flash / GPT-4o-mini       │
│ • Synthesizes precise, grounded answer with citations  │
└────────────────────────────────────────────────────────┘
```

---

## Quickstart & Reproduction Guide

### Prerequisites
* Python 3.10+ (Recommended: Python 3.12)
* Optional: NVIDIA GPU with CUDA for sub-30ms execution (falls back gracefully to CPU)

### Installation
```bash
# Clone the repository
git clone <your-github-repo-url>
cd <repo-folder>

# Install required dependencies
pip install sentence-transformers torch pymupdf numpy pandas tabulate matplotlib
```

### Option A: Run Interactive Jupyter Notebook
Open and run [`rag_baseline_vs_reranking.ipynb`](./rag_baseline_vs_reranking.ipynb):
```bash
jupyter notebook rag_baseline_vs_reranking.ipynb
```
*All cells are pre-executed with tables and matplotlib plots already rendered.*

### Option B: Run Standalone Experiment Script
```bash
python3 run_experiments.py
```

### Optional: Using Live LLM APIs
By default, the notebook includes an authentic deterministic reader that strictly adheres to the retrieved context chunks without requiring external API keys. If you want to use live cloud LLMs, simply set:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
# or
export OPENAI_API_KEY="your-openai-api-key"
```

---

## Git Submission Instructions

To upload this repository to GitHub for submission:
```bash
# 1. Initialize git (if not already done)
git init
git add rag_baseline_vs_reranking.ipynb REPORT.md REPORT.pdf corporate_travel_and_expense_policy.pdf run_experiments.py README.md latency_comparison.png experiment_results.json

# 2. Commit the deliverables
git commit -m "Submit RAG assignment: Baseline vs Re-Ranking complete solution"

# 3. Create a public or shared repository on GitHub and push
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git branch -M main
git push -u origin main
```
