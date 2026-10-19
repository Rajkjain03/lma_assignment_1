import os
import time
import json
import re
import pymupdf  # PyMuPDF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Configuration
BI_ENCODER_MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
PDF_PATH = "corporate_travel_and_expense_policy.pdf"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Init] Compute Device: {device}")

# -------------------------------------------------------------
# STEP 1: DOCUMENT INGESTION & ROBUST CHUNKING
# -------------------------------------------------------------
def extract_chunks_from_pdf(pdf_path: str):
    """
    Extracts text from PDF and segments into structured section chunks.
    Uses regex to avoid splitting on inline numbered lists.
    """
    doc = pymupdf.open(pdf_path)
    chunks = []
    chunk_counter = 1
    sec_re = re.compile(r'^(SECTION\s+\d+[:\s].*|\d+\.\d+\s+[A-Z].*)')

    for pno in range(len(doc)):
        page = doc[pno]
        lines = page.get_text().split('\n')
        current_title = f"Policy Overview (Page {pno+1})"
        current_body = []

        for line in lines:
            line_str = line.strip()
            if not line_str or "CONFIDENTIAL" in line_str or "Document ID" in line_str or line_str.startswith("Page "):
                continue

            if sec_re.match(line_str):
                if current_body:
                    body_text = " ".join(current_body).strip()
                    if len(body_text) > 30:
                        chunks.append({
                            "chunk_id": f"CHUNK_{chunk_counter:02d}",
                            "section_title": current_title,
                            "page": pno + 1,
                            "text": body_text
                        })
                        chunk_counter += 1
                    current_body = []
                current_title = line_str
            else:
                current_body.append(line_str)

        if current_body:
            body_text = " ".join(current_body).strip()
            if len(body_text) > 30:
                chunks.append({
                    "chunk_id": f"CHUNK_{chunk_counter:02d}",
                    "section_title": current_title,
                    "page": pno + 1,
                    "text": body_text
                })
                chunk_counter += 1

    doc.close()
    print(f"[Document Ingestion] Extracted {len(chunks)} chunks from {pdf_path}")
    return chunks

# -------------------------------------------------------------
# STEP 2: MODEL INITIALIZATION
# -------------------------------------------------------------
print(f"[Model Loading] Loading Bi-Encoder: {BI_ENCODER_MODEL_NAME}...")
bi_encoder = SentenceTransformer(BI_ENCODER_MODEL_NAME, device=device)

print(f"[Model Loading] Loading Cross-Encoder: {CROSS_ENCODER_MODEL_NAME}...")
cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL_NAME, device=device)

print(f"[Model Loading] Loading Local LLM: {LLM_MODEL_NAME}...")
llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
llm_model = AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME)
llm_pipe = pipeline("text-generation", model=llm_model, tokenizer=llm_tokenizer, device=device)

def build_vector_store(chunks, model):
    corpus_texts = [f"{c['section_title']}: {c['text']}" for c in chunks]
    embeddings = model.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return embeddings, corpus_texts

# -------------------------------------------------------------
# STEP 3: RETRIEVAL PIPELINES
# -------------------------------------------------------------
def pipeline_1_baseline_retrieval(query: str, chunks: list, corpus_texts: list, chunk_embeddings: np.ndarray, model, top_k_candidates=10, top_k_final=3):
    start_time = time.perf_counter()
    query_embedding = model.encode([query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)[0]
    cosine_sims = np.dot(chunk_embeddings, query_embedding)
    top_indices = np.argsort(cosine_sims)[::-1][:top_k_candidates]
    retrieval_time_ms = (time.perf_counter() - start_time) * 1000

    candidates = []
    for rank, idx in enumerate(top_indices, 1):
        c = dict(chunks[idx])
        c["initial_rank"] = rank
        c["cosine_score"] = float(cosine_sims[idx])
        candidates.append(c)

    top_3_baseline = candidates[:top_k_final]
    return candidates, top_3_baseline, retrieval_time_ms

def pipeline_2_reranking(query: str, candidates: list, cross_encoder_model, top_k_final=3):
    start_time = time.perf_counter()
    pairs = [(query, c["text"]) for c in candidates]
    cross_scores = cross_encoder_model.predict(pairs)
    rerank_time_ms = (time.perf_counter() - start_time) * 1000

    reranked_candidates = []
    for c, score in zip(candidates, cross_scores):
        item = dict(c)
        item["cross_score"] = float(score)
        reranked_candidates.append(item)

    reranked_candidates.sort(key=lambda x: x["cross_score"], reverse=True)
    for rank, item in enumerate(reranked_candidates, 1):
        item["reranked_rank"] = rank
        item["rank_delta"] = item["initial_rank"] - item["reranked_rank"]

    top_3_reranked = reranked_candidates[:top_k_final]
    return reranked_candidates, top_3_reranked, rerank_time_ms

# -------------------------------------------------------------
# STEP 4: REAL LLM GENERATION (QWEN2.5)
# -------------------------------------------------------------
def generate_answer(query: str, context_chunks: list):
    context_str = "\n\n".join([
        f"[{c['section_title']} (Page {c['page']})]\n{c['text']}"
        for c in context_chunks
    ])
    messages = [
        {
            "role": "system",
            "content": "You are a corporate compliance officer. Answer the user question based strictly and exclusively on the policy text provided below. If the policy does not allow something, does not mention an exception, or lists specific restrictions, state them clearly and concisely."
        },
        {
            "role": "user",
            "content": f"POLICY CONTEXT:\n{context_str}\n\nQUESTION: {query}"
        }
    ]
    outputs = llm_pipe(messages, max_new_tokens=150, do_sample=False)
    return outputs[0]["generated_text"][-1]["content"].strip()

# -------------------------------------------------------------
# STEP 5: TEST SUITE EXECUTION
# -------------------------------------------------------------
test_suite = [
    {
        "id": "TEST_01",
        "category": "Exceptions & Distractors",
        "query": "Can staff fly in premium cabins for international client negotiations?",
        "critical_section_keyword": "2.4",
        "description": "General coach mandate (2.1) dominates dense search; 0.5B model misses duration threshold without re-ranking.",
        "is_genuine_failure": True
    },
    {
        "id": "TEST_02",
        "category": "Negation & Distractor Chunk",
        "query": "What items are not covered on expense claims for client business meals?",
        "critical_section_keyword": "5.3",
        "description": "Dense search pulls lodging amenities (3.2) above meal prohibitions (5.3); 0.5B model is misled by distractor.",
        "is_genuine_failure": True
    },
    {
        "id": "TEST_03",
        "category": "Buried Clause: Hotel Laundry",
        "query": "Under what trip length can travelers expense laundry and dry cleaning?",
        "critical_section_keyword": "3.2",
        "description": "Dense retriever buries Section 3.2 at Rank #5 outside Top 3; Cross-Encoder promotes to Rank #1.",
        "is_genuine_failure": True
    },
    {
        "id": "TEST_04",
        "category": "Buried Clause: Ancillary Fees",
        "query": "Are flight cancellation insurance and seat selection fees reimbursable?",
        "critical_section_keyword": "2.3",
        "description": "Dense retriever buries baggage/ancillary fee clause at Rank #8; Cross-Encoder promotes to Rank #2.",
        "is_genuine_failure": True
    },
    {
        "id": "TEST_05",
        "category": "Control: Ground Transport",
        "query": "Can I book an executive airport transfer using Uber or ridesharing apps?",
        "critical_section_keyword": "4.4",
        "description": "Control query evaluating rideshare permissions (4.1) and limousine ban (4.4). Stable in Top 2.",
        "is_genuine_failure": False
    },
    {
        "id": "TEST_06",
        "category": "Control: Admin Penalty",
        "query": "What disciplinary action occurs if expense claims are submitted more than a month late?",
        "critical_section_keyword": "8.5",
        "description": "Control query with unambiguous high semantic overlap to Section 8.5. Top-ranked in both pipelines.",
        "is_genuine_failure": False
    }
]

chunks = extract_chunks_from_pdf(PDF_PATH)
chunk_embeddings, corpus_texts = build_vector_store(chunks, bi_encoder)

experiment_records = []
print("\n" + "="*80)
print("RUNNING RETRIEVAL EXPERIMENTS & FAILURE MODE STRESS TESTS")
print("="*80)

for test in test_suite:
    tid = test["id"]
    tcat = test["category"]
    query = test["query"]
    target_kw = test["critical_section_keyword"]
    is_fail = test["is_genuine_failure"]

    p1_candidates, p1_top3, p1_time = pipeline_1_baseline_retrieval(
        query, chunks, corpus_texts, chunk_embeddings, bi_encoder
    )
    p2_candidates, p2_top3, p2_time = pipeline_2_reranking(
        query, p1_candidates, cross_encoder
    )

    baseline_target = next((c for c in p1_candidates if target_kw in c["section_title"]), None)
    reranked_target = next((c for c in p2_candidates if target_kw in c["section_title"]), None)

    baseline_rank = baseline_target["initial_rank"] if baseline_target else None
    reranked_rank = reranked_target["reranked_rank"] if reranked_target else None
    rank_delta = (baseline_rank - reranked_rank) if (baseline_rank and reranked_rank) else 0

    print(f"\n>>> Running {tid}: [{tcat}]")
    print(f"Query: \"{query}\"")
    print(f"  Target Section: '{baseline_target['section_title'] if baseline_target else 'Unknown'}'")
    print(f"  Baseline Rank: #{baseline_rank} | Re-Ranked Rank: #{reranked_rank} (Shift: {rank_delta:+d})")

    ans1 = generate_answer(query, p1_top3)
    ans2 = generate_answer(query, p2_top3)

    record = {
        "test_id": tid,
        "category": tcat,
        "query": query,
        "target_section": baseline_target["section_title"] if baseline_target else "Not in Top 10",
        "target_page": baseline_target["page"] if baseline_target else None,
        "is_genuine_failure": is_fail,
        "baseline_rank": baseline_rank,
        "baseline_cosine_score": baseline_target["cosine_score"] if baseline_target else None,
        "reranked_rank": reranked_rank,
        "reranked_cross_score": reranked_target["cross_score"] if reranked_target else None,
        "rank_delta": rank_delta,
        "p1_top3": [{"title": c["section_title"], "rank": c["initial_rank"], "score": round(c["cosine_score"], 4)} for c in p1_top3],
        "p2_top3": [{"title": c["section_title"], "rank": c["reranked_rank"], "score": round(c["cross_score"], 4)} for c in p2_top3],
        "baseline_latency_ms": round(p1_time, 2),
        "rerank_latency_ms": round(p2_time, 2),
        "total_p2_latency_ms": round(p1_time + p2_time, 2),
        "answer_1_baseline": ans1,
        "answer_2_reranked": ans2
    }
    experiment_records.append(record)

# -------------------------------------------------------------
# STEP 6: BENCHMARK LATENCY (WARMED 10 ITERATIONS)
# -------------------------------------------------------------
print("\n" + "="*80)
print("PROFILING LATENCIES ACROSS 10 ITERATIONS (WITH 5 WARM-UP RUNS)")
print("="*80)

# Warm-up runs
for test in test_suite:
    for _ in range(5):
        c, _, _ = pipeline_1_baseline_retrieval(test["query"], chunks, corpus_texts, chunk_embeddings, bi_encoder)
        pipeline_2_reranking(test["query"], c, cross_encoder)

latency_benchmarks = []
NUM_WARMED_RUNS = 10

for test in test_suite:
    q = test["query"]
    p1_times = []
    p2_times = []

    for _ in range(NUM_WARMED_RUNS):
        c_p1, _, t1 = pipeline_1_baseline_retrieval(q, chunks, corpus_texts, chunk_embeddings, bi_encoder)
        _, _, t2 = pipeline_2_reranking(q, c_p1, cross_encoder)
        p1_times.append(t1)
        p2_times.append(t2)

    mean_p1 = np.mean(p1_times)
    mean_ce = np.mean(p2_times)
    mean_total = mean_p1 + mean_ce

    latency_benchmarks.append({
        "test_id": test["id"],
        "category": test["category"],
        "mean_p1_ms": round(float(mean_p1), 2),
        "mean_ce_ms": round(float(mean_ce), 2),
        "mean_total_ms": round(float(mean_total), 2),
        "ratio": round(float(mean_total / mean_p1), 2)
    })

# Merge benchmarks into records
for rec, bench in zip(experiment_records, latency_benchmarks):
    rec["benchmarked_p1_ms"] = bench["mean_p1_ms"]
    rec["benchmarked_ce_ms"] = bench["mean_ce_ms"]
    rec["benchmarked_total_ms"] = bench["mean_total_ms"]
    rec["overhead_ratio"] = bench["ratio"]

payload = {
    "device": device,
    "environment": "CPU Benchmark (Local)",
    "bi_encoder": BI_ENCODER_MODEL_NAME,
    "cross_encoder": CROSS_ENCODER_MODEL_NAME,
    "llm_model": LLM_MODEL_NAME,
    "experiment_records": experiment_records,
    "latency_benchmarks": latency_benchmarks
}

with open("experiment_results.json", "w") as f:
    json.dump(payload, f, indent=2)

print("\n[Complete] Saved experiment data to experiment_results.json")

# -------------------------------------------------------------
# STEP 7: GENERATE CHARTS DYNAMICALLY FROM experiment_records
# -------------------------------------------------------------
# Chart 1: Rank Shifts Comparison
fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
test_ids = [r["test_id"] for r in experiment_records]
categories = [f"{r['test_id']}\n({r['category'].split(':')[0]})" for r in experiment_records]
baseline_ranks = [r["baseline_rank"] for r in experiment_records]
reranked_ranks = [r["reranked_rank"] for r in experiment_records]

y = np.arange(len(test_ids))
height = 0.35

rects1 = ax.barh(y + height/2, baseline_ranks, height, label='Bi-Encoder (Pipeline 1)', color='#e74c3c', alpha=0.85, edgecolor='black')
rects2 = ax.barh(y - height/2, reranked_ranks, height, label='Cross-Encoder (Pipeline 2)', color='#2ecc71', alpha=0.85, edgecolor='black')

ax.axvline(x=3.5, color='#e67e22', linestyle='--', linewidth=2, label='Top-3 LLM Context Cutoff')
ax.set_xlabel('Retrieval Rank (Lower is Better / Rank 1 = Top Candidate)', fontsize=11, fontweight='bold')
ax.set_title('Qualitative Retrieval Rank Shifts: Baseline vs. Re-Ranking', fontsize=13, fontweight='bold', pad=15)
ax.set_yticks(y)
ax.set_yticklabels(categories, fontsize=9.5)
ax.invert_yaxis()
ax.legend(loc='lower right', framealpha=0.95, fontsize=9.5)
ax.grid(axis='x', linestyle=':', alpha=0.6)

for rect in rects1:
    width = rect.get_width()
    ax.annotate(f'Rank #{int(width)}', xy=(width, rect.get_y() + rect.get_height() / 2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold', color='#c0392b')

for rect in rects2:
    width = rect.get_width()
    ax.annotate(f'Rank #{int(width)}', xy=(width, rect.get_y() + rect.get_height() / 2),
                xytext=(5, 0), textcoords="offset points", ha='left', va='center', fontsize=9, fontweight='bold', color='#27ae60')

plt.tight_layout()
plt.savefig("rank_shifts_comparison.png")
plt.close()
print("[Chart Generated] rank_shifts_comparison.png")

# Chart 2: Latency Comparison
fig, ax = plt.subplots(figsize=(10, 5.2), dpi=300)
p1_bars = [b["mean_p1_ms"] for b in latency_benchmarks]
ce_bars = [b["mean_ce_ms"] for b in latency_benchmarks]

bar_width = 0.45
x_pos = np.arange(len(test_ids))

b1 = ax.bar(x_pos, p1_bars, bar_width, label='Pipeline 1 Baseline (Dense Retrieval)', color='#3498db', edgecolor='black')
b2 = ax.bar(x_pos, ce_bars, bar_width, bottom=p1_bars, label='Stage 2 Overhead (Cross-Encoder 10 Pairs)', color='#9b59b6', edgecolor='black')

ax.set_ylabel('Execution Runtime (milliseconds)', fontsize=11, fontweight='bold')
ax.set_title(f'Empirical Retrieval Latency Breakdown across 10 Iterations (CPU Benchmark)', fontsize=12, fontweight='bold', pad=15)
ax.set_xticks(x_pos)
ax.set_xticklabels([f"{b['test_id']}\n{b['category'].split(':')[0]}" for b in latency_benchmarks], fontsize=9)

# 35% headroom so bars and annotations never overlap legend
max_tot = max(p1_bars[i] + ce_bars[i] for i in range(len(test_ids)))
ax.set_ylim(0, max_tot * 1.35)
ax.legend(loc='upper left', framealpha=0.95, fontsize=9.5)
ax.grid(axis='y', linestyle=':', alpha=0.6)

for i in range(len(test_ids)):
    tot = p1_bars[i] + ce_bars[i]
    ax.text(x_pos[i], tot + 2.5, f"{tot:.1f} ms", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("latency_comparison.png")
plt.close()
print("[Chart Generated] latency_comparison.png")
