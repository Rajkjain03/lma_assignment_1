# RAG Assignment 1: Dense Baseline vs. Cross-Encoder Re-Ranking

**Course:** Large Language Models & AI Systems (LMA)  
**Author:** Raj K. Jain  
**GitHub Repository:** [https://github.com/Rajkjain03/lma_assignment_1](https://github.com/Rajkjain03/lma_assignment_1)  
**Date:** October 2026  

---

## 1. Architecture Comparison: Bi-Encoder vs. Cross-Encoder

In Retrieval-Augmented Generation (RAG), retrieval identifies relevant document passages from a corpus given a user query. Two distinct neural architectures are commonly used: **Bi-Encoders** and **Cross-Encoders**.

```
BI-ENCODER (Independent Encoding / Dense Vectors):
  Query (q)  -------> [ Transformer ] -------> Dense Vector u (384-d)  \
                                                                        ---> Cosine Similarity: cos(u, v)
  Chunk (c)  -------> [ Transformer ] -------> Dense Vector v (384-d)  /
  (Passage embeddings are computed once offline and indexed into FAISS / Vector DB)

CROSS-ENCODER (Joint Token-Level Multi-Head Self-Attention):
  [CLS] + Query tokens + [SEP] + Chunk tokens + [SEP]
                         │
                         ▼
             ┌────────────────────────┐
             │ Layer 1 Self-Attention │  <-- Every query token attends
             ├────────────────────────┤      directly to every chunk token
             │ ... All Layers ...     │
             ├────────────────────────┤
             │ Classification Head    │
             └────────────────────────┘
                         │
                         ▼
             Relevance Logit S(q, c) in (-inf, +inf)
  (Cannot be precomputed; requires a live joint forward pass for each candidate pair)
```

### Core Mechanical Differences

1. **Token Interaction & Representation Bottleneck:**
   * **Bi-Encoder (`all-MiniLM-L6-v2`):** Encodes the query $q$ and candidate chunk $c$ independently. Each text is compressed via mean-pooling into a single dense vector $\vec{u}, \vec{v} \in \mathbb{R}^{384}$. At retrieval time, similarity is measured purely through cosine similarity:
     $$\cos(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\|_2 \|\vec{v}\|_2}$$
     Because the two texts are encoded in isolation, query tokens never attend to chunk tokens. Squeezing an entire 200–300 word policy paragraph into 384 floating-point numbers creates an informational bottleneck where conditional exceptions, negative constraints, and subtle qualifiers are easily blurred.
   * **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`):** Receives the concatenated input sequence $[\text{CLS}] \circ q \circ [\text{SEP}] \circ c \circ [\text{SEP}]$ into all transformer layers. Through multi-head self-attention:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
     every query token directly compares against every token in the passage across all hidden layers. This allows the model to capture deep semantic dependencies, identify modifier words ("except when", "strictly prohibited"), and evaluate whether the chunk truly answers the specific intent of the query.

2. **Computational Complexity & Indexability:**
   * **Bi-Encoder:** Enables offline precomputation. For a corpus of $N$ chunks, all $N$ embeddings are computed once and stored in an Approximate Nearest Neighbor (ANN) index (e.g., HNSW or IVF in FAISS). At runtime, searching $1,000,000$ chunks requires just 1 forward pass for the query and a fast vector scan taking 5–10 milliseconds ($O(d \cdot \log N)$).
   * **Cross-Encoder:** Cannot be precomputed because the joint representation depends on both the query and the chunk simultaneously. Scoring $N$ candidate chunks requires $N$ full forward passes through the transformer with quadratic token complexity $O(N \cdot L^2)$.

### Summary Comparison

| Metric | Bi-Encoder (`all-MiniLM-L6-v2`) | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| :--- | :--- | :--- |
| **Model Topology** | Dual independent encoder towers | Single joint concatenated transformer |
| **Token Interaction** | Zero cross-attention across $q$ and $c$ | Full all-to-all cross-attention across all layers |
| **Relevance Output** | Geometric cosine similarity $\cos(\vec{u}, \vec{v})$ | Learned classification head logit $S(q, c)$ |
| **Offline Indexing** | Yes (vectors stored in FAISS / Vector DB) | No (requires live forward pass per pair) |
| **Search Complexity** | Sub-linear: $O(d \cdot \log N)$ with ANN | Linear: $O(N \cdot L^2)$ with $N$ forward passes |
| **Linguistic Precision** | Fast broad matching; prone to losing qualifiers | State-of-the-art; captures syntax and negations |

---

## 2. Qualitative & Rank Shift Analysis

### Evaluation Setup & Document Corpus
We evaluated both pipelines over a complex 4-page enterprise policy document: *"Acme Global Travel, Expense and Procurement Policy (2026)"* containing 30 structured clauses.

> **Evaluation Corpus Note:** This document is a synthetic enterprise compliance manual authored to stress-test RAG retrieval boundaries. It contains general rules with distant exception clauses, negative constraints, and high lexical overlap distractors.

We evaluated four queries:
* **TEST_01 & TEST_02:** Genuine retrieval degradation cases where the Bi-Encoder blurs context and re-ranking restores accuracy.
* **TEST_03 & TEST_04:** Control queries where both pipelines agree on the top candidates.

### Qualitative Retrieval Rank Shifts

| Test ID | Category | Target Policy Section | Bi-Encoder Rank | Cross-Encoder Rank | Net Shift | Target In Top 3? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **TEST_01** | Exceptions & Corner Cases | Sec 2.4 (Long-Haul Flight Exception) | Rank #3 (score: 0.4039) | **Rank #1 (score: +0.4351)** | **+2** | Promoted to #1 |
| **TEST_02** | Negation & Constraints | Sec 5.3 (Prohibited Items & Alcohol Limits) | Rank #3 (score: 0.4615) | **Rank #2 (score: +1.1171)** | **+1** | Promoted to #2 |
| **TEST_03** | Control: Ground Transport | Sec 4.4 (Limousine & Luxury Car Ban) | Rank #2 (score: 0.4516) | Rank #2 (score: +1.1141) | +0 | Retained in Top 2 |
| **TEST_04** | Control: Admin Penalty | Sec 8.5 (Late Submission Forfeiture) | Rank #1 (score: 0.6552) | Rank #1 (score: +6.1669) | +0 | Retained at Rank #1 |

![Qualitative Rank Shifts Comparison](rank_shifts_comparison.png)

---

### Detailed Case Study 1: Exceptions & Corner Cases (TEST_01)
* **Query:** *"Can staff fly in premium cabins for international client negotiations?"*
* **Target Truth:** Section 2.4 allows Business Class for continuous non-stop international flights exceeding 8 hours, or exceeding 6 hours if the traveler is scheduled to deliver a client presentation or attend executive contract negotiations within 12 hours of arrival, subject to VP and CFO approval.
* **Retrieval Comparison:**
  * **Pipeline 1 (Baseline Bi-Encoder):**
    * Rank #1: Section 2.1 (*Standard Cabin Class Mandate*) — Score: 0.5428
    * Rank #2: Section 1.3 (*Designated Corporate Travel Booking Engine*) — Score: 0.4127
    * Rank #3: Section 2.4 (*Long-Haul International Travel Exception Clause*) — Score: 0.4039
  * **Pipeline 2 (Cross-Encoder Re-Ranked):**
    * **Rank #1 (Promoted +2):** Section 2.4 (*Long-Haul International Travel Exception Clause*) — Score: +0.4351
    * Rank #2: Section 2.1 (*Standard Cabin Class Mandate*) — Score: -5.7229
    * Rank #3: Section 1.2 (*Mandatory Pre-Trip Travel Authorization*) — Score: -7.9689
* **LLM Answer Comparison (Qwen2.5-0.5B-Instruct):**
  * **Baseline Answer 1:**
    > *"Yes, staff can fly in premium cabins for international client negotiations. However, they need to obtain prior written concurrence from both the Division Vice President and the Chief Financial Officer at least 10 business days before ticketing."*
    * **Observation:** The baseline model omitted the critical flight duration criteria (must exceed 8 hours, or 6 hours with a presentation within 12 hours) because Section 2.1 dominated the prompt context.
  * **Re-Ranked Answer 2:**
    > *"Yes, staff can fly in premium cabins for international client negotiations under the long-haul international travel exception clause. The policy allows this exception when traveling on continuous non-stop international flights with flight durations exceeding 8 hours or 6 hours if the traveler is scheduled to deliver a client presentation or attend executive contract negotiations within 12 hours of flight arrival. Employees need prior written concurrence from both the Division Vice President and the Chief Financial Officer no later than 10 business days before booking their tickets."*
    * **Observation:** Re-ranking promoted Section 2.4 to the #1 position, providing the model with the exact hourly thresholds and full compliance requirements.

---

### Detailed Case Study 2: Negation & Constraints (TEST_02)
* **Query:** *"What items are not covered on expense claims for client business meals?"*
* **Target Truth:** Section 5.3 strictly limits alcohol to 1 drink per diner, prohibits solo alcohol, and bans nightclub bottle service, adult venues, tobacco, and gambling losses.
* **Retrieval Comparison:**
  * **Pipeline 1 (Baseline Bi-Encoder):**
    * Rank #1: Section 5.1 (*Client Entertainment Hosting*) — Score: 0.5936
    * Rank #2: Section 3.2 (*Prohibited Hotel Incidentals and Personal Amenities*) — Score: 0.5562
    * Rank #3: Section 5.3 (*Prohibited Entertainment Items and Alcohol Limits*) — Score: 0.4615
  * **Pipeline 2 (Cross-Encoder Re-Ranked):**
    * Rank #1: Section 5.1 (*Client Entertainment Hosting*) — Score: +2.6435
    * **Rank #2 (Promoted +1):** Section 5.3 (*Prohibited Entertainment Items and Alcohol Limits*) — Score: +1.1171
    * Rank #3: Section 3.2 (*Prohibited Hotel Incidentals and Personal Amenities*) — Score: -1.5279
* **LLM Answer Comparison (Qwen2.5-0.5B-Instruct):**
  * **Baseline Answer 1 (Hallucination):**
    > *"Based on the policy context provided, the following items are not covered on expense claims for client business meals:*  
    > *- In-room video on-demand/movie rentals*  
    > *- Mini-bar consumption*  
    > *- Laundry service for trips shorter than 5 days*  
    > *- Spa services*  
    > *- Sauna access*  
    > *- Dry cleaning fees*  
    > *- Hotel gym or fitness center pass fees*  
    > *These items are specifically excluded from the reimbursement of business meal expenses as outlined in the policy."*
    * **Observation:** Because the Bi-Encoder retrieved Section 3.2 (*Hotel Incidentals*) ahead of Section 5.3, the LLM hallucinates that hotel room amenities are the prohibited items for client business meals!
  * **Re-Ranked Answer 2 (Corrected):**
    > *"Based on the policy context provided, the items that are not covered on expense claims for client business meals are:*  
    > *(a) Alcoholic beverages exceeding one (1) standard single serving of wine or beer per person during authorized client dinners.*  
    > *(b) Adult entertainment venues, nightclub admission or table bottle service fees, social club charges, and gentlemen's club invoices.*  
    > *(c) Tobacco, vape products, cannabis, and casino gambling losses.*  
    > *(d) Meals or travel for spouses, partners, dependents, or personal companions accompanying the traveler."*
    * **Observation:** Promoting Section 5.3 to Rank #2 gave the model the correct dining restrictions, completely eliminating the hotel amenity hallucination.

---

## 3. Empirical Latency Profiling

We profiled both retrieval pipelines over **10 warmed iterations** (preceded by 5 warm-up passes to eliminate cold-start overhead) on our local CPU (x86_64, PyTorch CPU execution).

| Test ID | Query Category | Pipeline 1 (Dense Baseline) | Stage 2 (Cross-Encoder 10 Pairs) | Pipeline 2 (Total Retrieval) | Overhead Ratio |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **TEST_01** | Exceptions & Corner Cases | 9.78 ms | 122.83 ms | 132.61 ms | 13.56× |
| **TEST_02** | Negation & Constraints | 9.36 ms | 140.54 ms | 149.90 ms | 16.01× |
| **TEST_03** | Control: Ground Transport | 9.33 ms | 122.39 ms | 131.72 ms | 14.12× |
| **TEST_04** | Control: Admin Penalty | 9.47 ms | 126.31 ms | 135.78 ms | 14.34× |
| **MEAN** | **System Average** | **9.49 ms** | **128.02 ms** | **137.50 ms** | **14.51×** |

![Latency Comparison](latency_comparison.png)

### Latency Trade-Off Discussion

1. **Pipeline 1 (Dense Baseline):** Executes in **~9.5 ms** on CPU (query embedding generation takes ~9.0 ms, while cosine dot product across 30 chunks takes <0.5 ms).
2. **Stage 2 (Cross-Encoder Overhead):** Adds an average of **~128.0 ms** to evaluate joint self-attention across the 10 candidate pairs (~12.8 ms per pair on CPU), bringing total retrieval latency to **~137.5 ms**.
3. **Interactive SLA Feasibility:** In human-computer interaction, latencies below 100–200 ms are generally perceived as responsive. While ~138 ms on CPU is acceptable for interactive queries, deploying the Cross-Encoder on GPU with batching drops Stage 2 latency from ~128 ms down to ~15–25 ms, achieving total retrieval well under 35 ms.

---

## 4. Scaling Question: Why Not Run a Cross-Encoder Over the Whole Collection?

If Cross-Encoders exhibit superior accuracy, why not dispense with Bi-Encoders and evaluate every chunk in the corpus using a Cross-Encoder from the outset?

### 1. Quadratic Token-Level Computational Complexity
In a Cross-Encoder, the sequence length is $L = |q| + |c|$. Multi-head self-attention scales quadratically with respect to sequence length: $O(L^2)$ operations per layer. Scoring a corpus of $N$ chunks requires $N$ independent forward passes through the transformer:
* **Measured CPU Scaling:** Based on our measured CPU runtime of $\tau \approx 12.8\text{ ms}$ per candidate pair, evaluating a medium corporate corpus of $N = 1,000,000$ chunks would require:
  $$T_{\text{CPU}} = 1,000,000 \times 12.8\text{ ms} = 12,800\text{ seconds} \approx \mathbf{3.5\text{ hours per query!}}$$
* **GPU Server Estimate:** Even assuming high-end GPU hardware executing a forward pass in $\tau \approx 2.0\text{ ms}$:
  $$T_{\text{GPU}} = 1,000,000 \times 2.0\text{ ms} = 2,000\text{ seconds} \approx \mathbf{33.3\text{ minutes per query!}}$$
Under any concurrent user traffic, exhaustive Cross-Encoder scoring is computationally infeasible.

### 2. Impossibility of Offline Indexing and Precomputation
In a Bi-Encoder system, document representations $\vec{v}_i = \text{Encoder}(c_i)$ are static and completely decoupled from the runtime query. This enables:
* **Offline Precomputation:** All $N$ embeddings can be computed once offline.
* **Sub-linear Search:** Embeddings are indexed into Approximate Nearest Neighbor (ANN) data structures (e.g., HNSW graphs or IVF clusters in FAISS). At runtime, searching 1,000,000 chunks requires exploring only a small fraction of the graph ($O(d \cdot \log N)$), completing in **5 to 10 milliseconds**.

In contrast, the Cross-Encoder score $S(q, c_i) = \mathcal{M}([q; c_i])$ requires both inputs simultaneously. Because the score cannot be factored into independent document vectors, **no precomputed vector index can ever be built**. An exhaustive linear scan $O(N)$ is unavoidable.

### 3. The Two-Stage Architectural Necessity
The two-stage retrieval pipeline resolves this computational bottleneck by functioning as a progressive filtering funnel:
$$\text{Total Latency} = \underbrace{T_{\text{Bi-Encoder}}(1,000,000 \rightarrow 10)}_{\sim 5-10\text{ ms (ANN)}} + \underbrace{T_{\text{Cross-Encoder}}(10\text{ pairs})}_{\sim 15-25\text{ ms (GPU) / } 128\text{ ms (CPU)}} \approx \mathbf{25-138\text{ ms}}$$

Stage 1 operates as a high-recall, sub-linear filter that discards 99.999% of irrelevant chunks in milliseconds. Stage 2 then applies deep, token-level cross-attention only to the top 10 candidates, eliminating failure modes while preserving real-time response times.
