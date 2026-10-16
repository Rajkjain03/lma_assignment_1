# RAG Assignment Report: Baseline Dense Retrieval vs. Cross-Encoder Re-Ranking

**Student Name:** Raj K. Jain  
**Course:** Large Language Models & AI Systems (LMA)  
**Deliverables:** GitHub Repository + Technical Report  
**Date:** October 2026  

---

## 1. Architecture Comparison: Bi-Encoder vs. Cross-Encoder

In modern Retrieval-Augmented Generation (RAG) systems, document retrieval is tasked with identifying the most relevant chunks from an indexed corpus for a given user query. Two fundamentally distinct neural architectures are used for this purpose: **Bi-Encoders** and **Cross-Encoders**.

```
BI-ENCODER ARCHITECTURE (Two-Tower / Independent Encoding):
  Query (q)  -------> [ Transformer Tower ] -------> Dense Vector u (384-d)  \
                                                                                ---> Cosine Similarity: cos(u, v)
  Chunk (c)  -------> [ Transformer Tower ] -------> Dense Vector v (384-d)  /
  (Corpus vectors v are computed once offline and indexed into Vector DB / FAISS)

CROSS-ENCODER ARCHITECTURE (Joint All-to-All Self-Attention):
  [CLS] + Query tokens + [SEP] + Chunk tokens + [SEP]
                         │
                         ▼
             ┌────────────────────────┐
             │ Layer 1 Self-Attention │  <-- Every query token attends
             ├────────────────────────┤      directly to every chunk token
             │ Layer 2 Self-Attention │
             ├────────────────────────┤
             │ ... All Layers ...     │
             ├────────────────────────┤
             │ Classification Head    │
             └────────────────────────┘
                         │
                         ▼
             Relevance Logit S(q, c) in (-inf, +inf)
  (Cannot be precomputed; requires live forward pass for every query-chunk pair)
```

### Core Mechanical Differences

1. **Token Interaction & Representation Bottleneck:**
   * **Bi-Encoder (e.g., `all-MiniLM-L6-v2`):** Processes the query $q$ and document chunk $c$ in complete isolation. The transformer generates contextual token representations for the document alone and then compresses them via mean-pooling into a single fixed-length dense vector $\vec{v} \in \mathbb{R}^{384}$. At retrieval time, the query is similarly compressed into $\vec{u} \in \mathbb{R}^{384}$. **Tokens of the query never attend to tokens of the document.** Semantic similarity is approximated solely by the geometric angle between the two vectors: $\cos(\vec{u}, \vec{v}) = \frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}$. Squeezing an entire 300-word paragraph into a 384-dimensional vector causes a severe informational bottleneck where conditional exceptions, negations, and specific qualifiers are blurred.
   * **Cross-Encoder (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`):** Receives the concatenated sequence `[CLS] query [SEP] chunk [SEP]` and feeds it jointly into all transformer layers. Through bidirectional multi-head self-attention:
     $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
     every token in the query directly compares against every token in the passage across all hidden layers. This allows the model to capture deep semantic dependencies, identify modifier words ("only if", "except when", "prohibited"), and evaluate whether the chunk truly answers the specific intent of the query.

2. **Computational Complexity & Indexability:**
   * **Bi-Encoder:** Allows offline precomputation. For a corpus of $N$ chunks, all $N$ embeddings are computed once and stored in an Approximate Nearest Neighbor (ANN) index (e.g., HNSW or IVF in FAISS). At runtime, searching $1,000,000$ chunks requires just 1 forward pass for the query and a fast vector scan taking 3–10 milliseconds ($O(d \cdot \log N)$).
   * **Cross-Encoder:** Cannot be precomputed because the joint representation depends on both the query and the chunk simultaneously. Scoring $N$ candidate chunks requires $N$ full forward passes through the transformer with quadratic token complexity $O(N \cdot L^2)$.

---

## 2. Qualitative & Rank Shift Analysis

We evaluated the systems against our realistic enterprise compliance document: **"Acme Global Travel, Expense & Procurement Policy (2026)"** (30 structured clauses). 

Below is the documented evidence of rank shifts for failure queries where baseline dense retrieval stumbled, and two-stage cross-encoder re-ranking restored accuracy:

### Summary of Failure Modes and Rank Promotions

| Test ID | Failure Mode Category | Critical Document Clause | Baseline Dense Rank | Re-Ranked Cross-Encoder Rank | Net Promotion Shift | Buried in Baseline Top 3? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **TEST_01** | **Exceptions & Corner Cases** | Section 2.4 (*Long-Haul International Travel Exception Clause*) | **Rank #4** | **Rank #1** | **+3 Ranks** | **YES** (Omitted from Prompt) |
| **TEST_02** | **Negation & Constraints** | Section 5.3 (*Prohibited Entertainment Items & Alcohol Limits*) | **Rank #4** | **Rank #2** | **+2 Ranks** | **YES** (Omitted from Prompt) |
| **TEST_03** | **Keyword Trap / Distractor** | Section 4.4 (*Luxury Chauffeured Vehicles & Limousine Ban*) | Rank #2 | Rank #2 | Surfaced in Top 2 | Preserved in Top 2 |
| **TEST_04** | **Buried Penalty Clause** | Section 8.5 (*Late Submission Forfeiture & Card Revocation*) | Rank #1 | Rank #1 | Top Ranked | Retained at Rank #1 |

![Qualitative Rank Shifts Comparison](rank_shifts_comparison.png)

---

### Detailed Case Study 1: Exceptions & Corner Cases (TEST_01)
* **User Query:** *"Can staff fly in premium cabins for international client negotiations?"*
* **Target Truth:** Section 2.4 explicitly permits Business Class for international flights exceeding 6 hours if the traveler is presenting to a client or conducting negotiations within 12 hours of arrival, subject to VP and CFO approval.
* **Baseline Retrieval Failure:**
  * Baseline Dense Ranks:
    1. **Rank #1:** Section 2.1 (*Standard Cabin Class Mandate*) — Score: 0.441
    2. **Rank #2:** Section 1.3 (*Designated Corporate Travel Booking Engine*) — Score: 0.385
    3. **Rank #3:** General Overview (Page 1) — Score: 0.354
    4. **Rank #4 (BURIED):** Section 2.4 (*Long-Haul International Travel Exception Clause*) — Score: 0.348
  * Because Section 2.4 sat outside the Top 3, the Baseline Prompt only received Section 2.1 (which states that all travel must be in Standard Economy regardless of seniority).
  * **LLM Answer 1 (Baseline):**
    > *"No. Based strictly on the provided policy passages (Section 2.1 'Standard Cabin Class Mandate'), all domestic and international air travel must be booked exclusively in Standard Economy (Coach) class, regardless of employee seniority, managerial title, or total travel duration. The provided context contains no provisions or exceptions permitting business class or premium cabin bookings for client negotiations."*
    * **Failure Impact:** The baseline generated a **critically incorrect negative answer** (misstated policy), causing an employee to forgo a valid corporate travel accommodation.
* **Cross-Encoder Re-Ranking Resolution:**
  * The Cross-Encoder performed full attention between the query words ("premium cabins", "international client negotiations") and Section 2.4, immediately recognizing that Section 2.4's condition directly answers the query.
  * Re-Ranked Ranks:
    1. **Rank #1:** Section 2.4 (*Long-Haul International Travel Exception Clause*) — Score: +4.82 (**Promoted from Rank #4 to #1**)
    2. **Rank #2:** Section 2.1 (*Standard Cabin Class Mandate*) — Score: +1.15
    3. **Rank #3:** Section 1.2 (*Mandatory Pre-Trip Travel Authorization*) — Score: -1.84
  * **LLM Answer 2 (Re-Ranked):**
    > *"Yes, employees may book Business Class under specific long-haul conditions. According to Section 2.4 ('Long-Haul International Travel Exception Clause'), employees traveling on continuous non-stop international flights exceeding 8 hours—or exceeding 6 hours if scheduled to deliver a client presentation or attend executive contract negotiations within 12 hours of flight arrival—are eligible to book Business Class. This requires prior written approval from both the Division Vice President and Chief Financial Officer at least 10 business days prior to ticketing."*
    * **Result:** Re-ranking completely eliminated the error and produced 100% compliant guidance.

---

### Detailed Case Study 2: Negation & Constraints (TEST_02)
* **User Query:** *"What items are not covered on expense claims for client business meals?"*
* **Target Truth:** Section 5.3 strictly limits alcohol to 1 drink per diner, completely prohibits solo alcohol, bans adult entertainment/nightclub bottle fees, and disallows meals for accompanying spouses.
* **Baseline Retrieval Failure:**
  * Dense embeddings matched the positive words "client business meals" against Section 5.1 (*Client Entertainment and Prospective Partner Hosting*), ranking it at #1. Section 5.3 was buried at **Rank #4**.
  * **LLM Answer 1 (Baseline):**
    > *"Based on the provided context (Section 5.1 and Section 3.2), employees may expense reasonable business meals with prospective clients with an itemized receipt showing attendee names. Prohibited items mentioned in the provided text only cover hotel room incidentals (mini-bars, in-room movies, gym fees, laundry). The provided passages omit specific caps or prohibitions regarding alcohol, nightlife, or entertainment expenses incurred during client dinners."*
    * **Failure Impact:** Severe omission. The LLM was blind to the alcohol cap and nightclub prohibition, creating a compliance loophole for employee fraud.
* **Cross-Encoder Re-Ranking Resolution:**
  * The Cross-Encoder's cross-attention detected the negation qualifier "not covered" and aligned it with Section 5.3's "prohibited" and "strictly non-reimbursable" provisions.
  * Re-ranked Section 5.3 from **Rank #4 $\rightarrow$ Rank #2**.
  * **LLM Answer 2 (Re-Ranked):**
    > *"According to Section 5.3 ('Prohibited Entertainment Items and Alcohol Consumption Limits'), the following items are strictly prohibited and non-reimbursable:\n1. Alcoholic beverages exceeding one (1) standard single serving of wine or beer per person during authorized client dinners (alcohol during solo employee dining is 100% non-reimbursable).\n2. Adult entertainment venues, nightclub admission or table bottle service fees, social club charges, and gentlemen's club invoices.\n3. Tobacco, vape products, cannabis, and casino gambling losses.\n4. Meals or travel for spouses, partners, dependents, or personal companions accompanying the traveler."*

---

## 3. Latency Profiling

We recorded and tabulated runtimes across 10 warmed iterations on an NVIDIA GeForce RTX GPU. 

### Latency Benchmark Table

| Test Case | Evaluated Query Category | Pipeline 1: Baseline Dense Retrieval | Stage 2: Cross-Encoder Re-Ranking (Top 10) | Pipeline 2: Total Two-Stage Latency | Re-Ranking Latency Overhead Ratio |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **TEST_01** | Exceptions & Corner Cases | 9.80 ms | 17.99 ms | 27.79 ms | **2.83x** |
| **TEST_02** | Negation & Constraints | 9.91 ms | 21.02 ms | 30.93 ms | **3.12x** |
| **TEST_03** | Keyword Trap / Distractor | 8.81 ms | 19.03 ms | 27.84 ms | **3.16x** |
| **TEST_04** | Buried Penalty Clause | 9.53 ms | 19.42 ms | 28.95 ms | **3.04x** |
| **AVERAGE** | **System Mean Latency** | **9.51 ms** | **19.37 ms** | **28.88 ms** | **3.04x** |

![Comparative Latency Profiling](latency_comparison.png)

### Observations on Latency Trade-Off:
1. **Dense Retrieval (Pipeline 1):** Takes an average of **9.51 ms**, consisting of query embedding generation (~8 ms on GPU) and dot product vector search (~0.5 ms).
2. **Re-Ranking (Pipeline 2):** Adds an average overhead of **19.37 ms** to score the 10 candidate pairs jointly through the 6-layer MiniLM transformer.
3. **Total Overhead:** Pipeline 2 is **~3.0x slower** than Pipeline 1 (28.88 ms vs. 9.51 ms). However, in real-world human-facing applications, an end-to-end retrieval latency under **30 ms** is imperceptible to users (well below the 100 ms threshold for instantaneous response), while completely eliminating catastrophic hallucinations and missed policy exceptions.

---

## 4. Scaling Question: Why Not Run a Cross-Encoder Across the Entire Corpus?

A student encountering the superior accuracy of Cross-Encoders might naturally ask:  
*"If Cross-Encoders are so much better at capturing nuance and avoiding traps, why don't we simply score all document chunks with a Cross-Encoder from the very start and discard Bi-Encoders entirely?"*

### 1. The Mathematical Explosion of Computational Complexity
In a Cross-Encoder, the query and document chunk are concatenated and fed jointly through self-attention layers:
* Self-attention scales quadratically with sequence length: $O(L^2)$ where $L = |q| + |c|$.
* A single Cross-Encoder forward pass on a 256-token sequence takes approximately **2.0 milliseconds** on a high-end GPU.
* If an enterprise corpus contains **$N = 1,000,000$ document chunks**:
  $$\text{Query Latency} = 1,000,000 \times 2.0 \text{ ms} = 2,000,000 \text{ ms} = 2,000 \text{ seconds} \approx \mathbf{33.3 \text{ minutes per query!}}$$
* Serving just 10 concurrent user queries would require thousands of high-cost enterprise GPUs running at 100% capacity.

### 2. Impossibility of Vector Indexing & Precomputation
* **Bi-Encoders decouple document representations from query representations:**
  $$\vec{v}_i = \text{Encoder}(c_i) \quad \text{(computed once at ingestion time)}$$
  All million document vectors can be precomputed and organized into hierarchical graph structures (HNSW in FAISS/Pinecone/Milvus). Searching a query vector against 1,000,000 indexed chunks takes only **5 milliseconds** using sub-linear approximate nearest neighbor search ($O(\log N)$).
* **Cross-Encoders cannot be precomputed:**
  $$S(q, c_i) = \text{Model}([q; c_i])$$
  Because the score depends on joint attention between query tokens and chunk tokens, **no offline vector database can ever be built**. Every query would require an exhaustive brute-force pass through every single chunk in the database.

### 3. The Power of the Two-Stage Funnel
The Two-Stage Retrieval architecture resolves this dilemma by pairing both models in a high-efficiency funnel:
* **Stage 1 (Bi-Encoder):** Casts a wide net. Filters the entire database from **1,000,000 chunks down to 10 candidates in 5 ms**.
* **Stage 2 (Cross-Encoder):** Runs precision joint self-attention on only those **10 candidates in 15 ms**.
* **Total End-to-End Latency:** **$5\text{ ms} + 15\text{ ms} = 20\text{ ms}$**.

This delivers the **retrieval quality of a Cross-Encoder at the scale and speed of a Bi-Encoder**.
