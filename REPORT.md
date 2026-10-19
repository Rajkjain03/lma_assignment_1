# RAG Assignment 1: Dense Baseline vs. Cross-Encoder Re-Ranking

**Course:** Large Language Models & AI Systems (LMA)  
**Author:** Raj K. Jain (Roll: LMA-2026)  
**GitHub Repository:** [https://github.com/Rajkjain03/lma_assignment_1](https://github.com/Rajkjain03/lma_assignment_1)  
**Date:** October 2026  

---

## 1. Architecture Comparison: Bi-Encoder vs. Cross-Encoder

In this assignment, I implemented and evaluated two retrieval architectures for Retrieval-Augmented Generation (RAG): a single-stage dense Bi-Encoder baseline and a two-stage pipeline combining the Bi-Encoder with a Cross-Encoder re-ranker.

### Mechanical Differences

* **Bi-Encoder (`all-MiniLM-L6-v2`):** A Bi-Encoder processes the query $q$ and candidate document chunks $c$ through separate transformer passes, compressing each text into a standalone 384-dimensional dense vector. Relevance is computed via geometric cosine similarity between the two embeddings. Because chunk vectors are completely independent of any runtime query, all chunk embeddings across the entire corpus can be precomputed offline and indexed in an Approximate Nearest Neighbor (ANN) graph (such as HNSW in FAISS), enabling sub-linear searches in just a few milliseconds. However, compressing an entire 200-word paragraph into 384 numbers creates an informational bottleneck: query and document tokens never interact directly in attention, which means fine-grained constraints, exceptions, and negations often get lost.
* **Cross-Encoder (`ms-marco-MiniLM-L-6-v2`):** A Cross-Encoder concatenates the query and candidate chunk into a single sequence `[CLS] q [SEP] c [SEP]` and passes it through full transformer self-attention. Every query token attends directly to every chunk token across all layers. This all-to-all cross-attention allows the model to spot exact conditional qualifiers (such as specific flight duration thresholds) and negative constraints (such as non-reimbursable meal items). The core tradeoff is computational: query and chunk cannot be decoupled, so offline precomputation is impossible and each candidate pair requires a full forward pass at runtime.

| Feature | Bi-Encoder (`all-MiniLM-L6-v2`) | Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) |
| :--- | :--- | :--- |
| **Input Format** | Query and passage encoded independently | Concatenated pair $[q; c]$ encoded together |
| **Token Interaction** | Zero cross-attention between $q$ and $c$ tokens | Full all-to-all cross-attention across all layers |
| **Scoring Method** | Cosine similarity of mean-pooled embeddings | Classification logit from `[CLS]` token |
| **Offline Indexing** | Precomputed vectors stored in ANN index | Not precomputable; requires live forward pass |
| **Query Complexity** | Sub-linear $\mathcal{O}(d \cdot \log N)$ via vector search | Linear $\mathcal{O}(N \cdot L^2)$ with $N$ transformer passes |
| **Typical Role** | High-recall fast filter (Top 100 $\rightarrow$ 10) | High-precision re-ranker (Top 10 $\rightarrow$ 3) |

---

## 2. Qualitative Retrieval & Rank Shift Analysis

### Evaluation Document & Experimental Setup
I constructed a synthetic 4-page corporate compliance manual titled *"Acme Global Travel, Expense and Procurement Policy (2026)"* containing 30 structured sections. It contains realistic enterprise edge cases: a general coach airfare mandate with distant exception rules, meal entertainment caps alongside distractor hotel incidentals, and explicit late-submission forfeiture penalties.

To evaluate both pipelines, I designed a 6-query test suite. Tests 1 and 2 explore distractor and context ordering effects where the target was retrieved at Rank #3 in the baseline. Tests 3 and 4 evaluate genuine retrieval omission failures where the critical section was buried outside the Top 3 (ranks 5 and 8) in the baseline and promoted into the Top 3 by the Cross-Encoder. Tests 5 and 6 serve as stable controls. The local generator is `Qwen/Qwen2.5-0.5B-Instruct`, an open-weights 0.5B parameter LLM run locally on CPU.

### Qualitative Retrieval Rank Shifts

| Test ID | Query Category | Target Policy Section | Bi-Encoder Rank | Cross-Encoder Rank | Shift | Target in Top 3? |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: |
| **TEST_01** | Exceptions & Distractors | Sec 2.4 (Long-Haul Flight Exception) | Rank #3 (score: 0.4039) | **Rank #1 (score: +0.4351)** | **+2** | Promoted to #1 |
| **TEST_02** | Negation & Distractor Chunk | Sec 5.3 (Prohibited Meal Items & Alcohol) | Rank #3 (score: 0.4615) | **Rank #2 (score: +1.1171)** | **+1** | Promoted to #2 |
| **TEST_03** | Buried Clause: Hotel Laundry | Sec 3.2 (Prohibited Hotel Incidentals) | Rank #5 (score: 0.3667) | **Rank #1 (score: +2.4859)** | **+4** | Promoted into Top 3 |
| **TEST_04** | Buried Clause: Ancillary Fees | Sec 2.3 (Baggage & Ancillary Airline Fees) | Rank #8 (score: 0.3413) | **Rank #2 (score: -1.6741)** | **+6** | Promoted into Top 3 |
| **TEST_05** | [CONTROL] Ground Transport | Sec 4.4 (Luxury Chauffeured Car Ban) | Rank #2 (score: 0.4516) | Rank #2 (score: +1.1141) | +0 | Retained in Top 2 |
| **TEST_06** | [CONTROL] Admin Penalty | Sec 8.5 (Late Submission Forfeiture) | Rank #1 (score: 0.6565) | Rank #1 (score: +5.2319) | +0 | Retained at Rank #1 |

![Qualitative Rank Shifts Comparison](rank_shifts_comparison.png)

---

### Detailed Case Study 1: Exceptions & Distractor Context (TEST_01)
* **Query:** *"Can staff fly in premium cabins for international client negotiations?"*
* **Target Rule:** Section 2.4 allows Business Class on continuous non-stop international flights exceeding 8 hours (or exceeding 6 hours if presenting or negotiating within 12 hours of arrival), with written VP and CFO approval.
* **Retrieval Comparison:**
  * In the baseline Bi-Encoder, Section 2.1 (*Standard Cabin Class Mandate*) ranked at #1 (cosine 0.5428), Section 1.3 was #2 (0.4127), and the target exception Section 2.4 placed at Rank #3 (0.4039). Because Section 2.4 was in the Top 3 context, the baseline LLM did see it.
* **LLM Generation Comparison:**
  * **Answer 1 (Baseline):** *"Yes, staff can fly in premium cabins for international client negotiations. However, they need to obtain prior written concurrence from both the Division Vice President and the Chief Financial Officer at least 10 business days before ticketing."*
  * **Observation:** Answer 1 is not factually incorrect, but it missed the flight duration thresholds (> 8 hours, or > 6 hours for negotiations). Because our generator is a compact 0.5B model, having the general coach mandate at Rank #1 overshadowed the conditional numbers in the third chunk. A stronger frontier model might have synthesized the full answer from this context, but for a 0.5B model, context ordering matters significantly.
  * **Answer 2 (Re-Ranked):** *"Yes, staff can fly in premium cabins for international client negotiations under the long-haul international travel exception clause. The policy allows this exception when traveling on continuous non-stop international flights with flight durations exceeding 8 hours or 6 hours if the traveler is scheduled to deliver a client presentation or attend executive contract negotiations within 12 hours of flight arrival. Employees need prior written concurrence from both the Division Vice President and the Chief Financial Officer no later than 10 business days before booking their tickets."*
  * Promoting Section 2.4 to Rank #1 (+0.4351) placed the exception at the front of the prompt, ensuring all thresholds were captured.

---

### Detailed Case Study 2: Negation & Distractor Chunk (TEST_02)
* **Query:** *"What items are not covered on expense claims for client business meals?"*
* **Target Rule:** Section 5.3 restricts alcohol to 1 drink per diner, prohibits solo dining alcohol, and explicitly bans nightclub bottle service, adult entertainment, tobacco, and gambling losses.
* **Retrieval Comparison:**
  * The dense retriever matched "client business meals" to Section 5.1 at Rank #1 (0.5936). For the negative constraint "not covered", it pulled Section 3.2 (*Prohibited Hotel Incidentals*) at Rank #2 (0.5562). The actual meal prohibition clause (Section 5.3) sat at Rank #3 (0.4615).
* **LLM Generation Comparison:**
  * **Answer 1 (Baseline Distraction Error):** *"Based on the policy context provided, the following items are not covered on expense claims for client business meals: In-room video on-demand/movie rentals, Mini-bar consumption, Laundry service for trips shorter than 5 days, Spa services, Sauna access, Dry cleaning fees, Hotel gym or fitness center pass fees..."*
  * **Observation:** This was not an ungrounded hallucination; the 0.5B model faithfully copied items from a chunk it was given, but it was misled by a higher-ranked distractor (Section 3.2).
  * **Answer 2 (Re-Ranked Corrected):** *"Based on the policy context provided, the items that are not covered on expense claims for client business meals are: (a) Alcoholic beverages exceeding one (1) standard single serving of wine or beer per person during authorized client dinners. (b) Adult entertainment venues, nightclub admission or table bottle service fees, social club charges, and gentlemen's club invoices. (c) Tobacco, vape products, cannabis, and casino gambling losses. (d) Meals or travel for spouses, partners, dependents, or personal companions accompanying the traveler."*
  * The Cross-Encoder promoted Section 5.3 to Rank #2 (+1.1171) ahead of Section 3.2, resolving the distraction error.

---

### Detailed Case Studies 3 & 4: Buried Clause Omission Failures (TEST_03 & TEST_04)
Tests 3 and 4 represent genuine retrieval omission failures where the target clause was buried outside the baseline Top 3 context window and rescued into the Top 3 by the Cross-Encoder.

* **TEST_03 (*"Under what trip length can travelers expense laundry and dry cleaning?"*):**
  * Section 3.2 specifies that laundry is non-reimbursable for trips shorter than 5 days.
  * Dense search buried Section 3.2 at **Rank #5** (0.3667), loading only filing deadlines (Section 8.1) and pre-trip authorization (Section 1.2) into the baseline prompt.
  * Lacking the clause, the baseline LLM fabricated an answer: *"...travelers cannot expense laundry and dry cleaning for trips longer than 30 days."*
  * The Cross-Encoder promoted Section 3.2 directly to **Rank #1** (+2.4859), and the re-ranked model answered correctly: *"For trips shorter than 5 days, travelers cannot expense laundry and dry cleaning."*
* **TEST_04 (*"Are flight cancellation insurance and seat selection fees reimbursable?"*):**
  * Section 2.3 governs baggage allowances and ancillary airline fees.
  * Dense retrieval buried Section 2.3 at **Rank #8** (0.3413) due to vocabulary mismatch.
  * The Cross-Encoder promoted Section 2.3 to **Rank #2** (-1.6741), rescuing the chunk into the prompt context.

### Control Cases (TEST_05 & TEST_06)
In TEST_05 (airport rideshare vs limousine ban in Section 4.4) and TEST_06 (30-day late expense submission penalty in Section 8.5), the semantic overlap was already strong. Section 4.4 remained at Rank #2 in both pipelines, and Section 8.5 stayed at Rank #1 in both pipelines. These control cases verify that the Cross-Encoder does not destabilize rankings when the dense retriever already finds the correct chunk.

---

## 3. Empirical Latency Profiling

Both retrieval pipelines were benchmarked over **10 warmed iterations** (following 5 warm-up passes to eliminate initialization noise) on my local CPU.

| Test Query Category | P1 Baseline (Bi-Encoder) | Stage 2 (Cross-Encoder 10 Pairs) | P2 Total Latency | Overhead Ratio |
| :--- | :---: | :---: | :---: | :---: |
| **TEST_01 (Exceptions)** | 10.05 ms | 129.97 ms | 140.02 ms | 13.94x |
| **TEST_02 (Negations)** | 10.21 ms | 159.05 ms | 169.26 ms | 16.58x |
| **TEST_03 (Hotel Laundry)** | 10.14 ms | 133.98 ms | 144.13 ms | 14.21x |
| **TEST_04 (Ancillary Fees)** | 10.27 ms | 158.74 ms | 169.01 ms | 16.46x |
| **TEST_05 (Control: Ground)** | 10.21 ms | 132.93 ms | 143.14 ms | 14.02x |
| **TEST_06 (Control: Admin)** | 10.28 ms | 135.07 ms | 145.35 ms | 14.14x |
| **EMPIRICAL MEAN** | **10.19 ms** | **141.62 ms** | **151.82 ms** | **14.89x** |

![Empirical Latency Breakdown](latency_comparison.png)

### Latency Trade-Off Analysis
* The dense baseline runs in $\approx 10.2$ ms on CPU, with roughly 9.7 ms spent on query embedding generation and under 0.5 ms on cosine dot products across the 30 chunks.
* Re-ranking 10 candidate pairs through the Cross-Encoder added an average of 141.6 ms ($\approx 14.2$ ms per candidate pair), bringing total retrieval time to $\approx 151.8$ ms.
* While a $14.9\times$ overhead sounds large, an end-to-end retrieval latency of 152 ms remains well within the standard 200 ms threshold for interactive user search.
* On a dedicated GPU server, batched Cross-Encoder inference is projected to drop to an estimated 15–25 ms total ($\approx 2.0$ ms per candidate pair), bringing total retrieval runtime under 30 ms.

---

## 4. Scaling: Why Not Run Cross-Encoder Over the Whole Collection?

If Cross-Encoders are noticeably more accurate, why not bypass the Bi-Encoder entirely and score every chunk in the collection with a Cross-Encoder?

1. **Compute Cost and Quadratic Complexity:** A Cross-Encoder must perform full bidirectional self-attention across the concatenated sequence length $L = |q| + |c|$, which has quadratic complexity $\mathcal{O}(L^2)$. Scoring a collection of $N$ passages requires running $N$ complete transformer forward passes. Based on my measured CPU runtime of 14.2 ms per pair, evaluating a single query over a modest corporate corpus of $N = 1,000,000$ chunks would take $1,000,000 \times 14.2\text{ ms} \approx 3.9\text{ hours}$. Even on a modern GPU with an estimated latency of 2.0 ms per pair, a full linear scan would take over 33 minutes per query. Under interactive user traffic, this is completely infeasible.
2. **Impossibility of Offline Indexing:** With a Bi-Encoder, passage embeddings are calculated once offline and indexed into sub-linear Approximate Nearest Neighbor graph structures (e.g., HNSW in FAISS), allowing millisecond search across millions of documents. In contrast, a Cross-Encoder requires both the query and passage simultaneously to compute attention. Because the query is unknown until runtime, no precomputed index can be constructed, forcing an exhaustive $\mathcal{O}(N)$ scan over every single passage.
3. **The Two-Stage Solution:** The two-stage funnel reconciles this trade-off by using the Bi-Encoder to prune 99.9% of the corpus in milliseconds, after which the Cross-Encoder evaluates only the top 10 candidates with token-level cross-attention.
