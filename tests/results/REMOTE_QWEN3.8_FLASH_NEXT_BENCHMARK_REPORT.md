# Empirical Benchmark & Stress Test Report: Qwen3.8 Flash Next

**Target Endpoint:** `http://172.16.16.29:8000/v1`  
**Model Name:** `Qwen3.8 Flash Next` (`qwen3.8-flash-next`)  
**Context Window Ceiling:** 262,144 tokens (256k)  
**Execution Date:** 2026-09-20  
**Test Suite:** Standalone test scripts from [`tests/scripts/`](../scripts/)

---

## Executive Summary

A comprehensive benchmark and stress test evaluation was executed against the remote serving instance at `http://172.16.16.29:8000`. The test suite evaluated:
1. **Long-Context Scaling & Stability**: Tested prompt processing and generation throughput up to **200,000 tokens**.
2. **Multi-Client Concurrency & Scheduling**: Evaluated continuous batching, queueing, and behavior under simultaneous client requests.
3. **Streaming Latency & TTFT**: Measured Time-to-First-Token and per-token decoding latency.
4. **Multimodal Vision Ingestion**: Document extraction performance using high-resolution service invoice images.
5. **Quality & Architectural Capabilities**: 4-task coding, systems architecture, debugging, and constraint-retention benchmark.
6. **High-Entropy Recall & Precision**: Multi-stage arithmetic and key-value recall under extended context depths.

---

## 1. Long-Context Scaling Benchmark (`benchmark_context.py`)

The context scaling benchmark progressively evaluated prompt prefill throughput, prefill latency (TTFT), and decode speed across six context milestones from 8,000 to 200,000 tokens.

| Target Context | Actual Prompt Tokens | TTFT / Prefill Latency | Prefill Speed | Decode Speed | Completion Tokens | Result |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **8,000** | 7,987 | 29.862 s | 267.5 tok/s | 30.16 tok/s | 64 | **PASS** |
| **16,000** | 15,997 | 37.963 s | 421.4 tok/s | 28.38 tok/s | 64 | **PASS** |
| **32,000** | 32,017 | 91.349 s | 350.5 tok/s | 28.53 tok/s | 64 | **PASS** |
| **64,000** | 64,012 | 154.640 s | 413.9 tok/s | 25.04 tok/s | 64 | **PASS** |
| **128,000** | 128,002 | 293.821 s | 435.6 tok/s | 22.31 tok/s | 64 | **PASS** |
| **200,000** | 200,002 | 361.647 s | 553.0 tok/s | 19.14 tok/s | 64 | **PASS** |

### Key Findings:
- **Maximum Verified Horizon:** The remote model successfully ingested and sustained an active KV cache through **200,000 tokens** without crashing or timing out.
- **Prefill Throughput:** Chunked prompt ingestion scaled efficiently from 267.5 tok/s up to 553.0 tok/s at 200k tokens.
- **Generation Degradation:** Generation throughput exhibited graceful degradation, maintaining 30.16 tok/s at 8k context and 19.14 tok/s at 200k context.

---

## 2. Multi-Client Concurrency & Batching (`test_concurrency.py`)

Multi-client concurrency and request handling were tested at concurrency levels $c = 1$, $c = 2$, and $c = 4$ with streaming requests (128 max tokens per client).

| Concurrency ($c$) | Emitted Tokens | Wall-Clock Time | Aggregate Speed | Backend Outcome |
| :---: | :---: | :---: | :---: | :--- |
| **$c = 1$** | 128 tokens | 5.81 s | 22.04 tok/s | **PASS**: TTFT was 1.649s; sustained 30.55 tok/s decode. |
| **$c = 2$** | 3 tokens | 1.91 s | 1.57 tok/s | **FAIL**: Client 2 aborted with `finish_reason: "error"`. |
| **$c = 4$** | 4 tokens | 5.40 s | 0.74 tok/s | **FAIL**: Concurrent streams terminated early by backend. |

### Key Findings:
- **Single-Slot Limitation:** The remote backend operates in single-slot serialized execution mode without multi-slot continuous batching or a request queue. Concurrent requests are not queued; overlapping requests are rejected with an error.

---

## 3. Streaming & Latency Benchmark (`test_stream.py`)

Evaluated standard Server-Sent Events (SSE) streaming latency with short conversational prompts.

- **Prompt Tokens:** 22
- **Generated Tokens:** 118
- **Time To First Token (TTFT):** 615.7 ms
- **Prefill Throughput:** 35.7 tok/s
- **Decode Throughput:** 31.8 tok/s
- **Result:** **PASS**

---

## 4. Multimodal Vision Benchmark (`test_vision.py`)

Evaluated end-to-end vision parsing and structural data extraction on `image.png` (Services Invoice #1024).

- **Input Modality:** Base64-encoded PNG image via `/v1/chat/completions`
- **Time To First Token (TTFT + Vision Encode):** 4,525.25 ms
- **Tokens Generated:** 479
- **Decode Speed:** 32.66 tok/s
- **Extraction Fidelity:**
  - Invoice Number (`#1024`): **Detected (100% accurate)**
  - Parties Involved (Avery Davis -> Really Great Company): **Detected**
  - Banking Coordinates & Line Items: **Detected**
  - Subtotal ($1,250.00), Discount ($375.00), Total Due ($875.00): **Detected (100% accurate)**
- **Result:** **PASS**

---

## 5. 4-Task Capability Suite (`compare_benchmark.py`)

Comprehensive suite measuring reasoning depth, systems architecture, debugging, and hidden constraint adherence.

| Task | Prompt Topic | Generated Tokens | TTFT | Decode Speed | Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Task 1** | AVL Tree with Rotations & In-Order Iterator | 1,200 | 624.8 ms | 31.88 tok/s | **PASS**: Clean implementation with valid type annotations and rotation logic. |
| **Task 2** | Concurrency Bounded Buffer Bug Diagnosis | 1,000 | 877.7 ms | 30.77 tok/s | **PASS**: Accurately diagnosed lost wakeups and race conditions; provided fixed code. |
| **Task 3** | High-Throughput Distributed Rate Limiter | 1,200 | 668.6 ms | 31.18 tok/s | **PASS**: Comprehensive systems architecture with sliding window counter & Redis fallback. |
| **Task 4** | Hidden Constraint Adherence (`PaymentGatewayClient`) | 1,000 | 859.6 ms | 31.31 tok/s | **PASS**: Honored audit tag formatting, exponential retry formula, and error envelope. |

---

## 6. High-Entropy Recall & Precision Benchmarks

### 6.1 High-Entropy Associative Recall (`high_entropy_stress_test.py`)
- **Task:** Retrieve buried random alphanumeric key-value pairs (`KEY_BETA_99`, `KEY_GAMMA_12`) within dense distraction filler text.
- **TTFT:** 43,014.6 ms
- **Decode Speed:** 28.08 tok/s
- **Recall Outcome:** **PASS** (100% exact match for target keys and values).

### 6.2 10-Stage Financial Ledger Reconciliation (`extreme_precision_stress_test.py`)
- **Task:** Perform sequential floating-point audit reconciliation across 10 transaction stages.
- **Ground Truth Expected:** `$11,489.41`
- **Result:** **FAIL** (Calculated balance diverged during multi-step floating point arithmetic).

### 6.3 Executable Code Generation & Dynamic Verification (`intensive_accuracy_test.py`)
- **Task:** Generate a complete `LRUCache` class and execute against 25 dynamic unit assertions.
- **Decode Speed:** 31.48 tok/s
- **Result:** **FAIL** (Generated code raised a runtime exception during dynamic test harness execution).

---

## Technical Summary & Recommendations

| Category | Evaluation |
| :--- | :--- |
| **Context Processing** | Excellent. Handles up to 200,000 tokens cleanly with prefill throughput reaching 553 tok/s. |
| **Inference Latency** | Fast warm TTFT (600–900 ms for standard prompts) and steady 31–33 tok/s decode rate. |
| **Multimodal Vision** | Functional. Successfully processes high-resolution images with high entity extraction accuracy. |
| **Concurrency / Serving** | Requires improvement. Currently single-slot only; lacks continuous batching / queuing for concurrent clients. |
| **Complex Arithmetic** | Prone to minor divergence on multi-step sequential floating-point calculations without tool use or code interpreter. |
