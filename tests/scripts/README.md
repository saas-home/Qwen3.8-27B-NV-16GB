# Benchmark, Evaluation, and Testing Scripts

This directory contains standalone testing, stress testing, and enterprise qualification benchmark scripts for local ExLlamaV3 serving endpoints and any OpenAI-compatible LLM server.

---

## 1. `llm_server_full_test.py` (Flagship Enterprise Evaluation Suite)
Comprehensive 14-stage test and evaluation harness for enterprise software architecture, agent tool calling, and long-context production qualification.

### Features:
- Dynamic discovery & interactive picker for available models and endpoints.
- Auto-detects model context length and calculates milestone scaling targets.
- Evaluates:
  1. Streaming & TTFT Latency
  2. Multimodal Vision Document Extraction
  3. Continuous Batching Throughput & Latency Scaling (`--parallel` / `-p`)
  4. 4-Task Architecture Suite (Algorithms, Concurrency Debugging, System Design, Hidden Constraints)
  5. Strict OpenAI Tool / Function Calling Protocol with Autonomous Quota Recovery
  6. Structured JSON Schema Mode (`response_format`)
  7. Prefix Caching / KV Reuse Acceleration (cold vs warm speedup ratio)
  8. Client Socket Abort Recovery (verifies GPU slot release)
  9. Stop Words & Greedy Sampling Determinism
  10. High-Entropy Associative Key-Value Recall across extended context
  11. Extreme Precision Floating-Point Financial Ledger Reconciliation
  12. Dynamic Code Generation & Sandbox Execution Testing
  13. API Error Envelope Protocol Compliance
  14. Dynamic Context Scaling Horizon (Prefix Isolation & Accuracy)
  15. Adversarial Multi-Hop Graph Traversal with Deprecated Version Distractors
  16. Novel Algorithmic Code Synthesis & Automated 5,000-Op Property Fuzz Testing
  17. Combinatorial Anti-Constraint & Negative Instruction Following (IFEval Tier)
  18. Counterfactual Axiomatic Symbolic Deduction (Non-Commutative Modular Algebra)
  19. Extreme Frontier-Depth Multi-Needle Precision (0.5%, 50%, 99.5% Depth)
- Terminal summary scorecard and comprehensive JSON report export.

### Usage:
```bash
# Run full 19-stage qualification benchmark (automated against local server)
python3 tests/scripts/llm_server_full_test.py --auto

# Fast smoke qualification mode (compact context horizons, rapid execution)
python3 tests/scripts/llm_server_full_test.py --auto --quick

# Run specific suite: 'all' (1-19), 'flagship' (1-14), or 'adversarial' (15-19)
python3 tests/scripts/llm_server_full_test.py --auto --suite flagship
python3 tests/scripts/llm_server_full_test.py --auto --suite adversarial

# Run specific test numbers or ranges
python3 tests/scripts/llm_server_full_test.py --auto --test 16
python3 tests/scripts/llm_server_full_test.py --auto --test 15-19
python3 tests/scripts/llm_server_full_test.py --auto --test 1,5,16,18

# Target remote LLM server endpoint
python3 tests/scripts/llm_server_full_test.py \
  --endpoint http://192.168.1.50:8888/v1 \
  --model "qwen3.8-27b" \
  --auto \
  --out tests/results/eval_report.json
```

---

## 2. `test_concurrency.py`
High-concurrency batching and throughput stress test. Measures aggregate generation throughput and individual client latencies across multiple parallel worker threads.

### Usage:
```bash
# Test with 2 concurrent clients (ExLlamaV3 continuous batching)
python3 tests/scripts/test_concurrency.py --parallel 2

# Test with 4 concurrent clients against custom endpoint
python3 tests/scripts/test_concurrency.py --url http://127.0.0.1:8888/v1/chat/completions --model "qwen3.8-27b" -p 4
```

---

## 3. `benchmark_context.py`
Stress test and throughput benchmark across long context windows (e.g. 4k, 8k, 16k, 32k, 64k, 128k, 200k+ tokens).
Features epoch-based prompt salting to isolate true cold prefill speed from warm prefix caching hits, and tracks cached vs uncached tokens.

### Usage:
```bash
# Test 200k tokens (default)
python3 tests/scripts/benchmark_context.py

# Test multiple context milestones with isolated cold prefill
python3 tests/scripts/benchmark_context.py --tokens 4000 16000 64000 128000 200000 --url http://127.0.0.1:8888/v1/chat/completions

# Test with warm prefix caching enabled (no salt)
python3 tests/scripts/benchmark_context.py --tokens 32000 64000 128000 --no-salt
```

---

## 4. `compare_benchmark.py`
Comprehensive 4-task benchmark suite comparing inference backends across:
1. **Coding & Algorithms**: AVL tree implementation with rotations and in-order iterator.
2. **Code Debugging & Concurrency**: Diagnosis of race conditions and edge cases in multi-threaded bounded buffer.
3. **System Architecture**: High-throughput distributed rate limiter design document.
4. **Long-Context Retrieval**: Hidden constraints adherence across extended background documents.

### Usage:
```bash
python3 tests/scripts/compare_benchmark.py --url http://127.0.0.1:8888/v1 --model qwen3.8-27b-exl3-3.0bpw --out tests/results/compare_benchmark.json
```

---

## 5. `intensive_accuracy_test.py`
Deep algorithmic, reasoning, and retrieval benchmark:
1. **Executable Code Generation**: Generates custom O(1) LRU Cache and dynamically executes unit test assertions.
2. **Multi-Needle in a Haystack (NIAH)**: 5 hidden needles buried across ~60,000 tokens with mathematical cross-needle aggregation.
3. **Mathematical Reasoning**: Exact combinatorial and conditional probability deduction with fractional reduction.
4. **Concurrency & Deadlock Proof**: Resource Allocation Graph (WFG) analysis and Coffman deadlock cycle proof.
5. **Strict JSON Schema Validation**: Multi-node server configuration verified against 100% strict field and uniqueness constraints.

### Usage:
```bash
python3 tests/scripts/intensive_accuracy_test.py --url http://127.0.0.1:8888/v1/chat/completions --model qwen3.8-27b-exl3-3.0bpw
```

---

## 6. `extreme_precision_stress_test.py`
Calibrated for deep dependency tracking and floating-point ledger reconciliation across 60,000+ tokens:
1. **14-Stage Sequential ALU Dependency Chain**: Step-by-step arithmetic pipeline buried across 60k tokens.
2. **25-Transaction Floating-Point Ledger Reconciliation**: Multi-entity balance audit across 60k tokens.

### Usage:
```bash
python3 tests/scripts/extreme_precision_stress_test.py --url http://127.0.0.1:8888/v1/chat/completions --model qwen3.8-27b-exl3-3.0bpw
```

---

## 7. `high_entropy_stress_test.py`
Evaluates KV Cache quantization accuracy boundaries:
1. **High-Entropy Associative Key-Value Recall**: 25 high-entropy alphanumeric key-value pairs buried in 50k tokens.
2. **Multi-Hop Variable Tracking**: 8 sequential dependency stages in 50k tokens.

### Usage:
```bash
python3 tests/scripts/high_entropy_stress_test.py --url http://127.0.0.1:8888/v1/chat/completions --model qwen3.8-27b-exl3-3.0bpw
```

---

## 8. `test_stream.py`
Lightweight streaming verification and latency measurement script.

### Usage:
```bash
python3 tests/scripts/test_stream.py --prompt "Explain quantum computing in three sentences."
```

---

## 9. `test_vision.py`
Multimodal vision evaluation script for document parsing and entity extraction using high-resolution images.

### Usage:
```bash
python3 tests/scripts/test_vision.py --url http://127.0.0.1:8888/v1/chat/completions --model qwen3.8-27b-exl3-3.0bpw --out tests/results/vision_result.json
```

---

## 10. `test_adversarial_agent.py` (Enterprise Hardened Adversarial & Agentic Suite)
Extreme-difficulty qualification suite specifically designed to challenge quantized KV caches and deep reasoning models with tasks that cannot be solved via pre-training memorization:
1. **Multi-Hop Graph Traversal with Adversarial Distractors**: Ingests ~100k tokens with versioned temporal distractors (v1.0 deprecated, v2.2 revoked, v3.1 active) requiring complex multi-hop pathfinding.
2. **Novel Algorithmic Code Synthesis & 5,000-Op Fuzz Testing**: Synthesizes non-standard `ConcurrentMonotonicRingBuffer` and dynamically executes a 5,000-operation property-based fuzz harness checking for race conditions, wrap-arounds, and state invariants.
3. **Combinatorial Anti-Constraint Adherence (IFEval Tier)**: 6 simultaneous orthogonal negative/structural constraints (Section II lipogram with zero 'e'/'E', Section III word count 90-130 words, Section IV YAML AST validation, forbidden anti-tokens, and security termination stamp).
4. **Counterfactual Axiomatic Symbolic Deduction**: Non-commutative modular algebra over finite field $\mathbb{Z}_{23}$ testing pure scratchpad reasoning without web memorization.
5. **Frontier Depth Precision at 99.5% Context Horizon**: 3 needles placed at 0.5%, 50.0%, and 99.5% depth in a 100k+ token window with composite arithmetic verification.

### Usage:
```bash
# Run against local server with 100k context depth
python3 tests/scripts/test_adversarial_agent.py --endpoint http://127.0.0.1:8888/v1 --model qwen3.8-27b

# Run with custom context depth (e.g. 150k tokens)
python3 tests/scripts/test_adversarial_agent.py --endpoint http://127.0.0.1:8888/v1 --context-depth 150000
```
