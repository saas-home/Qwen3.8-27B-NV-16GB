# Benchmark and Testing Scripts

This directory contains standalone testing, stress test, and benchmarking scripts for evaluating ExLlamaV3 serving endpoints.

---

## 1. `benchmark_context.py`
Stress test and throughput benchmark across long context windows (e.g. 50k, 100k, 150k, 200k+ tokens).

### Usage:
```bash
# Test 200k tokens (default)
python3 tests/scripts/benchmark_context.py

# Test multiple context milestones
python3 tests/scripts/benchmark_context.py --tokens 32000 64000 128000 200000 --url http://127.0.0.1:8888/v1/chat/completions
```

---

## 2. `compare_benchmark.py`
Comprehensive 4-task benchmark suite comparing inference backends across:
1. **Coding & Algorithms**: AVL tree implementation with rotations and in-order iterator.
2. **Code Debugging & Concurrency**: Diagnosis of race conditions and edge cases in multi-threaded bounded buffer.
3. **System Architecture**: High-throughput distributed rate limiter design document.
4. **Long-Context Retrieval**: Hidden constraints adherence across extended background documents.

### Usage:
```bash
python3 tests/scripts/compare_benchmark.py --url http://127.0.0.1:8888/v1 --out tests/results/exl3_benchmark_tasks.json
```

---

## 3. `test_stream.py`
Lightweight streaming verification and latency measurement script.

### Usage:
```bash
python3 tests/scripts/test_stream.py --prompt "Explain quantum computing in three sentences."
```

---

## 4. `test_vision.py`
Multimodal vision evaluation script for document parsing and entity extraction using high-resolution images.

### Usage:
```bash
python3 tests/scripts/test_vision.py --url http://127.0.0.1:8888/v1/chat/completions --model qwen3.8-27b-exl3-3.0bpw --out tests/results/exl3_vision_result.json
```
