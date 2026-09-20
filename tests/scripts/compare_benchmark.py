#!/usr/bin/env python3
"""
Benchmark suite comparing LLM inference backends (ExLlamaV3 vs llama.cpp).
Measures:
1. TTFT (Time To First Token) / Prefill Latency
2. Decode throughput (tok/s)
3. Total generation time & token count
4. Coding accuracy / syntax validation
5. Concurrency bug diagnosis
6. High-throughput system architecture quality
7. Long-context hidden constraint retrieval
"""

import sys
import os
import time
import json
import urllib.request
import urllib.error
import argparse

BENCHMARK_TASKS = [
    {
        "id": "task1_avl_tree",
        "category": "Coding & Algorithms",
        "name": "AVL Tree with Rotations & In-Order Iterator",
        "system": "You are an expert systems software engineer. Write clean, production-grade, bug-free Python code.",
        "prompt": (
            "Write a complete, production-grade self-balancing AVL Tree in Python with complete type annotations.\n"
            "Requirements:\n"
            "1. Implement `AVLNode[T]` and `AVLTree[T]` with support for `insert(val: T) -> None`, `delete(val: T) -> bool`, and `search(val: T) -> bool`.\n"
            "2. Implement left rotation, right rotation, and double rotations correctly with node height recalculations.\n"
            "3. Implement an iterator `__iter__` that performs in-order traversal yielding values in sorted order.\n"
            "4. Include a self-contained unit test function `test_avl()` that inserts [50, 25, 75, 10, 30, 60, 80, 5, 15, 27, 55], asserts sorted traversal, deletes several nodes (including root and leaf), and verifies tree invariants."
        ),
        "max_tokens": 1500,
        "temperature": 0.2,
    },
    {
        "id": "task2_concurrency_debug",
        "category": "Code Debugging & Reasoning",
        "name": "Concurrent Bounded Buffer Bug Diagnosis",
        "system": "You are a senior concurrency and distributed systems engineer. Be precise, identify exact root causes, and provide corrected code.",
        "prompt": (
            "Analyze the following Python multi-threaded bounded buffer implementation. Identify all concurrency bugs, race conditions, and edge-case failures, explain why they occur, and provide the fully corrected, thread-safe implementation:\n\n"
            "```python\n"
            "import threading\n"
            "import time\n\n"
            "class BuggyBoundedBuffer:\n"
            "    def __init__(self, capacity):\n"
            "        self.capacity = capacity\n"
            "        self.buffer = []\n"
            "        self.lock = threading.Lock()\n"
            "        self.not_full = threading.Condition(self.lock)\n"
            "        self.not_empty = threading.Condition(self.lock)\n"
            "        self.processed_items = set()\n\n"
            "    def put(self, item):\n"
            "        self.lock.acquire()\n"
            "        if len(self.buffer) >= self.capacity:\n"
            "            self.not_full.wait()\n"
            "        self.buffer.append(item)\n"
            "        self.processed_items.add(item)\n"
            "        self.not_empty.signal()\n"
            "        self.lock.release()\n\n"
            "    def get(self):\n"
            "        self.lock.acquire()\n"
            "        if len(self.buffer) == 0:\n"
            "            self.not_empty.wait()\n"
            "        item = self.buffer.pop(0)\n"
            "        self.not_full.signal()\n"
            "        self.lock.release()\n"
            "        return item\n"
            "```\n"
        ),
        "max_tokens": 1200,
        "temperature": 0.2,
    },
    {
        "id": "task3_system_design",
        "category": "System Architecture & Design",
        "name": "High-Throughput Distributed Rate Limiter",
        "system": "You are a Principal Infrastructure Architect at a global scale technology company.",
        "prompt": (
            "Write a detailed technical architecture and design document for a Global Distributed Rate Limiter & Abuse Prevention Service.\n"
            "Scale & Requirements:\n"
            "- 500,000 requests/second globally across 4 regions.\n"
            "- P99 latency overhead < 3ms on the critical request path.\n"
            "- Consistent policy enforcement across multi-tenant API clients.\n\n"
            "Structure your document with:\n"
            "1. Architectural Overview & Component Topology (ASCII/Mermaid diagram).\n"
            "2. Algorithm Comparison & Selection (Token Bucket vs Leaky Bucket vs Sliding Window Counter with Redis/Aerospike/local memory).\n"
            "3. Synchronization & Global State Strategy (dealing with regional replication lag, eventual consistency vs strict limits).\n"
            "4. Resiliency & Failure Modes (how the system behaves during network partition between regions or datastore degradation).\n"
            "5. Concrete Redis Lua script or Go/Rust data structure for atomic rate evaluation."
        ),
        "max_tokens": 1800,
        "temperature": 0.5,
    },
    {
        "id": "task4_long_context_constraints",
        "category": "Long Context & Retrieval",
        "name": "Hidden Architectural Constraints Adherence",
        "system": "You are a senior backend engineer implementing client libraries strictly according to specifications.",
        "prompt": (
            "You are implementing an enterprise payment gateway client. Below is a background document describing company standards.\n\n"
            + ("# Section: Enterprise Architectural Standards and Policies\n"
               "This corporate standard defines logging, metrics, error wrapping, and connection pooling rules.\n"
               "All microservices communicating over gRPC or HTTPS must adhere to the standard definitions.\n"
               "The platform team continuously monitors adherence to these standards via telemetry probes.\n\n" * 25)
            + "\n--- CRITICAL HIDDEN SPECIFICATION 1 ---\n"
            + "MANDATORY DATABASE AUDIT TAG: Whenever initializing database or downstream RPC sessions, the connection header `x-sec-audit-tag` MUST be set strictly to the static constant `SEC_AUDIT_PROD_9981`. Any other value will trigger security revocation.\n\n"
            + ("# Section: Data Pipeline Standards\n"
               "Standard ETL patterns require idempotency keys and partitioned Kafka topics.\n"
               "Message deduplication windows are fixed at 3600 seconds.\n\n" * 25)
            + "\n--- CRITICAL HIDDEN SPECIFICATION 2 ---\n"
            + "MANDATORY RETRY BACKOFF FORMULA: Retries must NEVER use pure exponential backoff. The delay `d` in seconds must strictly follow the formula: `d = min(15.0, 0.25 * (2 ** attempt) + random.uniform(-0.05, 0.05))`.\n\n"
            + ("# Section: Observability and Error Propagation\n"
               "Distributed tracing spans must be passed across HTTP headers.\n"
               "Tracing identifiers follow OpenTelemetry standard baggage.\n\n" * 25)
            + "\n--- CRITICAL HIDDEN SPECIFICATION 3 ---\n"
            + "MANDATORY ERROR ENVELOPE: All exception responses must return a dictionary with exact keys `error_code`, `message`, and `x_custom_trace_id` where `x_custom_trace_id` is generated via `uuid.uuid5(uuid.NAMESPACE_DNS, f'error.{error_code}')`.\n\n"
            + "TASK: Now, write a Python class `PaymentGatewayClient` that implements an `execute_payment(user_id: str, amount_cents: int) -> dict` method honoring all 3 critical hidden specifications above (Audit Tag, Retry Formula, and Error Envelope). Highlight where each constraint is implemented."
        ),
        "max_tokens": 1200,
        "temperature": 0.2,
    }
]


def run_benchmark(base_url: str, output_file: str, model: str = "qwen3.8-27b-exl3-3.0bpw", api_key: str = ""):
    print(f"=== Starting Benchmark Suite against {base_url} (Model: {model}) ===")
    results = []

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    # Check models endpoint
    try:
        req = urllib.request.Request(f"{base_url}/models", headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            models_data = json.loads(resp.read().decode())
            print(f"Connected to endpoint. Models: {json.dumps(models_data)}")
    except Exception as e:
        print(f"Warning: Could not fetch models endpoint: {e}")

    for idx, task in enumerate(BENCHMARK_TASKS, 1):
        print(f"\n[{idx}/{len(BENCHMARK_TASKS)}] Running {task['id']} ({task['name']})...")
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": task["system"]},
                {"role": "user", "content": task["prompt"]},
            ],
            "max_tokens": task["max_tokens"],
            "temperature": task["temperature"],
            "stream": True,
            "stream_options": {"include_usage": True}
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=data,
            headers=headers,
        )

        t_start = time.perf_counter()
        t_first_token = None
        output_chunks = []
        token_count = 0
        completion_tokens = 0

        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                for line in response:
                    line = line.decode("utf-8", errors="replace").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except Exception:
                        continue

                    if "error" in chunk:
                        print(f"\n[ERROR from server]: {chunk['error']}", flush=True)
                        break

                    if "usage" in chunk and chunk["usage"]:
                        completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue

                    if choices[0].get("finish_reason") == "error":
                        err_msg = choices[0].get("message", {}).get("content", "Server error during generation")
                        print(f"\n[ERROR from model]: {err_msg}", flush=True)
                        break

                    delta = choices[0].get("delta", {})
                    text_piece = delta.get("content") or delta.get("reasoning_content") or ""
                    if text_piece:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        output_chunks.append(text_piece)
                        token_count += 1

            t_end = time.perf_counter()
            total_time = t_end - t_start
            ttft = (t_first_token - t_start) if t_first_token else total_time
            gen_time = (t_end - t_first_token) if t_first_token else total_time
            final_tokens = completion_tokens if completion_tokens > 0 else token_count
            tok_s = ((final_tokens - 1) / gen_time) if gen_time > 0 and final_tokens > 1 else (final_tokens / gen_time if gen_time > 0 else 0.0)

            full_output = "".join(output_chunks)

            print(f"  -> Generated {final_tokens} tokens in {total_time:.2f}s")
            print(f"  -> TTFT (prefill latency): {ttft*1000:.1f} ms")
            print(f"  -> Decode Speed: {tok_s:.2f} tok/s")

            task_res = {
                "task_id": task["id"],
                "name": task["name"],
                "category": task["category"],
                "tokens": final_tokens,
                "total_time_s": round(total_time, 3),
                "ttft_ms": round(ttft * 1000, 1),
                "tok_per_sec": round(tok_s, 2),
                "output_sample": full_output[:400] + "..." if len(full_output) > 400 else full_output,
                "full_output": full_output,
            }

            # Accuracy & Constraint checks
            if task["id"] == "task1_avl_tree":
                task_res["has_avl_class"] = "class AVLTree" in full_output or "class AvlTree" in full_output
                task_res["has_rotations"] = "rotate" in full_output.lower()
                task_res["has_test"] = "def test_avl" in full_output
            elif task["id"] == "task2_concurrency_debug":
                task_res["found_spurious_wakeup"] = "while" in full_output and ("wait" in full_output or "spurious" in full_output.lower())
                task_res["found_lock_context"] = "with self.lock" in full_output or "try:" in full_output
                task_res["found_memory_leak"] = "processed_items" in full_output
            elif task["id"] == "task4_long_context_constraints":
                task_res["spec1_audit_tag"] = "SEC_AUDIT_PROD_9981" in full_output
                task_res["spec2_retry_formula"] = "0.25 * (2 ** attempt)" in full_output or "0.25 *(2**attempt)" in full_output or "random.uniform(-0.05, 0.05)" in full_output
                task_res["spec3_uuid5"] = "uuid.uuid5" in full_output and "x_custom_trace_id" in full_output

            results.append(task_res)

        except Exception as e:
            print(f"  -> Error executing {task['id']}: {e}")
            results.append({"task_id": task["id"], "error": str(e)})

    # Summary
    avg_speed = sum(r.get("tok_per_sec", 0) for r in results if "tok_per_sec" in r) / max(1, len([r for r in results if "tok_per_sec" in r]))
    summary = {
        "base_url": base_url,
        "model": model,
        "avg_decode_tok_s": round(avg_speed, 2),
        "results": results,
    }

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n=== Benchmark completed! Saved to {output_file} (Avg: {avg_speed:.2f} tok/s) ===")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run comparison benchmark against OpenAI-compatible endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8888/v1", help="Base API URL (e.g. http://127.0.0.1:8888/v1)")
    parser.add_argument("--model", default="qwen3.8-27b-exl3-3.0bpw", help="Model name or ID")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key")
    parser.add_argument("--out", default="bench_results.json", help="Output JSON results path")
    args = parser.parse_args()

    run_benchmark(args.url, args.out, model=args.model, api_key=args.api_key)
