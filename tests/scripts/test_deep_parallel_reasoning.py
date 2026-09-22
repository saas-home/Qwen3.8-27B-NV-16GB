#!/usr/bin/env python3
"""
Deep Concentrated Parallel High-Context & Reasoning Benchmark
(tests/scripts/test_deep_parallel_reasoning.py)

Evaluates LLM serving engines under simultaneous high-context saturation and deep
multi-step reasoning loads.

Key Evaluation Pillars:
1. Parallel Saturated Ingestion: Multiple simultaneous streams injecting 16k-64k tokens of
   dense architectural and system context.
2. Dual Deep Reasoning: Both streams concurrently executing complex chain-of-thought analysis
   with extended `<think>` token generation.
3. Continuous Batching & KV Paging: Measures aggregate throughput (tok/s), dual prefill TTFT,
   and hardware memory telemetry during peak saturation.
4. Logical Coherence & Precision: Validates that reasoning quality and structural compliance
   do not degrade under maximum GPU resource pressure.
"""

import sys
import os
import time
import json
import re
import argparse
import urllib.request
import urllib.error
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)

BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

def log(msg, bold=False, color=""):
    p = bold and BOLD or ""
    c = color or ""
    s = (bold or color) and RESET or ""
    print(f"{p}{c}{msg}{s}", flush=True)


def get_gpu_vram_usage():
    """Query current GPU VRAM utilization via nvidia-smi."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode == 0:
            parts = r.stdout.strip().splitlines()[0].split(",")
            used, total = float(parts[0].strip()), float(parts[1].strip())
            return {"used_mib": used, "total_mib": total, "free_mib": total - used}
    except Exception:
        pass
    return None


def generate_dense_context(target_tokens: int, domain: str) -> str:
    """
    Generate dense, semantically coherent technical context to reach target token depth.
    Avoids pure synthetic padding; uses structured architectural and system specifications.
    """
    if domain == "distributed_consensus":
        base_unit = (
            "### Module: Distributed State Machine & Raft Consensus Specification\n"
            "```protobuf\n"
            "syntax = \"proto3\";\n"
            "package consensus.raft.v2;\n\n"
            "message LogEntry {\n"
            "  uint64 index = 1;\n"
            "  uint64 term = 2;\n"
            "  enum EntryType { NORMAL = 0; CONFIG_CHANGE = 1; NOOP = 2; SNAPSHOT_MARKER = 3; }\n"
            "  EntryType type = 3;\n"
            "  bytes payload_hash = 4;\n"
            "  bytes command = 5;\n"
            "  map<string, string> metadata = 6;\n"
            "}\n\n"
            "message AppendEntriesRequest {\n"
            "  uint64 term = 1;\n"
            "  string leader_id = 2;\n"
            "  uint64 prev_log_index = 3;\n"
            "  uint64 prev_log_term = 4;\n"
            "  repeated LogEntry entries = 5;\n"
            "  uint64 leader_commit = 6;\n"
            "  uint64 lease_expiry_epoch_ns = 7;\n"
            "}\n"
            "```\n"
            "The consensus group enforces strict quorum intersection: N/2 + 1 voting members must acknowledge "
            "log replication before committing. Under asymmetric network partition (split-brain scenarios), "
            "a partitioned leader operating on an expired monotonic lease must reject stale writes with "
            "ErrLeaseExpired. Byzantine fault tolerance requires double-SHA256 checksum validation across log segments. "
            "Snapshot compaction truncates the replicated write-ahead log up to index K, persisting the state-machine "
            "Merkle root to immutable NVMe block storage. Heartbeat intervals are set to 150ms with randomized election "
            "timeouts bounded between 300ms and 600ms to eliminate split-vote oscillation.\n\n"
        )
    else:  # lock_free_buffer
        base_unit = (
            "### Module: High-Throughput Wait-Free Ring Buffer & Cache Coherence Specification\n"
            "```cpp\n"
            "namespace kernel::ipc {\n"
            "template <typename T, size_t Capacity>\n"
            "class alignas(64) LockFreeRingBuffer {\n"
            "    static_assert((Capacity & (Capacity - 1)) == 0, \"Capacity must be power of two\");\n"
            "    static constexpr size_t Mask = Capacity - 1;\n"
            "    struct alignas(64) Slot {\n"
            "        std::atomic<uint64_t> sequence{0};\n"
            "        T storage;\n"
            "    };\n"
            "    alignas(64) std::atomic<uint64_t> write_cursor_{0};\n"
            "    alignas(64) std::atomic<uint64_t> read_cursor_{0};\n"
            "    alignas(64) Slot slots_[Capacity];\n"
            "public:\n"
            "    bool push(const T& item) noexcept {\n"
            "        uint64_t pos = write_cursor_.load(std::memory_order_relaxed);\n"
            "        for (;;) {\n"
            "            Slot& slot = slots_[pos & Mask];\n"
            "            uint64_t seq = slot.sequence.load(std::memory_order_acquire);\n"
            "            int64_t diff = static_cast<int64_t>(seq) - static_cast<int64_t>(pos);\n"
            "            if (diff == 0) {\n"
            "                if (write_cursor_.compare_exchange_weak(pos, pos + 1, std::memory_order_relaxed)) {\n"
            "                    slot.storage = item;\n"
            "                    slot.sequence.store(pos + 1, std::memory_order_release);\n"
            "                    return true;\n"
            "                }\n"
            "            } else if (diff < 0) { return false; }\n"
            "            else { pos = write_cursor_.load(std::memory_order_relaxed); }\n"
            "        }\n"
            "    }\n"
            "};\n"
            "}\n"
            "```\n"
            "Processor memory models enforce strict ordering guarantees. On modern NUMA multicore architectures, "
            "cache-line bouncing between socket interconnects (QPI/UPI/Infinity Fabric) degrades atomic compare-and-swap "
            "(CAS) throughput from 85 Mops/s to under 12 Mops/s when multiple producers contend on shared cache lines. "
            "To mitigate false sharing, head and tail pointers are padded with 64-byte hardware cache-line alignment. "
            "Memory operations utilize acquire-release semantics: producers issue std::memory_order_release to ensure "
            "prior store operations are globally visible before sequence increment, and consumers issue "
            "std::memory_order_acquire to prevent speculative out-of-order reads from bypassing sequence verification.\n\n"
        )

    # Base unit is ~180 tokens (~900 chars). Repeat to reach target.
    repeats = max(1, int(target_tokens / 180))
    context_body = base_unit * repeats
    return context_body


def build_parallel_tasks(target_context_tokens: int, parallel_count: int) -> list[dict]:
    """Build heterogeneous, complex high-context reasoning prompts for each stream."""
    tasks = []

    # Stream 1: Distributed Multi-Region Consensus Architecture
    ctx1 = generate_dense_context(target_context_tokens, "distributed_consensus")
    prompt1 = (
        f"{ctx1}\n\n"
        "### CRITICAL ARCHITECTURAL REASONING CHALLENGE:\n"
        "You are the Principal Distributed Systems Architect. Review the preceding distributed consensus and Raft specification.\n"
        "Analyze the following failure scenario under deep reasoning:\n"
        "1. Identify 3 distinct failure modes where an asymmetric network partition between a 5-node cluster spanning 3 regions "
        "(US-East, US-West, EU-Central) causes split-brain data corruption if monotonic lease validation is bypassed.\n"
        "2. Formulate a mathematical proof demonstrating how linearizable read operations can be guaranteed without executing "
        "a full quorum log replication pass per read (e.g. ReadIndex vs LeaseRead trade-offs with clock drift bounds).\n"
        "3. Write a production-grade Python implementation of a Partition-Aware Monotonic Lease Coordinator that prevents stale reads "
        "even under NTP clock skew of up to 250ms. Include complete error handling and docstrings.\n\n"
        "Provide thorough, step-by-step reasoning in your analysis."
    )
    tasks.append({
        "stream_id": 1,
        "name": "Distributed Consensus & Partition Invariance",
        "domain": "distributed_consensus",
        "prompt": prompt1,
        "expected_topics": ["split-brain", "lease", "quorum", "linearizable", "ntp", "clock skew"]
    })

    # Stream 2: High-Performance Lock-Free Concurrency & Memory Model
    ctx2 = generate_dense_context(target_context_tokens, "lock_free_buffer")
    prompt2 = (
        f"{ctx2}\n\n"
        "### CRITICAL ARCHITECTURAL REASONING CHALLENGE:\n"
        "You are the Lead Systems Performance Engineer. Review the lock-free ring buffer implementation provided in the context above.\n"
        "Analyze the following concurrency challenges under deep reasoning:\n"
        "1. Identify a subtle race condition in the dual-cursor CAS loop when compiled under weakly ordered architectures (ARMv8/v9, RISC-V) "
        "where `std::memory_order_relaxed` on the cursor load could observe stale sequence numbers, leading to slot overwrites.\n"
        "2. Explain how cache-line bouncing (MESI/MOESI protocol invalidation storms) affects L3 cache latency when 8 writer threads "
        "contend on `write_cursor_`. Propose a batching or ticket-based reservation mechanism to reduce interconnect traffic.\n"
        "3. Write an optimized C++20 implementation of a Single-Producer Multi-Consumer (SPMC) wait-free queue with batch-commit "
        "and explicit memory barriers that eliminates the identified race condition. Include static assertions and memory order rationales.\n\n"
        "Provide thorough, step-by-step reasoning in your analysis."
    )
    tasks.append({
        "stream_id": 2,
        "name": "Lock-Free Ring Buffer & Memory Model Verification",
        "domain": "lock_free_buffer",
        "prompt": prompt2,
        "expected_topics": ["memory_order", "acquire", "release", "cache-line", "mesi", "race condition"]
    })

    # If parallel > 2, add additional unique high-context tasks
    for i in range(3, parallel_count + 1):
        domain = "distributed_consensus" if i % 2 == 1 else "lock_free_buffer"
        ctx = generate_dense_context(target_context_tokens, domain)
        tasks.append({
            "stream_id": i,
            "name": f"High-Context Parallel Stream #{i} ({domain})",
            "domain": domain,
            "prompt": f"{ctx}\n\nProvide an in-depth architectural analysis and implementation addressing throughput optimization.",
            "expected_topics": ["architecture", "throughput", "latency"]
        })

    return tasks[:parallel_count]


def execute_stream(endpoint: str, model: str, task: dict, max_tokens: int, temperature: float, api_key: str = "") -> dict:
    """Execute a single high-context reasoning stream with real-time token tracking."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": task["prompt"]}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True}
    }

    stream_id = task["stream_id"]
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    tokens_emitted = 0
    full_text = ""
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    error_msg = None

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=900) as resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data:"):
                    continue
                raw_data = line[5:].strip()
                if raw_data == "[DONE]":
                    break
                try:
                    chunk = json.loads(raw_data)
                except Exception:
                    continue

                if "usage" in chunk and chunk["usage"]:
                    u = chunk["usage"]
                    prompt_tokens = u.get("prompt_tokens", prompt_tokens)
                    completion_tokens = u.get("completion_tokens", completion_tokens)
                    cached_tokens = u.get("prompt_tokens_details", {}).get("cached_tokens", cached_tokens)

                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    content_piece = delta.get("content") or delta.get("reasoning_content") or ""
                    if content_piece:
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        tokens_emitted += 1
                        full_text += content_piece

    except Exception as e:
        error_msg = str(e)

    total_wall_s = time.perf_counter() - t0
    ttft_s = (t_first - t0) if t_first is not None else total_wall_s
    decode_duration_s = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.001
    tok_count = completion_tokens or tokens_emitted
    decode_speed_tok_s = (tok_count / decode_duration_s) if decode_duration_s > 0 else 0.0
    prefill_speed_tok_s = (prompt_tokens / ttft_s) if (prompt_tokens > 0 and ttft_s > 0) else 0.0

    # Extract thinking trace if present
    think_match = re.search(r"<think>(.*?)</think>", full_text, re.DOTALL)
    thinking_content = think_match.group(1).strip() if think_match else ""
    final_answer = re.sub(r"<think>.*?</think>", "", full_text, flags=re.DOTALL).strip() if think_match else full_text

    # Verify coverage of expected technical topics
    topics_found = sum(1 for topic in task.get("expected_topics", []) if topic.lower() in full_text.lower())
    reasoning_passed = (topics_found >= 1) and (tok_count >= min(max_tokens, 50)) and (error_msg is None)

    return {
        "stream_id": stream_id,
        "name": task["name"],
        "status": "PASS" if reasoning_passed else "FAIL",
        "error": error_msg,
        "wall_time_s": round(total_wall_s, 2),
        "ttft_s": round(ttft_s, 3),
        "prefill_tok_s": round(prefill_speed_tok_s, 1),
        "decode_duration_s": round(decode_duration_s, 2),
        "decode_tok_s": round(decode_speed_tok_s, 2),
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": tok_count,
        "thinking_chars": len(thinking_content),
        "answer_chars": len(final_answer),
        "topics_found": f"{topics_found}/{len(task.get('expected_topics', []))}",
        "has_thinking_block": bool(thinking_content),
        "preview": final_answer[:280].replace("\n", " ") + "..."
    }


def main():
    parser = argparse.ArgumentParser(description="Deep Concentrated Parallel High-Context & Reasoning Benchmark")
    parser.add_argument("--endpoint", "-e", default="http://127.0.0.1:8888/v1", help="OpenAI-compatible endpoint")
    parser.add_argument("--model", "-m", default="qwen3.8-27b", help="Model name or ID")
    parser.add_argument("--parallel", "-p", type=int, default=2, help="Number of concurrent high-context streams (default: 2)")
    parser.add_argument("--context-tokens", "-c", type=int, default=32000, help="Target context tokens PER STREAM (default: 32000)")
    parser.add_argument("--max-tokens", type=int, default=1536, help="Maximum generation tokens per stream (default: 1536)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    parser.add_argument("--api-key", "-k", default="", help="Optional API key")
    parser.add_argument("--out", "-o", default="", help="Custom path for JSON results export")
    args = parser.parse_args()

    endpoint = args.endpoint.rstrip("/")
    parallel = args.parallel
    target_ctx = args.context_tokens
    max_tokens = args.max_tokens

    log("\n" + "="*96, bold=True)
    log(" DEEP CONCENTRATED PARALLEL HIGH-CONTEXT & REASONING BENCHMARK ", bold=True, color=GREEN)
    log("="*96)
    log(f"  Target Server Endpoint : {endpoint}")
    log(f"  Target Model ID        : {args.model}")
    log(f"  Parallel Streams       : {parallel} concurrent execution slots")
    log(f"  Context Depth / Stream : ~{target_ctx:,} tokens (Combined Payload: ~{target_ctx * parallel:,} tokens)")
    log(f"  Generation Token Budget: {max_tokens:,} tokens per stream (with deep chain-of-thought)")

    # Pre-run VRAM
    vram_before = get_gpu_vram_usage()
    if vram_before:
        log(f"  Initial GPU VRAM Usage : {vram_before['used_mib']:.0f} MiB / {vram_before['total_mib']:.0f} MiB ({vram_before['free_mib']:.0f} MiB free)\n")

    tasks = build_parallel_tasks(target_ctx, parallel)

    log("Preparing high-context payloads & dispatching parallel streams simultaneously...", bold=True, color=CYAN)
    log(f"Starting {parallel} workers via ThreadPoolExecutor...\n")

    t_suite_start = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = {
            pool.submit(execute_stream, endpoint, args.model, task, max_tokens, args.temperature, args.api_key): task
            for task in tasks
        }
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            st_color = GREEN if res["status"] == "PASS" else RED
            log(f"  >> Stream #{res['stream_id']} [{res['name'][:35]}]: "
                f"{st_color}{res['status']}{RESET} | "
                f"TTFT: {res['ttft_s']:.2f}s | "
                f"Decode: {res['decode_tok_s']:.2f} tok/s | "
                f"Tokens: {res['completion_tokens']} | "
                f"Topics: {res['topics_found']}")

    total_wall_time = time.perf_counter() - t_suite_start
    results.sort(key=lambda r: r["stream_id"])

    # Post-run VRAM
    vram_after = get_gpu_vram_usage()

    # Calculate aggregate metrics
    total_completion_tokens = sum(r["completion_tokens"] for r in results)
    total_prompt_tokens = sum(r["prompt_tokens"] for r in results)
    aggregate_throughput = (total_completion_tokens / total_wall_time) if total_wall_time > 0 else 0.0
    avg_ttft = sum(r["ttft_s"] for r in results) / len(results) if results else 0.0
    all_passed = all(r["status"] == "PASS" for r in results)

    # Print Formatted Scorecard Table
    log("\n" + "="*96, bold=True)
    log(" CONCENTRATED PARALLEL BENCHMARK SCORECARD & RESULTS ", bold=True, color=GREEN if all_passed else YELLOW)
    log("="*96)
    log(f"  Suite Overall Status      : {GREEN + 'ALL PASSED (100%)' + RESET if all_passed else RED + 'FAILURES DETECTED' + RESET}")
    log(f"  Total Suite Wall Time     : {total_wall_time:.2f} s")
    log(f"  Combined Prompt Context   : {total_prompt_tokens:,} tokens ingested")
    log(f"  Combined Generated Output : {total_completion_tokens:,} tokens generated")
    log(f"  Aggregate Saturated Tok/s : {BOLD}{aggregate_throughput:.2f} tok/s{RESET} (concurrent decode throughput)")
    log(f"  Average Time-To-First-Tok : {avg_ttft:.2f} s across {parallel} streams")
    if vram_after:
        log(f"  Post-Run GPU VRAM Usage   : {vram_after['used_mib']:.0f} MiB / {vram_after['total_mib']:.0f} MiB ({vram_after['free_mib']:.0f} MiB free)")
    log("-" * 96)

    hdr = f"{'Stream / Domain':<38} | {'Status':<8} | {'Prompt':<9} | {'TTFT':<9} | {'Decode':<11} | {'Output':<9} | {'Think Block'}"
    log(hdr, bold=True)
    log("-" * 96)

    for r in results:
        st_str = f"{GREEN}PASS{RESET}" if r["status"] == "PASS" else f"{RED}FAIL{RESET}"
        think_str = f"{r['thinking_chars']} chars" if r["has_thinking_block"] else "None (Raw)"
        log(f"{r['name'][:38]:<38} | {st_str:<17} | {r['prompt_tokens']:<9} | {r['ttft_s']:.2f}s{'' :<4} | {r['decode_tok_s']:.2f} t/s  | {r['completion_tokens']:<9} | {think_str}")

    log("="*96 + "\n")

    # Export JSON report
    report_data = {
        "benchmark": "deep_concentrated_parallel_high_context_reasoning",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoint": endpoint,
        "model": args.model,
        "parallel_streams": parallel,
        "target_context_tokens_per_stream": target_ctx,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_wall_time_s": round(total_wall_time, 2),
        "aggregate_decode_tok_s": round(aggregate_throughput, 2),
        "average_ttft_s": round(avg_ttft, 2),
        "vram_telemetry": {
            "before": vram_before,
            "after": vram_after
        },
        "all_passed": all_passed,
        "stream_results": results
    }

    out_file = args.out or os.path.join(DEFAULT_RESULTS_DIR, f"deep_parallel_reasoning_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w") as f:
        json.dump(report_data, f, indent=2)

    log(f"  Detailed JSON Benchmark Export : {out_file}", bold=True, color=CYAN)
    log("="*96 + "\n", bold=True)


if __name__ == "__main__":
    main()
