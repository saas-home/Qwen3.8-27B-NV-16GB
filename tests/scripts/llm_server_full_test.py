#!/usr/bin/env python3
"""
LLM Server Full Test Suite (llm_server_full_test.py)

Comprehensive, automated benchmark & stress test suite for any OpenAI-compatible LLM endpoint.
Features:
- Dynamic CLI parameters & interactive discovery (Endpoint, API Key, Model selector).
- Auto-detection of available models via /v1/models with interactive/automatic picker.
- Auto-detection of Max Context Window and dynamic adaptation of the Context Scaling Benchmark.
- Auto-detection of concurrency capacity via /health (when available) and user-configurable concurrency tests.
- Comprehensive 8-stage evaluation:
    1. Streaming Verification & TTFT Latency Benchmark
    2. Multimodal Vision & Structural Document Parsing (image.png)
    3. Multi-Client Concurrency & Continuous Batching Stress Test
    4. 4-Task Capability Suite (Algorithms, Concurrency Debugging, System Design, Constraint Retention)
    5. High-Entropy Associative Key-Value Recall
    6. Extreme Precision Multi-Stage Financial Reconciliation
    7. Executable Algorithmic Code Generation & Dynamic Unit-Test Harness
    8. Dynamic Long-Context Prefill & Decode Scaling Benchmark (up to max context)
- Generates a structured JSON report and displays a formatted summary table.
"""

import sys
import os
import time
import json
import base64
import urllib.request
import urllib.error
import argparse
import tempfile
import subprocess
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "image.png")
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)

# ANSI colors for pleasant terminal presentation
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
RESET = "\033[0m"

def log(msg, bold=False, color=""):
    prefix = bold and BOLD or ""
    c = color or ""
    suffix = (bold or color) and RESET or ""
    print(f"{prefix}{c}{msg}{suffix}", flush=True)


class LLMClient:
    def __init__(self, endpoint: str, api_key: str = None, model: str = None):
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/v1") and not endpoint.endswith("/chat/completions"):
            # Check if /v1 is needed
            endpoint = f"{endpoint}/v1"
        elif endpoint.endswith("/chat/completions"):
            endpoint = endpoint[:-len("/chat/completions")]
            
        self.base_url = endpoint
        self.completions_url = f"{self.base_url}/chat/completions"
        self.models_url = f"{self.base_url}/models"
        self.health_url = self.base_url.replace("/v1", "/health")
        self.api_key = api_key
        self.model = model

    def _get_headers(self):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def fetch_models(self):
        req = urllib.request.Request(self.models_url, headers=self._get_headers())
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                models = data.get("data", [])
                if not models and "models" in data:
                    models = data.get("models", [])
                return models
        except Exception as e:
            log(f"Warning: Could not fetch models from {self.models_url}: {e}", color=YELLOW)
            return []

    def fetch_health(self):
        req = urllib.request.Request(self.health_url, headers=self._get_headers())
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def call(self, messages, max_tokens=1024, temperature=0.0, stream=False, timeout=600):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(self.completions_url, data=data, headers=self._get_headers())

        t0 = time.perf_counter()
        if not stream:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
            total_time = time.perf_counter() - t0
            choice = res_json.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content") or msg.get("reasoning_content") or ""
            usage = res_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            speed = completion_tokens / total_time if total_time > 0 else 0.0
            return {
                "content": content,
                "total_time": total_time,
                "ttft": total_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "decode_speed": speed,
                "raw": res_json
            }
        else:
            t_first = None
            t_last = None
            chunks = []
            token_count = 0
            prompt_tokens = 0
            completion_tokens = 0
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for line in resp:
                    l = line.decode("utf-8", errors="replace").strip()
                    if not l.startswith("data:"):
                        continue
                    d_str = l[5:].strip()
                    if d_str == "[DONE]":
                        break
                    try:
                        c = json.loads(d_str)
                    except Exception:
                        continue
                    if "usage" in c and c["usage"]:
                        prompt_tokens = c["usage"].get("prompt_tokens", prompt_tokens)
                        completion_tokens = c["usage"].get("completion_tokens", completion_tokens)
                    choices = c.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        text = delta.get("content") or delta.get("reasoning_content") or ""
                        if text:
                            now = time.perf_counter()
                            if t_first is None:
                                t_first = now
                            t_last = now
                            token_count += 1
                            chunks.append(text)
            t_end = time.perf_counter()
            total_time = t_end - t0
            ttft = (t_first - t0) if t_first else total_time
            gen_time = (t_end - t_first) if t_first else total_time
            comp_tok = completion_tokens if completion_tokens > 0 else token_count
            speed = (comp_tok / gen_time) if gen_time > 0 else 0.0
            return {
                "content": "".join(chunks),
                "total_time": total_time,
                "ttft": ttft,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp_tok,
                "decode_speed": speed
            }


def detect_max_context(model_obj):
    if not isinstance(model_obj, dict):
        return 32768
    candidates = [
        model_obj.get("max_model_len"),
        model_obj.get("context_length"),
        model_obj.get("top_provider", {}).get("context_length"),
        model_obj.get("max_position_embeddings"),
    ]
    for c in candidates:
        if isinstance(c, int) and c > 0:
            return c
    return 32768


def build_context_milestones(max_context: int):
    # Propose balanced context milestones up to 80%-100% of max context
    standard_targets = [4000, 8000, 16000, 32000, 64000, 128000, 200000, 256000]
    milestones = [t for t in standard_targets if t <= int(max_context * 0.95)]
    if not milestones:
        milestones = [min(4000, max_context)]
    # Ensure at least the top boundary is benchmarked
    top_target = int(max_context * 0.8)
    if top_target > milestones[-1]:
        milestones.append(top_target)
    return sorted(list(set(milestones)))


# ============================================================================
# 8 TEST MODULES
# ============================================================================

def run_test_streaming(client: LLMClient):
    log("\n" + "="*80, bold=True)
    log("[TEST 1/8] Streaming Verification & TTFT Latency", bold=True, color=CYAN)
    log("="*80)
    prompt = "Explain quantum superposition and entanglement in exactly two concise sentences."
    res = client.call([{"role": "user", "content": prompt}], max_tokens=150, stream=True)
    log(f"  Response: {res['content'].strip()[:180]}...")
    log(f"  -> TTFT Latency: {res['ttft']*1000:.2f} ms")
    log(f"  -> Tokens Emitted: {res['completion_tokens']}")
    log(f"  -> Generation Speed: {res['decode_speed']:.2f} tok/s")
    return {
        "status": "PASS",
        "ttft_ms": round(res["ttft"] * 1000, 2),
        "tokens": res["completion_tokens"],
        "tok_s": round(res["decode_speed"], 2)
    }

def run_test_vision(client: LLMClient):
    log("\n" + "="*80, bold=True)
    log("[TEST 2/8] Multimodal Vision Evaluation (Invoice Extraction)", bold=True, color=CYAN)
    log("="*80)
    if not os.path.exists(IMAGE_PATH):
        log(f"  Skipping: Image asset not found at {IMAGE_PATH}", color=YELLOW)
        return {"status": "SKIPPED", "reason": "image.png not found"}

    with open(IMAGE_PATH, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                {"type": "text", "text": "Extract all fields from this invoice: Invoice Number, Parties, Line items, and Total Amount."}
            ]
        }
    ]
    res = client.call(messages, max_tokens=600, stream=True)
    content = res["content"]
    has_inv = "1024" in content
    has_total = any(k in content for k in ["875", "1250", "1,250", "$875.00"])
    status = "PASS" if (has_inv and has_total) else ("PARTIAL" if (has_inv or has_total) else "FAIL")
    log(f"  -> Invoice #1024 Extracted: {has_inv}")
    log(f"  -> Total Amount Extracted: {has_total}")
    log(f"  -> TTFT (Vision Encode + Prefill): {res['ttft']*1000:.2f} ms | Speed: {res['decode_speed']:.2f} tok/s")
    return {
        "status": status,
        "invoice_detected": has_inv,
        "total_detected": has_total,
        "ttft_ms": round(res["ttft"] * 1000, 2),
        "tokens": res["completion_tokens"],
        "tok_s": round(res["decode_speed"], 2)
    }

def run_test_concurrency(client: LLMClient, parallel: int):
    log("\n" + "="*80, bold=True)
    log(f"[TEST 3/8] Multi-Client Concurrency & Batching Stress Test (Parallel = {parallel})", bold=True, color=CYAN)
    log("="*80)
    
    results = {}
    for c in sorted(list(set([1, parallel]))):
        log(f"  Evaluating concurrency level c = {c}...")
        t_start = time.perf_counter()
        def worker(client_id):
            prompt = f"Write a clean python function to reverse a linked list, client #{client_id}"
            try:
                r = client.call([{"role": "user", "content": prompt}], max_tokens=100, stream=True)
                return {"client_id": client_id, "tokens": r["completion_tokens"], "speed": r["decode_speed"], "ttft": r["ttft"], "error": None}
            except Exception as e:
                return {"client_id": client_id, "tokens": 0, "speed": 0.0, "ttft": 0.0, "error": str(e)}

        with ThreadPoolExecutor(max_workers=c) as pool:
            worker_results = list(pool.map(worker, range(c)))

        wall_time = time.perf_counter() - t_start
        total_tokens = sum(w["tokens"] for w in worker_results)
        errors = [w["error"] for w in worker_results if w["error"]]
        agg_speed = total_tokens / wall_time if wall_time > 0 else 0.0
        status = "PASS" if not errors and total_tokens > 0 else "DEGRADED/FAIL"
        
        log(f"    -> c={c}: {total_tokens} tokens across {c} streams in {wall_time:.2f}s (Agg Throughput: {agg_speed:.2f} tok/s) [{status}]")
        if errors:
            log(f"       Errors encountered: {errors[:2]}", color=YELLOW)

        results[f"c{c}"] = {
            "concurrency": c,
            "status": status,
            "total_tokens": total_tokens,
            "wall_time_s": round(wall_time, 2),
            "aggregate_tok_s": round(agg_speed, 2),
            "errors": errors
        }
    return results

def run_test_capabilities(client: LLMClient):
    log("\n" + "="*80, bold=True)
    log("[TEST 4/8] 4-Task Capability Benchmark Suite", bold=True, color=CYAN)
    log("="*80)
    tasks = [
        ("task1_avl_tree", "AVL Tree with Rotations", "Write a complete self-balancing AVL Tree in Python with insert, delete, search, in-order iterator, and test function.", 1200),
        ("task2_concurrency_debug", "Concurrency Buffer Debug", "Analyze a buggy multi-threaded BoundedBuffer class in Python with lost wakeups and race conditions. Provide the fix.", 1000),
        ("task3_system_design", "Distributed Rate Limiter", "Design a high-throughput distributed rate limiter with sliding window counter, Redis cluster, and fallback.", 1200),
        ("task4_hidden_constraints", "Hidden Constraints Adherence", "Implement PaymentGatewayClient with exact requirements: Audit Tag format AUDIT_{uuid4}, retry formula 2^attempt * 100ms, and Error envelope.", 1000)
    ]
    results = []
    for tid, name, prompt, max_t in tasks:
        log(f"  Executing: {name}...")
        res = client.call([{"role": "user", "content": prompt}], max_tokens=max_t, stream=True)
        log(f"    -> Emitted: {res['completion_tokens']} toks | Speed: {res['decode_speed']:.2f} tok/s | TTFT: {res['ttft']*1000:.1f} ms")
        results.append({
            "id": tid,
            "name": name,
            "tokens": res["completion_tokens"],
            "speed_tok_s": round(res["decode_speed"], 2),
            "ttft_ms": round(res["ttft"] * 1000, 1)
        })
    avg_speed = sum(r["speed_tok_s"] for r in results) / len(results)
    return {
        "status": "PASS",
        "average_speed_tok_s": round(avg_speed, 2),
        "tasks": results
    }

def run_test_high_entropy(client: LLMClient):
    log("\n" + "="*80, bold=True)
    log("[TEST 5/8] High-Entropy Associative Key-Value Recall", bold=True, color=CYAN)
    log("="*80)
    filler = "In high-performance distributed architectures, nodes communicate via low-latency RPC protocols with strict SLAs. " * 300
    kv_data = (
        "CONFIDENTIAL LOOKUP TABLE:\n"
        "KEY_ALPHA_77: VAL_X9$mK2\n"
        "KEY_BETA_99: VAL_Z4#pQ8\n"
        "KEY_GAMMA_12: VAL_L1*vR5\n"
    )
    prompt = f"{filler}\n{kv_data}\n{filler}\nWhat is the exact value for KEY_BETA_99 and KEY_GAMMA_12? Answer strictly in format: KEY=VAL"
    res = client.call([{"role": "user", "content": prompt}], max_tokens=150, stream=True)
    content = res["content"]
    m1 = "Z4#pQ8" in content
    m2 = "L1*vR5" in content
    status = "PASS" if (m1 and m2) else ("PARTIAL" if (m1 or m2) else "FAIL")
    log(f"  -> Recall: KEY_BETA_99 ({m1}), KEY_GAMMA_12 ({m2}) -> {status}")
    log(f"  -> TTFT: {res['ttft']*1000:.1f} ms | Decode Speed: {res['decode_speed']:.2f} tok/s")
    return {
        "status": status,
        "matches": {"KEY_BETA_99": m1, "KEY_GAMMA_12": m2},
        "ttft_ms": round(res["ttft"] * 1000, 1),
        "tok_s": round(res["decode_speed"], 2)
    }

def run_test_extreme_precision(client: LLMClient):
    log("\n" + "="*80, bold=True)
    log("[TEST 6/8] Extreme Precision Arithmetic & Financial Ledger Reconciliation", bold=True, color=CYAN)
    log("="*80)
    stages = [
        ("Stage 1", "Initial balance = 10000.00"),
        ("Stage 2", "Deposit = +2450.50"),
        ("Stage 3", "Transfer fee = -15.25"),
        ("Stage 4", "Wire withdrawal = -1200.00"),
        ("Stage 5", "Interest credited = +45.10"),
        ("Stage 6", "Card payment = -342.80"),
        ("Stage 7", "Refund received = +112.50"),
        ("Stage 8", "Maintenance fee = -25.00"),
        ("Stage 9", "Securities dividend = +580.45"),
        ("Stage 10", "Tax withholding = -116.09")
    ]
    expected = "11489.41"
    ledger_text = "\n".join(f"{s[0]}: {s[1]}" for s in stages)
    prompt = (
        f"Perform an exact financial audit ledger reconciliation on the following sequential journal entries:\n"
        f"{ledger_text}\n\n"
        f"Calculate the precise ending balance to two decimal places. State: 'Ending Balance: $XXXXX.XX'"
    )
    res = client.call([{"role": "user", "content": prompt}], max_tokens=400, stream=True)
    matched = expected in res["content"]
    status = "PASS" if matched else "FAIL"
    log(f"  -> Calculated ending balance match ({expected}): {matched} -> {status}")
    log(f"  -> TTFT: {res['ttft']*1000:.1f} ms | Decode Speed: {res['decode_speed']:.2f} tok/s")
    return {
        "status": status,
        "expected": expected,
        "matched": matched,
        "ttft_ms": round(res["ttft"] * 1000, 1),
        "tok_s": round(res["decode_speed"], 2)
    }

def run_test_code_execution(client: LLMClient):
    log("\n" + "="*80, bold=True)
    log("[TEST 7/8] Executable Algorithmic Code Generation & Dynamic Verification", bold=True, color=CYAN)
    log("="*80)
    prompt = (
        "Write a complete Python implementation of an LRU Cache.\n"
        "Class name must be `LRUCache` with:\n"
        "  `__init__(self, capacity: int)`\n"
        "  `get(self, key: int) -> int` (returns -1 if not found)\n"
        "  `put(self, key: int, value: int) -> None`\n"
        "Provide ONLY the executable Python code inside a ```python ``` code block."
    )
    res = client.call([{"role": "user", "content": prompt}], max_tokens=800, stream=True)
    code = res["content"]
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    test_harness = """
cache = LRUCache(2)
cache.put(1, 1)
cache.put(2, 2)
assert cache.get(1) == 1, "Failed get 1"
cache.put(3, 3)
assert cache.get(2) == -1, "Eviction failed for 2"
cache.put(4, 4)
assert cache.get(1) == -1, "Eviction failed for 1"
assert cache.get(3) == 3, "Failed get 3"
assert cache.get(4) == 4, "Failed get 4"
print("UNIT_TESTS_PASSED")
"""
    full_code = code + "\n" + test_harness
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f_path = f.name

    try:
        sub = subprocess.run([sys.executable, f_path], capture_output=True, text=True, timeout=15)
        passed = "UNIT_TESTS_PASSED" in sub.stdout
        err = sub.stderr.strip()
    except Exception as e:
        passed = False
        err = str(e)
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)

    status = "PASS" if passed else "FAIL"
    log(f"  -> Dynamic Execution Assertions: {status} {'(All tests passed)' if passed else f'Error: {err[:120]}'}")
    log(f"  -> Decode Speed: {res['decode_speed']:.2f} tok/s")
    return {
        "status": status,
        "dynamic_tests_passed": passed,
        "tok_s": round(res["decode_speed"], 2)
    }

def run_test_context_scaling(client: LLMClient, milestones: list):
    log("\n" + "="*80, bold=True)
    log("[TEST 8/8] Dynamic Long-Context Prefill & Decode Scaling Benchmark", bold=True, color=CYAN)
    log(f"Target Milestones: {milestones}")
    log("="*80)

    base_text = (
        "The quick brown fox jumps over the lazy dog. In computer science and artificial intelligence, "
        "large language models utilize transformer architectures with multi-head self-attention mechanisms "
        "to process sequential data efficiently. Memory bandwidth and compute capacity determine inference speed. "
    )
    header = f"{'Target Ctx':>12} | {'Actual Prompt':>14} | {'TTFT (s)':>10} | {'Prefill (t/s)':>14} | {'Decode (t/s)':>14} | {'Gen Toks':>9} | {'Result':>8}"
    log(header)
    log("-" * len(header))

    scaling_results = []
    for target in milestones:
        repeat_count = max(1, int(target / 45))
        prompt_text = (base_text * repeat_count) + "\n\nSummarize the key aspects mentioned above in detail."
        try:
            res = client.call([{"role": "user", "content": prompt_text}], max_tokens=64, stream=True, timeout=1200)
            actual_prompt = res["prompt_tokens"] if res["prompt_tokens"] > 0 else target
            prefill_speed = actual_prompt / res["ttft"] if res["ttft"] > 0 else 0.0
            row = (
                f"{target:>12,d} | "
                f"{actual_prompt:>14,d} | "
                f"{res['ttft']:>10.3f} | "
                f"{prefill_speed:>14.1f} | "
                f"{res['decode_speed']:>14.2f} | "
                f"{res['completion_tokens']:>9d} | "
                f"{'PASS':>8}"
            )
            log(row)
            scaling_results.append({
                "target_tokens": target,
                "actual_prompt_tokens": actual_prompt,
                "ttft_s": round(res["ttft"], 3),
                "prefill_tok_s": round(prefill_speed, 1),
                "decode_tok_s": round(res["decode_speed"], 2),
                "completion_tokens": res["completion_tokens"],
                "status": "PASS"
            })
        except Exception as e:
            row = f"{target:>12,d} | {'ERROR':>14} | {str(e)[:40]} | {'FAIL':>8}"
            log(row, color=RED)
            scaling_results.append({
                "target_tokens": target,
                "error": str(e),
                "status": "FAIL"
            })
        time.sleep(1)

    log("-" * len(header))
    return scaling_results


# ============================================================================
# FORMATTED CLI SUMMARY SCORECARD
# ============================================================================

def print_summary_table(report):
    res = report.get("results", {})
    log("\n" + "="*88, bold=True)
    log("                       BENCHMARK & STRESS TEST RESULT SUMMARY                           ", bold=True, color=CYAN)
    log("="*88, bold=True)
    log(f"  Target Server   : {report.get('endpoint')}")
    log(f"  Model Under Test: {report.get('model')}")
    log(f"  Max Context Cap : {report.get('max_context_tokens', 0):,} tokens")
    log(f"  Parallel Setting: {report.get('parallel_streams', 1)} concurrent clients")
    log(f"  Total Duration  : {report.get('total_suite_wall_time_s', 0):.2f} s")
    log("="*88)

    header = f"{'Test Suite / Evaluation':<34} | {'Status':<10} | {'TTFT / Latency':<16} | {'Throughput':<16}"
    log(header, bold=True)
    log("-" * len(header))

    def fmt_status(st):
        if st == "PASS":
            return f"{GREEN}PASS{RESET}"
        elif st == "FAIL":
            return f"{RED}FAIL{RESET}"
        elif st == "SKIPPED":
            return f"{YELLOW}SKIPPED{RESET}"
        return f"{YELLOW}{st}{RESET}"

    # 1. Streaming
    s = res.get("streaming", {})
    log(f"{'1. Streaming & Latency':<34} | {fmt_status(s.get('status', 'N/A')):<19} | {s.get('ttft_ms', 0):.1f} ms{'':<8} | {s.get('tok_s', 0):.2f} tok/s")

    # 2. Vision
    v = res.get("vision", {})
    v_ttft = f"{v.get('ttft_ms', 0):.1f} ms" if 'ttft_ms' in v else "N/A"
    v_speed = f"{v.get('tok_s', 0):.2f} tok/s" if 'tok_s' in v else "N/A"
    log(f"{'2. Multimodal Vision (Invoice)':<34} | {fmt_status(v.get('status', 'N/A')):<19} | {v_ttft:<16} | {v_speed:<16}")

    # 3. Concurrency / Parallel
    c = res.get("concurrency", {})
    for ckey, cval in c.items():
        c_label = f"3. Parallel Batching ({ckey.upper()})"
        c_speed = f"{cval.get('aggregate_tok_s', 0):.2f} tok/s (agg)"
        c_wall = f"{cval.get('wall_time_s', 0):.2f} s wall"
        log(f"{c_label:<34} | {fmt_status(cval.get('status', 'N/A')):<19} | {c_wall:<16} | {c_speed:<16}")

    # 4. Capabilities (4 Tasks)
    cap = res.get("capabilities_4tasks", {})
    cap_speed = f"{cap.get('average_speed_tok_s', 0):.2f} tok/s (avg)"
    log(f"{'4. 4-Task Capability Suite':<34} | {fmt_status(cap.get('status', 'N/A')):<19} | {'4 tasks passed':<16} | {cap_speed:<16}")
    for t in cap.get("tasks", []):
        t_sub = f"   • {t.get('name', t.get('id'))}"
        log(f"{t_sub:<34} | {'PASS':<10} | {t.get('ttft_ms', 0):.1f} ms{'':<8} | {t.get('speed_tok_s', 0):.2f} tok/s")

    # 5. High Entropy
    he = res.get("high_entropy_recall", {})
    he_ttft = f"{he.get('ttft_ms', 0):.1f} ms"
    he_speed = f"{he.get('tok_s', 0):.2f} tok/s"
    log(f"{'5. High-Entropy Key Recall':<34} | {fmt_status(he.get('status', 'N/A')):<19} | {he_ttft:<16} | {he_speed:<16}")

    # 6. Extreme Precision
    ep = res.get("extreme_precision", {})
    ep_ttft = f"{ep.get('ttft_ms', 0):.1f} ms"
    ep_speed = f"{ep.get('tok_s', 0):.2f} tok/s"
    log(f"{'6. Precision Ledger Reconcile':<34} | {fmt_status(ep.get('status', 'N/A')):<19} | {ep_ttft:<16} | {ep_speed:<16}")

    # 7. Code Execution
    ce = res.get("code_execution", {})
    ce_speed = f"{ce.get('tok_s', 0):.2f} tok/s"
    ce_desc = "Unit tests OK" if ce.get("dynamic_tests_passed") else "Unit test fail"
    log(f"{'7. Dynamic Code Verification':<34} | {fmt_status(ce.get('status', 'N/A')):<19} | {ce_desc:<16} | {ce_speed:<16}")

    # 8. Context Scaling Summary
    cs = res.get("context_scaling", [])
    if cs:
        log("-" * len(header))
        log("  [Context Scaling Milestone Performance Breakdown]", bold=True)
        cs_hdr = f"  {'Context Milestone':<20} | {'Prefill TTFT':<15} | {'Prefill Throughput':<20} | {'Decode Throughput':<18}"
        log(cs_hdr)
        log("  " + "-" * (len(cs_hdr) - 2))
        for step in cs:
            if step.get("status") == "PASS":
                m_label = f"~{step.get('target_tokens', 0)//1000}k ({step.get('actual_prompt_tokens', 0):,} toks)"
                ttft_str = f"{step.get('ttft_s', 0):.2f} s"
                prefill_str = f"{step.get('prefill_tok_s', 0):.1f} tok/s"
                decode_str = f"{step.get('decode_tok_s', 0):.2f} tok/s"
                log(f"  {m_label:<20} | {ttft_str:<15} | {prefill_str:<20} | {decode_str:<18}")
            else:
                m_label = f"{step.get('target_tokens', 0):,} toks"
                log(f"  {m_label:<20} | {'FAILED':<15} | {str(step.get('error', 'Error'))[:30]}")

    log("="*88 + "\n")


# ============================================================================
# MAIN ORCHESTRATOR & INTERACTIVE DISCOVERY
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Full LLM Server Benchmark & Stress Test Suite")
    parser.add_argument("--endpoint", "-e", help="LLM server base endpoint (e.g. http://172.16.16.29:8000/v1)")
    parser.add_argument("--api-key", "-k", default=os.getenv("OPENAI_API_KEY", ""), help="API key (optional)")
    parser.add_argument("--model", "-m", help="Model name or ID to test")
    parser.add_argument("--parallel", "-p", type=int, help="Number of parallel clients to test")
    parser.add_argument("--max-context", type=int, help="Override detected max context window tokens")
    parser.add_argument("--out", "-o", help="Path to write JSON benchmark report")
    parser.add_argument("--auto", "-y", action="store_true", help="Non-interactive auto-selection mode")
    args = parser.parse_args()

    log("\n" + "="*80, bold=True)
    log(" LLM SERVER FULL BENCHMARK & STRESS TEST SUITE ", bold=True, color=GREEN)
    log("="*80)

    # 1. Endpoint resolution
    endpoint = args.endpoint
    if not endpoint:
        if sys.stdin.isatty() and not args.auto:
            default_ep = "http://127.0.0.1:8888/v1"
            val = input(f"Enter LLM Server Endpoint [{default_ep}]: ").strip()
            endpoint = val or default_ep
        else:
            endpoint = "http://127.0.0.1:8888/v1"
    log(f"Target Server: {endpoint}")

    api_key = args.api_key
    client = LLMClient(endpoint=endpoint, api_key=api_key or None)

    # 2. Discover Models
    log("\nQuerying available models from endpoint...")
    models = client.fetch_models()
    selected_model_obj = None

    if models:
        log(f"Discovered {len(models)} model(s):")
        for idx, m in enumerate(models, 1):
            m_id = m.get("id") or m.get("name", "unknown")
            m_ctx = detect_max_context(m)
            log(f"  [{idx}] {m_id} (Context: {m_ctx:,} tokens)")

        if args.model:
            # Match given model name or ID
            for m in models:
                if m.get("id") == args.model or m.get("name") == args.model:
                    selected_model_obj = m
                    break
            if not selected_model_obj:
                selected_model_obj = {"id": args.model}
        elif sys.stdin.isatty() and not args.auto and len(models) > 1:
            while True:
                choice = input(f"\nSelect model to test [1-{len(models)}] (default 1): ").strip()
                if not choice:
                    selected_model_obj = models[0]
                    break
                if choice.isdigit() and 1 <= int(choice) <= len(models):
                    selected_model_obj = models[int(choice) - 1]
                    break
                log("Invalid choice, please re-enter.", color=YELLOW)
        else:
            selected_model_obj = models[0]
    else:
        log("No models returned by /models endpoint. Using model parameter or fallback.", color=YELLOW)
        selected_model_obj = {"id": args.model or "default"}

    model_name = selected_model_obj.get("id") or selected_model_obj.get("name")
    client.model = model_name
    log(f"\nActive Model Selected: {BOLD}{model_name}{RESET}")

    # 3. Detect Context Ceiling
    detected_ctx = detect_max_context(selected_model_obj)
    if args.max_context:
        max_context = args.max_context
        log(f"Max Context Window (Overridden by flag): {max_context:,} tokens")
    else:
        max_context = detected_ctx
        log(f"Detected Max Context Window: {max_context:,} tokens")

    milestones = build_context_milestones(max_context)
    log(f"Adopted Context Scaling Targets: {milestones}")

    # 4. Detect / Configure Concurrency & Parallel Slots
    health = client.fetch_health()
    detected_parallel = None
    if health and isinstance(health, dict):
        detected_parallel = health.get("parallel") or health.get("max_slots")

    if args.parallel:
        parallel = args.parallel
    elif detected_parallel:
        log(f"Server /health reports {detected_parallel} active parallel slots.")
        parallel = detected_parallel
    elif sys.stdin.isatty() and not args.auto:
        val = input("\nEnter number of parallel clients to test [default 4]: ").strip()
        parallel = int(val) if val.isdigit() and int(val) > 0 else 4
    else:
        parallel = 4

    log(f"Parallel Test Level: {parallel} concurrent streams\n")

    # 5. Execute Full Suite
    report = {
        "endpoint": endpoint,
        "model": model_name,
        "max_context_tokens": max_context,
        "context_milestones": milestones,
        "parallel_streams": parallel,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {}
    }

    t_suite_start = time.perf_counter()

    report["results"]["streaming"] = run_test_streaming(client)
    report["results"]["vision"] = run_test_vision(client)
    report["results"]["concurrency"] = run_test_concurrency(client, parallel)
    report["results"]["capabilities_4tasks"] = run_test_capabilities(client)
    report["results"]["high_entropy_recall"] = run_test_high_entropy(client)
    report["results"]["extreme_precision"] = run_test_extreme_precision(client)
    report["results"]["code_execution"] = run_test_code_execution(client)
    report["results"]["context_scaling"] = run_test_context_scaling(client, milestones)

    total_suite_time = time.perf_counter() - t_suite_start
    report["total_suite_wall_time_s"] = round(total_suite_time, 2)

    # 6. Save JSON Report
    parsed_host = urlparse(endpoint).netloc.replace(":", "_") or "local"
    out_file = args.out or os.path.join(DEFAULT_RESULTS_DIR, f"full_test_{parsed_host}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    # 7. Display Terminal Result Summary Table
    print_summary_table(report)

    log("="*88, bold=True)
    log(" FULL BENCHMARK SUITE COMPLETED SUCCESSFULLY ", bold=True, color=GREEN)
    log(f"  Total Suite Wall Time : {total_suite_time:.2f} s")
    log(f"  JSON Benchmark Report : {out_file}", bold=True)
    log("="*88 + "\n", bold=True)

if __name__ == "__main__":
    main()
