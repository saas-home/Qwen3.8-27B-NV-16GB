#!/usr/bin/env python3
"""
Comprehensive Remote LLM Benchmark & Test Runner
Executes the full suite against any OpenAI-compatible endpoint.

Target:
  Endpoint: http://172.16.16.29:8000/v1
  Model: Qwen3.8 Flash Next
"""

import sys
import os
import time
import json
import base64
import urllib.request
import urllib.error
import tempfile
import subprocess
from concurrent.futures import ThreadPoolExecutor

BASE_URL = "http://172.16.16.29:8000/v1"
COMPLETIONS_URL = f"{BASE_URL}/chat/completions"
MODELS_URL = f"{BASE_URL}/models"
DEFAULT_MODEL = "Qwen3.8 Flash Next"
IMAGE_PATH = os.path.join(os.path.dirname(__file__), "image.png")
OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_FILE = os.path.join(OUT_DIR, "remote_test_suite_results.json")

results_summary = {
    "target_url": BASE_URL,
    "model": DEFAULT_MODEL,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "tests": {}
}

def log(msg):
    print(msg, flush=True)

def call_api(messages, max_tokens=1024, temperature=0.0, stream=False, model=DEFAULT_MODEL, timeout=600):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(COMPLETIONS_URL, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.perf_counter()
    if not stream:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
        total_time = time.perf_counter() - t0
        choice = res_json.get("choices", [{}])[0]
        msg = choice.get("message", {})
        content = msg.get("content") or msg.get("reasoning_content") or ""
        usage = res_json.get("usage", {})
        return {
            "content": content,
            "raw": res_json,
            "total_time": total_time,
            "ttft": total_time,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "decode_speed": usage.get("completion_tokens", 0) / total_time if total_time > 0 else 0.0
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

# ============================================================================
# 1. STREAMING LATENCY TEST
# ============================================================================
def test_streaming():
    log("\n" + "="*80)
    log("[TEST 1/7] Streaming & Latency Benchmark")
    log("="*80)
    prompt = "Explain quantum computing and superposition in exactly two sentences."
    res = call_api([{"role": "user", "content": prompt}], max_tokens=150, stream=True)
    log(f"Response: {res['content'].strip()[:200]}...")
    log(f"  -> TTFT: {res['ttft']*1000:.2f} ms")
    log(f"  -> Tokens Generated: {res['completion_tokens']}")
    log(f"  -> Decode Speed: {res['decode_speed']:.2f} tok/s")
    results_summary["tests"]["streaming"] = {
        "status": "PASS",
        "ttft_ms": round(res["ttft"]*1000, 2),
        "tokens": res["completion_tokens"],
        "tok_s": round(res["decode_speed"], 2)
    }

# ============================================================================
# 2. MULTIMODAL VISION TEST
# ============================================================================
def test_vision():
    log("\n" + "="*80)
    log("[TEST 2/7] Multimodal Vision Evaluation (image.png)")
    log("="*80)
    if not os.path.exists(IMAGE_PATH):
        log("  Skipping: image.png not found")
        results_summary["tests"]["vision"] = {"status": "SKIPPED"}
        return

    with open(IMAGE_PATH, "rb") as f:
        b64_img = base64.b64encode(f.read()).decode("utf-8")
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}},
                {"type": "text", "text": "Extract all fields from this invoice: Invoice Number, Billed To, Line Items, and Total Amount."}
            ]
        }
    ]
    res = call_api(messages, max_tokens=600, stream=True)
    content = res["content"]
    has_inv = "1024" in content
    has_total = "875" in content or "1250" in content or "1,250" in content
    status = "PASS" if (has_inv and has_total) else "PARTIAL"
    log(f"  -> Invoice #1024 Detected: {has_inv}")
    log(f"  -> Total Detected: {has_total}")
    log(f"  -> TTFT: {res['ttft']*1000:.2f} ms | Decode Speed: {res['decode_speed']:.2f} tok/s")
    results_summary["tests"]["vision"] = {
        "status": status,
        "ttft_ms": round(res["ttft"]*1000, 2),
        "tokens": res["completion_tokens"],
        "tok_s": round(res["decode_speed"], 2),
        "invoice_detected": has_inv,
        "total_detected": has_total
    }

# ============================================================================
# 3. CONCURRENCY & BATCHING TEST
# ============================================================================
def test_concurrency():
    log("\n" + "="*80)
    log("[TEST 3/7] Concurrency & Continuous Batching Test (c=1, c=2)")
    log("="*80)
    for c in [1, 2]:
        log(f"  Testing concurrency level c={c}...")
        t_start = time.perf_counter()
        def worker(i):
            return call_api([{"role": "user", "content": f"Count numbers 1 to 20 slowly: {i}"}], max_tokens=60, stream=True)
        
        with ThreadPoolExecutor(max_workers=c) as pool:
            client_results = list(pool.map(worker, range(c)))
        wall_time = time.perf_counter() - t_start
        total_tokens = sum(r["completion_tokens"] for r in client_results)
        agg_speed = total_tokens / wall_time if wall_time > 0 else 0.0
        log(f"  -> Concurrency {c}: {total_tokens} tokens in {wall_time:.2f}s (Agg Speed: {agg_speed:.2f} tok/s)")
        results_summary["tests"][f"concurrency_c{c}"] = {
            "concurrency": c,
            "total_tokens": total_tokens,
            "wall_time_s": round(wall_time, 2),
            "aggregate_tok_s": round(agg_speed, 2)
        }

# ============================================================================
# 4. 4-TASK COMPARE BENCHMARK
# ============================================================================
def test_compare():
    log("\n" + "="*80)
    log("[TEST 4/7] 4-Task Benchmark Suite (Compare Benchmark)")
    log("="*80)
    tasks = [
        ("AVL Tree", "Write a complete self-balancing AVL Tree in Python with insert, delete, search, in-order traversal, and a unit test.", 1200),
        ("Concurrency Debug", "Analyze a buggy multi-threaded BoundedBuffer class in Python with lost wakeups and race conditions. Provide the fix.", 1000),
        ("System Architecture", "Design a high-throughput distributed rate limiter with sliding window counter, Redis cluster, and fallback.", 1200),
        ("Constraint Adherence", "Implement a PaymentGatewayClient with exact requirements: Audit Tag format AUDIT_{uuid4}, retry formula 2^attempt * 100ms, and Error envelope.", 1000)
    ]
    task_res = []
    for name, prompt, max_t in tasks:
        log(f"  Running: {name}...")
        res = call_api([{"role": "user", "content": prompt}], max_tokens=max_t, stream=True)
        log(f"    -> Tokens: {res['completion_tokens']}, Speed: {res['decode_speed']:.2f} tok/s, TTFT: {res['ttft']*1000:.1f} ms")
        task_res.append({
            "task": name,
            "tokens": res["completion_tokens"],
            "speed_tok_s": round(res["decode_speed"], 2),
            "ttft_ms": round(res["ttft"]*1000, 1)
        })
    avg_speed = sum(t["speed_tok_s"] for t in task_res) / len(task_res)
    results_summary["tests"]["compare_tasks"] = {
        "status": "PASS",
        "avg_speed_tok_s": round(avg_speed, 2),
        "tasks": task_res
    }

# ============================================================================
# 5. HIGH-ENTROPY KV ASSOCIATIVE RECALL & MULTI-HOP TEST
# ============================================================================
def test_high_entropy():
    log("\n" + "="*80)
    log("[TEST 5/7] High-Entropy Associative Key-Value Recall & Multi-Hop Test")
    log("="*80)
    # Generate long context with high-entropy pairs
    filler = "In high-performance distributed architectures, nodes communicate via low-latency RPC protocols with strict SLAs. " * 300
    kv_data = (
        "CONFIDENTIAL LOOKUP TABLE:\n"
        "KEY_ALPHA_77: VAL_X9$mK2\n"
        "KEY_BETA_99: VAL_Z4#pQ8\n"
        "KEY_GAMMA_12: VAL_L1*vR5\n"
    )
    prompt = f"{filler}\n{kv_data}\n{filler}\nWhat is the exact value for KEY_BETA_99 and KEY_GAMMA_12? Answer strictly in format: KEY=VAL"
    res = call_api([{"role": "user", "content": prompt}], max_tokens=150, stream=True)
    content = res["content"]
    m1 = "Z4#pQ8" in content
    m2 = "L1*vR5" in content
    status = "PASS" if (m1 and m2) else ("PARTIAL" if (m1 or m2) else "FAIL")
    log(f"  -> High-Entropy Recall: Beta={m1}, Gamma={m2} -> {status}")
    log(f"  -> TTFT: {res['ttft']*1000:.1f} ms | Decode Speed: {res['decode_speed']:.2f} tok/s")
    results_summary["tests"]["high_entropy_recall"] = {
        "status": status,
        "matches": {"beta": m1, "gamma": m2},
        "ttft_ms": round(res["ttft"]*1000, 1),
        "tok_s": round(res["decode_speed"], 2)
    }

# ============================================================================
# 6. EXTREME PRECISION FINANCIAL LEDGER & ALU PIPELINE
# ============================================================================
def test_extreme_precision():
    log("\n" + "="*80)
    log("[TEST 6/7] Extreme Precision 10-Stage Arithmetic & Financial Reconciliation")
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
    # 10000 + 2450.50 - 15.25 - 1200.00 + 45.10 - 342.80 + 112.50 - 25.00 + 580.45 - 116.09 = 11489.41
    ledger_text = "\n".join(f"{s[0]}: {s[1]}" for s in stages)
    prompt = (
        f"Perform an exact financial audit ledger reconciliation on the following sequential journal entries:\n"
        f"{ledger_text}\n\n"
        f"Calculate the precise ending balance to two decimal places. State: 'Ending Balance: $XXXXX.XX'"
    )
    res = call_api([{"role": "user", "content": prompt}], max_tokens=400, stream=True)
    content = res["content"]
    expected = "11489.41"
    matched = expected in content
    status = "PASS" if matched else "FAIL"
    log(f"  -> Calculated ending balance match ({expected}): {matched} -> {status}")
    log(f"  -> TTFT: {res['ttft']*1000:.1f} ms | Decode Speed: {res['decode_speed']:.2f} tok/s")
    results_summary["tests"]["precision_ledger"] = {
        "status": status,
        "expected": expected,
        "matched": matched,
        "ttft_ms": round(res["ttft"]*1000, 1),
        "tok_s": round(res["decode_speed"], 2)
    }

# ============================================================================
# 7. EXECUTABLE CODE GENERATION & DYNAMIC UNIT TEST EXECUTION
# ============================================================================
def test_code_execution():
    log("\n" + "="*80)
    log("[TEST 7/7] Executable Algorithmic Code Generation & Dynamic Verification")
    log("="*80)
    prompt = (
        "Write a complete Python implementation of an LRU Cache using an OrderedDict or Doubly Linked List.\n"
        "Class name must be `LRUCache` with `__init__(self, capacity: int)`, `get(self, key: int) -> int`, and `put(self, key: int, value: int) -> None`.\n"
        "Return ONLY valid Python code inside a ```python ``` block."
    )
    res = call_api([{"role": "user", "content": prompt}], max_tokens=800, stream=True)
    code = res["content"]
    # Extract python code
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
    full_script = code + "\n" + test_harness
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_script)
        f_path = f.name

    try:
        sub_res = subprocess.run([sys.executable, f_path], capture_output=True, text=True, timeout=15)
        passed = "UNIT_TESTS_PASSED" in sub_res.stdout
        err = sub_res.stderr.strip()
    except Exception as e:
        passed = False
        err = str(e)
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)

    status = "PASS" if passed else "FAIL"
    log(f"  -> Dynamic Execution Result: {status} {'(All assertions passed)' if passed else f'Error: {err[:150]}'}")
    log(f"  -> Decode Speed: {res['decode_speed']:.2f} tok/s")
    results_summary["tests"]["code_execution"] = {
        "status": status,
        "dynamic_tests_passed": passed,
        "tok_s": round(res["decode_speed"], 2)
    }

def main():
    log("="*80)
    log(f"STARTING COMPREHENSIVE BENCHMARK RUNNER AGAINST {BASE_URL}")
    log(f"Model ID: {DEFAULT_MODEL}")
    log("="*80)

    try:
        test_streaming()
        test_vision()
        test_concurrency()
        test_compare()
        test_high_entropy()
        test_extreme_precision()
        test_code_execution()
    except Exception as e:
        log(f"\nCRITICAL RUNNER ERROR: {e}")
        results_summary["error"] = str(e)

    log("\n" + "="*80)
    log("BENCHMARK SUITE EXECUTION FINISHED")
    log(f"Saving final report to: {OUT_FILE}")
    with open(OUT_FILE, "w") as f:
        json.dump(results_summary, f, indent=2)
    log("="*80)

if __name__ == "__main__":
    main()
