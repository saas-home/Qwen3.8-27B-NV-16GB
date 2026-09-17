#!/usr/bin/env python3
"""
Multi-client concurrency and continuous batching stress test script.
Tests:
- Multiple concurrent streaming requests executing against the API server.
- Continuous batching across parallel GPU slots.
- FIFO request queueing when concurrency exceeds available slots.
- Latency (TTFT), duration, throughput (tok/s), and server health telemetry.
"""

import json
import time
import urllib.request
from urllib.parse import urljoin
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor

DEFAULT_API_URL = "http://127.0.0.1:8888/v1/chat/completions"
DEFAULT_HEALTH_URL = "http://127.0.0.1:8888/health"

SAMPLE_PROMPTS = [
    "Write a detailed Python function implementing merge sort with clear explanations and docstrings.",
    "Write a detailed Python function implementing quick sort with clear explanations and docstrings.",
    "Write a detailed Python class implementing a Binary Search Tree with insert and search methods.",
    "Write a detailed Python class implementing a Min-Heap priority queue with push and pop methods.",
    "Write a detailed Python function implementing Dijkstra's shortest path algorithm with explanations.",
    "Write a detailed Python function implementing breadth-first search on a directed graph with explanations.",
    "Write a detailed Python function implementing depth-first search with topological sort for a DAG.",
    "Write a detailed Python class implementing an LRU cache using a doubly linked list and hash map."
]

def check_health(health_url):
    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def client_task(client_id, prompt, api_url, model, max_tokens, temperature, results, lock):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(api_url, data=data, headers={"Content-Type": "application/json"})
    
    t_start = time.perf_counter()
    t_first = None
    t_last = None
    token_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    error_msg = None
    
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            for line in resp:
                line = line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                line_data = line[5:].strip()
                if line_data == "[DONE]":
                    break
                try:
                    chunk = json.loads(line_data)
                except Exception:
                    continue
                
                if "error" in chunk:
                    error_msg = str(chunk["error"])
                    break
                    
                if "usage" in chunk and chunk["usage"]:
                    prompt_tokens = chunk["usage"].get("prompt_tokens", 0)
                    completion_tokens = chunk["usage"].get("completion_tokens", 0)
                
                choices = chunk.get("choices") or []
                if choices:
                    delta = choices[0].get("delta", {})
                    content = delta.get("content") or delta.get("reasoning_content")
                    if content:
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        token_count += 1
    except Exception as e:
        error_msg = str(e)

    t_end = time.perf_counter()
    ttft = (t_first - t_start) if t_first else 0.0
    duration = (t_end - t_start)
    if completion_tokens == 0:
        completion_tokens = token_count
        
    decode_time = (t_last - t_first) if (t_first and t_last and t_last > t_first) else 0.0
    decode_speed = (completion_tokens - 1) / decode_time if (decode_time > 0 and completion_tokens > 1) else 0.0
    
    with lock:
        results[client_id] = {
            "prompt": prompt[:40] + ("..." if len(prompt) > 40 else ""),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_s": ttft,
            "total_time_s": duration,
            "tok_per_sec": decode_speed,
            "start_offset_s": t_start,
            "end_offset_s": t_end,
            "error": error_msg
        }

def main():
    parser = argparse.ArgumentParser(description="Multi-client concurrency and continuous batching stress test.")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="API completions endpoint URL")
    parser.add_argument("--health-url", default=None, help="Server health endpoint URL")
    parser.add_argument("--model", default="qwen3.8-27b-exl3-3.0bpw", help="Model ID")
    parser.add_argument("--concurrency", "-c", type=int, default=4, help="Number of concurrent clients (default: 4)")
    parser.add_argument("--tokens", "-t", type=int, default=128, help="Max generation tokens per client (default: 128)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    args = parser.parse_args()

    health_url = args.health_url
    if not health_url:
        if "/v1/" in args.url:
            health_url = args.url.split("/v1/")[0] + "/health"
        else:
            health_url = DEFAULT_HEALTH_URL

    print("=" * 80)
    print("Multi-Client Concurrency & Continuous Batching Stress Test")
    print(f"Target URL   : {args.url}")
    print(f"Health URL   : {health_url}")
    print(f"Model ID     : {args.model}")
    print(f"Concurrency  : {args.concurrency} concurrent client requests")
    print(f"Max Tokens   : {args.tokens} tokens/client")
    print(f"Temperature  : {args.temperature}")
    print("=" * 80)

    h_before = check_health(health_url)
    print(f"Health check before test: {h_before}\n")

    results = {}
    lock = threading.Lock()
    
    # Assign prompts to clients
    prompts = [SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)] for i in range(args.concurrency)]

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                client_task,
                i + 1,
                prompts[i],
                args.url,
                args.model,
                args.tokens,
                args.temperature,
                results,
                lock
            )
            for i in range(args.concurrency)
        ]
        
        # Probe health shortly after launching all requests
        time.sleep(0.5)
        h_mid = check_health(health_url)
        print(f"Health check during active generation: {h_mid}\n")
        
        for f in futures:
            f.result()

    t_total = time.perf_counter() - t0
    h_after = check_health(health_url)
    print(f"Health check after test: {h_after}\n")

    total_tokens = sum(r["completion_tokens"] for r in results.values() if not r.get("error"))
    aggregate_tok_s = total_tokens / t_total if t_total > 0 else 0.0

    header = f"{'Client':<8} | {'Tokens':<8} | {'TTFT (s)':<10} | {'Total Time (s)':<16} | {'Speed (tok/s)':<14} | {'Timeline (s)'}"
    print(header)
    print("-" * len(header))
    
    has_errors = False
    for cid in sorted(results.keys()):
        r = results[cid]
        if r.get("error"):
            has_errors = True
            print(f"Client {cid:<2} | {'FAILED':<8} | Error: {r['error']}")
            continue
        rel_start = r["start_offset_s"] - t0
        rel_end = r["end_offset_s"] - t0
        timeline = f"{rel_start:.2f}s -> {rel_end:.2f}s"
        print(f"Client {cid:<2} | {r['completion_tokens']:<8} | {r['ttft_s']:<10.3f} | {r['total_time_s']:<16.2f} | {r['tok_per_sec']:<14.2f} | {timeline}")

    print("-" * len(header))
    print(f"Total Wall-Clock Time : {t_total:.2f} s")
    print(f"Total Tokens Emitted  : {total_tokens} tokens")
    print(f"Aggregate Throughput  : {aggregate_tok_s:.2f} tokens/s")
    print("=" * 80)

    if has_errors:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
