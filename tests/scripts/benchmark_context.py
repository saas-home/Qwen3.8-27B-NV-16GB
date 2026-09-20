#!/usr/bin/env python3
"""
Stress test and throughput benchmark across long context windows (up to 204.8k tokens).
Measures:
- Prompt token count
- TTFT (Time To First Token) & Prefill speed (tokens/sec)
- Completion token count & Decode speed (tokens/sec)
"""

import json
import time
import urllib.request
import urllib.error
import sys
import os
import uuid
import argparse

DEFAULT_API_URL = "http://127.0.0.1:8888/v1/chat/completions"

BASE_TEXT = (
    "The quick brown fox jumps over the lazy dog. In computer science and artificial intelligence, "
    "large language models utilize transformer architectures with multi-head self-attention mechanisms "
    "to process sequential data efficiently. Memory bandwidth and compute capacity determine inference speed. "
)

def build_prompt(target_tokens, salt=True):
    # Base text is ~45 tokens and ~230 chars (~5 chars per token)
    repeat_count = max(1, int((target_tokens - (25 if salt else 0)) / 45))
    prefix = f"[Benchmark Target: {target_tokens} | Epoch: {time.time():.4f} | UUID: {uuid.uuid4()}]\n" if salt else ""
    text = (BASE_TEXT * repeat_count)
    return prefix + text + "\n\nSummarize the key aspects mentioned above in detail."

def run_benchmark_run(api_url, prompt_text, max_tokens=64, model="qwen3.8-27b-exl3-3.0bpw", target_tokens=0, api_key=""):
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt_text}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True}
    }
    
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(api_url, data=data, headers=headers)
    
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    tokens_emitted = 0
    prompt_tokens = 0
    completion_tokens = 0
    cached_tokens = 0
    
    try:
        with urllib.request.urlopen(req, timeout=1200) as response:
            for line in response:
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
                    print(f"\n[ERROR from server]: {chunk['error']}", flush=True)
                    return {"error": str(chunk["error"])}
                
                if "usage" in chunk and chunk["usage"]:
                    prompt_tokens = chunk["usage"].get("prompt_tokens", prompt_tokens)
                    completion_tokens = chunk["usage"].get("completion_tokens", completion_tokens)
                    cached_tokens = chunk["usage"].get("prompt_tokens_details", {}).get("cached_tokens", cached_tokens)
                
                choices = chunk.get("choices") or []
                if not choices:
                    continue

                if choices[0].get("finish_reason") == "error":
                    err_msg = choices[0].get("message", {}).get("content", "Server error during generation")
                    print(f"\n[ERROR from model]: {err_msg}", flush=True)
                    return {"error": err_msg}

                delta = choices[0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    tokens_emitted += 1
    except Exception as e:
        print(f"\n[REQUEST EXCEPTION]: {e}", flush=True)
        return {
            "error": str(e),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "ttft_s": time.perf_counter() - t0,
            "cold_prefill_tok_s": 0.0,
            "effective_prefill_tok_s": 0.0,
            "decode_time_s": 0.0,
            "decode_tok_s": 0.0
        }

    if prompt_tokens == 0 and target_tokens > 0:
        prompt_tokens = target_tokens

    if completion_tokens == 0:
        completion_tokens = tokens_emitted

    if t_last is None:
        t_last = time.perf_counter()
    if t_first is None:
        t_first = t_last

    ttft = t_first - t0
    decode_time = t_last - t_first
    
    uncached_tokens = max(0, prompt_tokens - cached_tokens)
    cold_prefill_speed = (uncached_tokens / ttft) if ttft > 0 and uncached_tokens > 0 else (prompt_tokens / ttft if ttft > 0 else 0.0)
    effective_prefill_speed = prompt_tokens / ttft if ttft > 0 else 0.0
    decode_speed = (completion_tokens - 1) / decode_time if decode_time > 0 and completion_tokens > 1 else 0.0
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "cached_tokens": cached_tokens,
        "ttft_s": ttft,
        "cold_prefill_tok_s": cold_prefill_speed,
        "effective_prefill_tok_s": effective_prefill_speed,
        "decode_time_s": decode_time,
        "decode_tok_s": decode_speed
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark long context prefill and decode speeds.")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="Endpoint URL")
    parser.add_argument("--model", default="qwen3.8-27b-exl3-3.0bpw", help="Model ID")
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""), help="API key")
    parser.add_argument("--tokens", type=int, nargs="+", default=[200000], help="Context token target(s)")
    parser.add_argument("--gen-tokens", type=int, default=64, help="Max generation tokens")
    parser.add_argument("--no-salt", action="store_true", help="Disable unique salt prepending (allows warm prefix caching)")
    args = parser.parse_args()

    print("=" * 96)
    print("Prefill & Decode Speed Benchmark Across Context Levels")
    print(f"Target URL: {args.url}")
    print(f"Model ID  : {args.model}")
    print(f"Prefix Isolation: {'Disabled (warm caching allowed)' if args.no_salt else 'Enabled (cold prefill isolated)'}")
    print("=" * 96)
    
    # Warmup
    print("Warming up GPU...", end="", flush=True)
    w = run_benchmark_run(args.url, "Warmup: reply with 5 words.", max_tokens=10, model=args.model, target_tokens=10, api_key=args.api_key)
    print(f" Ready. (Warmup prompt: {w.get('prompt_tokens', 0)} toks, decode: {w.get('decode_tok_s', 0):.1f} t/s)\n")
    
    header = f"{'Target Ctx':>12} | {'Actual Prompt':>14} | {'Cached':>8} | {'TTFT (s)':>10} | {'Cold (t/s)':>12} | {'Effective':>12} | {'Decode (t/s)':>12} | {'Gen Toks':>9}"
    print(header)
    print("-" * len(header))
    
    for target in args.tokens:
        print(f">> Testing ~{target//1024}k context...", end="", flush=True)
        prompt = build_prompt(target, salt=not args.no_salt)
        res = run_benchmark_run(args.url, prompt, max_tokens=args.gen_tokens, model=args.model, target_tokens=target, api_key=args.api_key)
        print("\r" + " " * 40 + "\r", end="")
        
        if "error" in res and res["error"]:
            print(f"{target:>12,d} | {'FAILED':>14} | {'-':>8} | {'-':>10} | {str(res['error'])[:24]:>12}")
            continue
            
        row = (
            f"{target:>12,d} | "
            f"{res['prompt_tokens']:>14,d} | "
            f"{res['cached_tokens']:>8,d} | "
            f"{res['ttft_s']:>10.3f} | "
            f"{res['cold_prefill_tok_s']:>12.1f} | "
            f"{res['effective_prefill_tok_s']:>12.1f} | "
            f"{res['decode_tok_s']:>12.2f} | "
            f"{res['completion_tokens']:>9d}"
        )
        print(row, flush=True)
        time.sleep(2)
        
    print("-" * len(header))
    print("Benchmark complete.")

if __name__ == "__main__":
    main()
