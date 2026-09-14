#!/usr/bin/env python3
"""
Simple streaming and latency verification script.
Tests:
- Streaming chunk reception
- First token latency (TTFT)
- Generation throughput
- Server health check
"""

import json
import time
import urllib.request
import argparse

DEFAULT_API_URL = "http://127.0.0.1:8888/v1/chat/completions"

def run_test(api_url, prompt_text, max_tokens=64, model="qwen3.8-27b-exl3-3.0bpw"):
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
    req = urllib.request.Request(api_url, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.perf_counter()
    t_first = None
    t_last = None
    chunks_count = 0
    prompt_tokens = 0
    completion_tokens = 0
    text_accum = []
    
    with urllib.request.urlopen(req, timeout=300) as response:
        for line in response:
            line = line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            line_data = line[5:].strip()
            if line_data == "[DONE]":
                break
            try:
                chunk = json.loads(line_data)
            except Exception:
                continue
            
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
                    chunks_count += 1
                    text_accum.append(content)
                    print(content, end="", flush=True)

    print()
    if t_last is None:
        t_last = time.perf_counter()
    if t_first is None:
        t_first = t_last

    ttft = t_first - t0
    decode_time = t_last - t_first
    
    prefill_speed = prompt_tokens / ttft if ttft > 0 else 0
    decode_speed = (completion_tokens - 1) / decode_time if decode_time > 0 and completion_tokens > 1 else 0
    
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "ttft_s": ttft,
        "prefill_tok_s": prefill_speed,
        "decode_time_s": decode_time,
        "decode_tok_s": decode_speed,
        "text": "".join(text_accum)
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test streaming response and measure token latency.")
    parser.add_argument("--url", default=DEFAULT_API_URL, help="API URL")
    parser.add_argument("--prompt", default="Why is the sky blue? Answer in 2 sentences.", help="Prompt text")
    parser.add_argument("--tokens", type=int, default=128, help="Max tokens to generate")
    parser.add_argument("--model", default="qwen3.8-27b-exl3-3.0bpw", help="Model ID")
    args = parser.parse_args()

    print(f"Connecting to {args.url} (Model: {args.model})...")
    res = run_test(args.url, args.prompt, max_tokens=args.tokens, model=args.model)
    print("\n--- Summary ---")
    print(f"Prompt Tokens: {res['prompt_tokens']}")
    print(f"Gen Tokens   : {res['completion_tokens']}")
    print(f"TTFT         : {res['ttft_s']*1000:.1f} ms")
    print(f"Prefill Speed: {res['prefill_tok_s']:.1f} tok/s")
    print(f"Decode Speed : {res['decode_tok_s']:.1f} tok/s")
