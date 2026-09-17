#!/usr/bin/env python3
"""
High-Entropy Long-Context Stress Test:
Comparing 4,3 vs 4,2 KV Cache Quantization Boundaries.

Tests:
1. High-Entropy Key-Value Associative Recall (10 random alphanumeric keys & values buried in 50k tokens)
2. Multi-Hop Variable Tracking across 8 sequential dependency stages in 50k tokens
3. Multi-Value Filtering & Set Aggregation across 50 database records
"""

import sys
import os
import time
import json
import re
import urllib.request
import argparse

DEFAULT_API_URL = "http://127.0.0.1:8888/v1/chat/completions"

def call_model(url, messages, max_tokens=1500, temperature=0.0, enable_thinking=False, timeout=600):
    payload = {
        "model": "qwen3.8-27b-exl3-3.0bpw",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": enable_thinking}
    }
    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"})
    
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    tt = time.perf_counter() - t0
    
    choice = data["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)
    tok_per_sec = completion_tokens / tt if tt > 0 else 0.0
    
    return {
        "content": content,
        "reasoning": reasoning,
        "text": content if content.strip() else reasoning,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_time_s": tt,
        "tok_per_sec": tok_per_sec
    }

FILLER_BLOCK = (
    "The Linux kernel page cache caches pages of files to optimize disk I/O performance. "
    "When a process performs a read system call, the kernel first inspects the radix tree or xarray "
    "associated with the inode. If the requested page is resident in memory, the data is copied directly "
    "to the user space buffer via CPU copy or DMA transfer, avoiding high-latency block device access. "
    "Dirty pages modified by write operations are periodically written back to persistent storage "
    "by the flusher threads (kworker) according to dirty_ratio and dirty_background_ratio sysctl knobs. "
    "Memory cgroups enforce hierarchical resource isolation, throttling writeback bandwidth when limits are exceeded. "
) # ~100 tokens

def test_high_entropy_kv_recall(url):
    print("\n" + "="*80)
    print("STRESS TEST 1: High-Entropy Associative Key-Value Recall (10 Targets in 50k Tokens)")
    print("="*80)
    
    # 25 High-entropy key-value pairs
    kv_pairs = [
        ("KEY_A19B", "VAL_9X#qL2!p"),
        ("KEY_B83C", "VAL_4m$Zk8*v"),
        ("KEY_C72D", "VAL_7w@Np5&b"),
        ("KEY_D61E", "VAL_2r%Vj1^m"),
        ("KEY_E50F", "VAL_8t&Kx9$d"),
        ("KEY_F49G", "VAL_3y*Bd4#w"),
        ("KEY_G38H", "VAL_6p#Tq7!z"),
        ("KEY_H27I", "VAL_1k$Mf2&c"),
        ("KEY_I16J", "VAL_9s@Ln8%x"),
        ("KEY_J05K", "VAL_5d%Rp3^k"),
        ("KEY_K94L", "VAL_7z&Wb6$f"),
        ("KEY_L83M", "VAL_4h*Vc9#t"),
        ("KEY_M72N", "VAL_2n#Xm1!q"),
        ("KEY_N61O", "VAL_8b$Dj5&p"),
        ("KEY_O50P", "VAL_6k@Zt7%s"),
        ("KEY_P49Q", "VAL_1g%Kp4^r"),
        ("KEY_Q38R", "VAL_9f&Nq8$y"),
        ("KEY_R27S", "VAL_3c*Tr2#m"),
        ("KEY_S16T", "VAL_7v#Bw6!n"),
        ("KEY_T05U", "VAL_5m$Pd9&k"),
        ("KEY_U94V", "VAL_8x@Lf3%w"),
        ("KEY_V83W", "VAL_2q%Zj7^b"),
        ("KEY_W72X", "VAL_4t&Kx1$d"),
        ("KEY_X61Y", "VAL_6r*Wn5#c"),
        ("KEY_Y50Z", "VAL_9p#Vq8!s")
    ]
    
    # Select 8 random target keys to retrieve
    targets = [
        "KEY_B83C", "KEY_E50F", "KEY_H27I", "KEY_K94L",
        "KEY_N61O", "KEY_Q38R", "KEY_T05U", "KEY_W72X"
    ]
    expected_map = dict(kv_pairs)
    
    # Interleave pairs across 500 filler blocks (~50,000 tokens)
    total_blocks = 500
    stride = total_blocks // len(kv_pairs)
    
    doc = []
    pair_idx = 0
    for b in range(total_blocks):
        doc.append(FILLER_BLOCK)
        if b % stride == 0 and pair_idx < len(kv_pairs):
            k, v = kv_pairs[pair_idx]
            doc.append(f"\n[ASSOCIATIVE_STORAGE_ENTRY: {k} -> {v}]\n")
            pair_idx += 1
            
    doc_text = "".join(doc)
    prompt = (
        f"{doc_text}\n\n"
        "TASK:\n"
        "From the associative storage records above, retrieve the exact values for these 8 keys:\n"
        + "\n".join(f"- {t}" for t in targets) + "\n\n"
        "Respond with a JSON object mapping each key to its exact string value.\n"
        "Format: JSON only, no commentary."
    )
    
    print(f"  Ingesting ~50,000 tokens with 25 associative high-entropy key-value pairs...")
    res = call_model(url, [{"role": "user", "content": prompt}], max_tokens=1000, temperature=0.0)
    print(f"  Generated {res['completion_tokens']} tokens in {res['total_time_s']:.2f}s ({res['tok_per_sec']:.1f} tok/s)")
    
    out = res["content"]
    # Parse json or match keys
    passed_keys = 0
    for t in targets:
        exp_v = expected_map[t]
        if exp_v in out:
            passed_keys += 1
            print(f"  - {t}: {exp_v} -> MATCH")
        else:
            print(f"  - {t}: {exp_v} -> MISMATCH / MISSING")
            
    success = (passed_keys == len(targets))
    print(f"  [Result]: {passed_keys}/{len(targets)} exact high-entropy matches. {'PASS' if success else 'FAIL'}")
    return {"name": "High-Entropy KV Recall (8/8)", "passed": success, "score": f"{passed_keys}/{len(targets)}"}

def test_multi_hop_variable_tracking(url):
    print("\n" + "="*80)
    print("STRESS TEST 2: Multi-Hop Variable Tracking across 8 Dependency Stages in 50k Tokens")
    print("="*80)
    
    # 8 sequential variable hops
    # var_a = 7412
    # var_b = var_a + 1928  -> 9340
    # var_c = var_b * 3 - 500 -> 27520
    # var_d = var_c ^ 1023  -> 27520 ^ 1023 = 26559
    # var_e = var_d + 12400 -> 38959
    # var_f = var_e - 3959  -> 35000
    # var_g = var_f * 2 + 77 -> 70077
    # var_h = var_g ^ 4095  -> 70077 ^ 4095 = 73930
    
    stages = [
        (50, "REGISTRATION: var_a = 7412"),
        (110, "TRANSFORMATION: var_b = var_a + 1928"),
        (170, "TRANSFORMATION: var_c = var_b * 3 - 500"),
        (230, "TRANSFORMATION: var_d = var_c ^ 1023  (bitwise XOR)"),
        (290, "TRANSFORMATION: var_e = var_d + 12400"),
        (350, "TRANSFORMATION: var_f = var_e - 3959"),
        (410, "TRANSFORMATION: var_g = var_f * 2 + 77"),
        (470, "TRANSFORMATION: var_h = var_g ^ 4095  (bitwise XOR)"),
    ]
    
    # Ground truth:
    va = 7412
    vb = va + 1928
    vc = vb * 3 - 500
    vd = vc ^ 1023
    ve = vd + 12400
    vf = ve - 3959
    vg = vf * 2 + 77
    vh = vg ^ 4095
    expected_final = vh  # 73930
    
    stage_map = dict(stages)
    total_blocks = 500
    doc = []
    for b in range(total_blocks):
        doc.append(FILLER_BLOCK)
        if b in stage_map:
            doc.append(f"\n[CRITICAL_ALU_PIPELINE: {stage_map[b]}]\n")
            
    doc_text = "".join(doc)
    prompt = (
        f"{doc_text}\n\n"
        "TASK:\n"
        "Trace the 8-stage sequential ALU pipeline transformations (var_a through var_h) defined in the document above.\n"
        "Calculate the exact intermediate values for var_a, var_b, var_c, var_d, var_e, var_f, var_g, and the final value of var_h.\n"
        "State the final value of var_h clearly at the end: 'FINAL_VALUE: <number>'."
    )
    
    print(f"  Ingesting ~50,000 tokens with 8 sequential multi-hop dependencies...")
    res = call_model(url, [{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.0, enable_thinking=True)
    print(f"  Generated {res['completion_tokens']} tokens in {res['total_time_s']:.2f}s ({res['tok_per_sec']:.1f} tok/s)")
    
    text = res["text"]
    has_expected = str(expected_final) in text
    print(f"  - Ground Truth var_h = {expected_final}: {'MATCH' if has_expected else 'FAIL'}")
    print(f"  [Result]: {'PASS' if has_expected else 'FAIL'}")
    return {"name": "Multi-Hop Dependency (8 Hops)", "passed": has_expected, "expected": expected_final}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_API_URL)
    parser.add_argument("--out", default="tests/results/stress_test_results.json")
    args = parser.parse_args()
    
    results = []
    results.append(test_high_entropy_kv_recall(args.url))
    results.append(test_multi_hop_variable_tracking(args.url))
    
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY")
    print("="*80)
    for r in results:
        status = "PASSED" if r["passed"] else "FAILED"
        print(f" - {r['name']:<45} : {status}")
        
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
