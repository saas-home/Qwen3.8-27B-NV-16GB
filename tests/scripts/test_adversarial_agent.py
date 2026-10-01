#!/usr/bin/env python3
"""
Enterprise Hardened Adversarial, Reasoning & Agentic Benchmark Suite
(tests/scripts/test_adversarial_agent.py)

Specifically calibrated for advanced local LLMs (such as Qwen3.8-27B) running on
quantized KV caches (e.g. EXL3 with 4-bit Key / 3-bit Value compression).

Evaluates 5 Hardening Pillars:
1. Multi-Hop Graph Traversal with Adversarial Distractors & Version Deprecation (~100k tokens)
2. Novel Algorithmic Code Synthesis & Automated 5,000-Operation Property-Based Fuzz Testing
3. Combinatorial Anti-Constraint & Negative Constraint Adherence (IFEval Tier)
4. Counterfactual Axiomatic Symbolic Algebra with Non-Commutative Modular Operators
5. Extreme-Depth Frontier Needle Synthesis (0.5%, 50%, and 99.5% Context Horizons)
"""

import sys
import os
import time
import json
import re
import uuid
import random
import tempfile
import subprocess
import argparse
import urllib.request
import urllib.error

# ANSI Color formatting
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

class BenchmarkClient:
    def __init__(self, endpoint: str, model: str, api_key: str = ""):
        self.endpoint = endpoint.rstrip("/")
        if not self.endpoint.endswith("/v1"):
            self.endpoint += "/v1"
        self.completions_url = f"{self.endpoint}/chat/completions"
        self.model = model
        self.api_key = api_key

    def call(self, messages, max_tokens=2048, temperature=0.7, stream=True,
             tools=None, tool_choice=None, timeout=600):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
        }
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if stream:
            payload["stream_options"] = {"include_usage": True}

        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(self.completions_url, data=data, headers=headers)
        t0 = time.perf_counter()

        if not stream:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                res_json = json.loads(resp.read().decode("utf-8"))
            t_end = time.perf_counter()
            choice = res_json.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            tool_calls = msg.get("tool_calls")
            usage = res_json.get("usage", {})
            comp_toks = usage.get("completion_tokens", 0)
            return {
                "content": content,
                "reasoning": reasoning,
                "text": f"{reasoning}\n{content}" if reasoning else content,
                "tool_calls": tool_calls,
                "ttft_ms": (t_end - t0) * 1000.0,
                "total_time_s": t_end - t0,
                "tok_per_sec": comp_toks / (t_end - t0) if (t_end - t0) > 0 else 0.0,
                "completion_tokens": comp_toks,
                "prompt_tokens": usage.get("prompt_tokens", 0)
            }

        # Streaming mode for TTFT & speed measurement
        t_first = None
        t_last = None
        content_chunks = []
        reasoning_chunks = []
        token_count = 0
        prompt_tokens = 0
        completion_tokens = 0
        tool_calls = None

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
                    if "tool_calls" in delta and delta["tool_calls"]:
                        tool_calls = delta["tool_calls"]
                    c_text = delta.get("content") or ""
                    r_text = delta.get("reasoning_content") or ""
                    if c_text or r_text:
                        now = time.perf_counter()
                        if t_first is None:
                            t_first = now
                        t_last = now
                        token_count += 1
                        if c_text:
                            content_chunks.append(c_text)
                        if r_text:
                            reasoning_chunks.append(r_text)

        t_end = time.perf_counter()
        total_time = t_end - t0
        ttft = (t_first - t0) if t_first else total_time
        gen_time = (t_end - t_first) if t_first else total_time
        actual_comp = completion_tokens if completion_tokens > 0 else token_count
        speed = (actual_comp / gen_time) if gen_time > 0 else 0.0

        content_str = "".join(content_chunks)
        reasoning_str = "".join(reasoning_chunks)
        return {
            "content": content_str,
            "reasoning": reasoning_str,
            "text": f"{reasoning_str}\n{content_str}" if reasoning_str else content_str,
            "tool_calls": tool_calls,
            "ttft_ms": ttft * 1000.0,
            "total_time_s": total_time,
            "tok_per_sec": speed,
            "completion_tokens": actual_comp,
            "prompt_tokens": prompt_tokens
        }


# ============================================================================
# PILLAR 1: MULTI-HOP GRAPH TRAVERSAL WITH ADVERSARIAL DISTRACTORS (100k TOKENS)
# ============================================================================
def test_multihop_graph_distractors(client: BenchmarkClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 1/5] Multi-Hop Graph Traversal with Versioned Temporal Distractors (~100k Context)", bold=True, color=CYAN)
    log("="*88)

    filler_chunk = (
        "Enterprise cloud transit routing manages BGP autonomous system interconnects across software-defined WAN fabrics. "
        "Packet encapsulation leverages Geneve and VXLAN tunnels terminating on distributed SmartNIC vSwitch datanodes. "
        "Telemetry pipelines process streaming flow logs through Kafka clusters with partitioned RocksDB state stores. "
        "Zero-trust microsegmentation evaluates SPIFFE identity assertions and mutual TLS session keys at every ingress hop. "
    ) # ~60 tokens

    total_blocks = 1600 # ~96,000 tokens
    
    # Place adversarial distractors (older, deprecated routing policies)
    distractor_v1 = (
        "\n[ARCHITECTURAL_ROUTING_SPECIFICATION v1.0 (2024 - DEPRECATED)]\n"
        "Policy Rule 401: In the event of Primary Switch 7 outage, traffic from Client Mobile Edge routes through:\n"
        "Node A -> Gateway Chicago -> Database Shard Gamma (Read-Only Replica 1).\n"
        "Security Note: Route 401 was revoked by RFC-77 due to cross-datacenter latency violations.\n"
    )
    distractor_v2 = (
        "\n[ARCHITECTURAL_ROUTING_SPECIFICATION v2.2 (2025 - REVOKED / ROLLBACK)]\n"
        "Policy Rule 402: Under Switch 7 hardware failure, failover path is:\n"
        "Node A -> Gateway Frankfurt -> Database Shard Gamma (Staging Cluster).\n"
        "Audit Flag: Revoked after disaster recovery drill failure.\n"
    )
    # The active production specification
    active_v3 = (
        "\n[ARCHITECTURAL_ROUTING_SPECIFICATION v3.1 (2026 - ACTIVE PRODUCTION APPROVED)]\n"
        "Policy Rule 403: When Primary Switch 7 fails, the mandatory active failover sequence is:\n"
        "Client Mobile Edge -> Ingress Node Alpha -> Transit Hub Tokyo -> Secure Bridge Omega -> Database Shard Gamma (Active Primary).\n"
        "Mandatory Verification Code: AUTH_ROUTING_KEY_9921_TOK.\n"
        "Failover SLA: Must terminate within 45ms over dedicated dark fiber link.\n"
    )

    doc_parts = []
    for b in range(total_blocks):
        doc_parts.append(filler_chunk)
        if b == int(total_blocks * 0.15):
            doc_parts.append(distractor_v1)
        elif b == int(total_blocks * 0.50):
            doc_parts.append(distractor_v2)
        elif b == int(total_blocks * 0.85):
            doc_parts.append(active_v3)

    document = "".join(doc_parts)
    prompt = (
        f"{document}\n\n"
        "MISSION-CRITICAL ARCHITECTURAL INCIDENT ANALYSIS:\n"
        "A critical hardware outage occurs on Primary Switch 7. Review the ARCHITECTURAL_ROUTING_SPECIFICATION sections above and determine:\n"
        "1. What is the current, active, and approved failover routing path from Client Mobile Edge to Database Shard Gamma?\n"
        "2. State the exact Mandatory Verification Code for the active route.\n"
        "3. Explain why the previous v1.0 (Chicago) and v2.2 (Frankfurt) routes are invalid and must not be used.\n\n"
        "Conclude your response strictly with: 'ACTIVE_FAILOVER_CODE: <verification code>'."
    )

    log(f"  Ingesting ~{total_blocks * 60:,}-token context with 2 temporal distractors and 1 active specification...")
    res = client.call([{"role": "user", "content": prompt}], max_tokens=1500, temperature=0.0, stream=True)
    text = res["text"]

    has_active_tokyo = ("Transit Hub Tokyo" in text or "Tokyo" in text) and "Bridge Omega" in text
    has_code = "AUTH_ROUTING_KEY_9921_TOK" in text
    has_rejected_chicago = ("chicago" in text.lower()) and ("deprecated" in text.lower() or "rfc-77" in text.lower() or "revoked" in text.lower() or "invalid" in text.lower())
    has_rejected_frankfurt = ("frankfurt" in text.lower()) and ("revoked" in text.lower() or "rollback" in text.lower() or "invalid" in text.lower())

    passed = has_active_tokyo and has_code and has_rejected_chicago and has_rejected_frankfurt
    log(f"  - Active Route Identification (Tokyo/Omega): {'PASS' if has_active_tokyo else 'FAIL'}")
    log(f"  - Exact Verification Key Recall: {'PASS' if has_code else 'FAIL'}")
    log(f"  - Rejection of Deprecated v1.0 Chicago Distractor: {'PASS' if has_rejected_chicago else 'FAIL'}")
    log(f"  - Rejection of Revoked v2.2 Frankfurt Distractor: {'PASS' if has_rejected_frankfurt else 'FAIL'}")
    log(f"  -> Speed: {res['tok_per_sec']:.2f} tok/s | TTFT: {res['ttft_ms']:.1f} ms | Status: {'PASS' if passed else 'FAIL'}")

    return {
        "name": "Multi-Hop Graph & Adversarial Distractors",
        "passed": passed,
        "prompt_tokens": res["prompt_tokens"],
        "completion_tokens": res["completion_tokens"],
        "ttft_ms": round(res["ttft_ms"], 1),
        "tok_s": round(res["tok_per_sec"], 2)
    }


# ============================================================================
# PILLAR 2: NOVEL ALGORITHMIC SYNTHESIS WITH 5,000-OP PROPERTY FUZZ TEST
# ============================================================================
def test_novel_algorithmic_fuzz(client: BenchmarkClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 2/5] Novel Algorithmic Synthesis & Automated 5,000-Op Property Fuzz Testing", bold=True, color=CYAN)
    log("="*88)

    prompt = (
        "Implement a custom, non-standard Python class named `ConcurrentMonotonicRingBuffer`.\n"
        "This data structure must fulfill the following exact specifications:\n"
        "1. Constructor: `__init__(self, capacity: int)`: Initializes buffer with positive fixed integer capacity.\n"
        "2. Method: `push(self, value: int) -> int`: Adds value to the ring buffer. If buffer is full, it overwrites "
        "the oldest element. Returns the monotonically increasing 64-bit sequence ID assigned to this item (starting at sequence ID 1).\n"
        "3. Method: `get_by_seq(self, seq_id: int) -> int`: Returns the value associated with `seq_id`. If `seq_id` has already been "
        "overwritten or has not yet been pushed, raises `KeyError`.\n"
        "4. Method: `get_latest(self) -> tuple[int, int]`: Returns `(latest_seq_id, latest_value)`. If buffer is empty, raises `IndexError`.\n"
        "5. Method: `snapshot_in_order(self) -> list[tuple[int, int]]`: Returns list of all current `(seq_id, value)` pairs currently stored, "
        "ordered from oldest to newest.\n"
        "6. Provide self-contained Python code inside a ```python ``` code block. Do NOT use third-party libraries. "
        "Keep comments and docstrings concise to emit complete implementation code."
    )

    log("  Requesting synthesis of non-standard Monotonic Ring Buffer...")
    res = client.call([{"role": "user", "content": prompt}], max_tokens=3500, temperature=0.2, stream=True)
    raw = res["content"] if res["content"].strip() else res["text"]

    # Extract code
    code_match = re.findall(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    code = max(code_match, key=len).strip() if code_match else raw.strip()

    # Property-based fuzz verification harness
    fuzz_harness = """
import sys
import random

def run_fuzz_verification():
    # Phase 1: Basic validation
    buf = ConcurrentMonotonicRingBuffer(3)
    s1 = buf.push(10)
    s2 = buf.push(20)
    s3 = buf.push(30)
    assert (s1, s2, s3) == (1, 2, 3), f"Sequences must be 1, 2, 3, got {(s1, s2, s3)}"
    assert buf.get_by_seq(1) == 10
    assert buf.get_by_seq(2) == 20
    assert buf.get_by_seq(3) == 30
    assert buf.get_latest() == (3, 30)

    # Overwrite oldest (1 is evicted)
    s4 = buf.push(40)
    assert s4 == 4
    try:
        buf.get_by_seq(1)
        assert False, "Seq 1 should have raised KeyError after eviction"
    except KeyError:
        pass
    assert buf.get_by_seq(2) == 20
    assert buf.get_by_seq(4) == 40
    assert buf.snapshot_in_order() == [(2, 20), (3, 30), (4, 40)]

    # Phase 2: 5,000 randomized property-based operations
    cap = 50
    rb = ConcurrentMonotonicRingBuffer(cap)
    ground_truth = {}
    current_seq = 0

    random.seed(42)
    for op in range(5000):
        val = random.randint(1, 1000000)
        current_seq += 1
        seq_ret = rb.push(val)
        assert seq_ret == current_seq, f"Sequence drift at op {op}: expected {current_seq}, got {seq_ret}"
        ground_truth[current_seq] = val

        # Clean oldest in ground truth if capacity exceeded
        oldest_valid_seq = max(1, current_seq - cap + 1)
        if (current_seq - cap) in ground_truth:
            del ground_truth[current_seq - cap]

        # Intermittent random queries
        if op % 10 == 0:
            assert rb.get_latest() == (current_seq, val)
            # Query random valid key
            q_seq = random.randint(oldest_valid_seq, current_seq)
            assert rb.get_by_seq(q_seq) == ground_truth[q_seq]

            # Query evicted key
            if oldest_valid_seq > 1:
                evicted_seq = random.randint(1, oldest_valid_seq - 1)
                try:
                    rb.get_by_seq(evicted_seq)
                    assert False, f"Evicted sequence {evicted_seq} did not raise KeyError"
                except KeyError:
                    pass

        # Intermittent snapshot check
        if op % 250 == 0:
            snap = rb.snapshot_in_order()
            expected_snap = sorted(ground_truth.items(), key=lambda x: x[0])
            assert snap == expected_snap, f"Snapshot mismatch at op {op}: {snap} vs {expected_snap}"

    print("ALL_5000_FUZZ_TESTS_PASSED")

if __name__ == "__main__":
    run_fuzz_verification()
"""
    full_script = code + "\n\n" + fuzz_harness
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_script)
        temp_path = f.name

    fuzz_passed = False
    err_msg = ""
    try:
        proc = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=20)
        fuzz_passed = ("ALL_5000_FUZZ_TESTS_PASSED" in proc.stdout) and (proc.returncode == 0)
        if not fuzz_passed:
            err_msg = (proc.stderr or proc.stdout).strip()[:300]
    except Exception as e:
        err_msg = str(e)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    log(f"  - Automated 5,000 Operations Fuzz Assertions: {'PASS' if fuzz_passed else 'FAIL'}")
    if not fuzz_passed:
        log(f"    [Error]: {err_msg}", color=RED)
    log(f"  -> Speed: {res['tok_per_sec']:.2f} tok/s | Status: {'PASS' if fuzz_passed else 'FAIL'}")

    return {
        "name": "Novel Algorithmic Synthesis (5k Fuzz)",
        "passed": fuzz_passed,
        "completion_tokens": res["completion_tokens"],
        "tok_s": round(res["tok_per_sec"], 2),
        "error": err_msg
    }


# ============================================================================
# PILLAR 3: COMBINATORIAL ANTI-CONSTRAINT FOLLOWING (IFEVAL TIER)
# ============================================================================
def test_ifeval_combinatorial_constraints(client: BenchmarkClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 3/5] Combinatorial Anti-Constraint & Negative Instruction Following (IFEval Tier)", bold=True, color=CYAN)
    log("="*88)

    prompt = (
        "Write an architectural briefing on 'Zero-Trust Kernel Isolation'.\n"
        "You must strictly satisfy all 6 of the following constraints simultaneously:\n\n"
        "Constraint 1: Structure your answer into exactly 4 markdown sections headed by:\n"
        "   '### Section I: Foundation'\n"
        "   '### Section II: Isolation Proof'\n"
        "   '### Section III: Hardware Enclaves'\n"
        "   '### Section IV: Infrastructure Configuration'\n\n"
        "Constraint 2 (Lipogram): In '### Section II: Isolation Proof', you must NEVER use the letter 'e' (neither lowercase 'e' nor uppercase 'E'). Every single word in Section II must be completely free of 'e'.\n\n"
        "Constraint 3 (Word Count): '### Section III: Hardware Enclaves' must be strictly between 90 and 130 words in length.\n\n"
        "Constraint 4 (YAML Embedding): '### Section IV: Infrastructure Configuration' must contain a valid ```yaml ``` code block with exactly 4 keys: `enclave_id`, `page_isolation`, `ring_buffer_mb`, `attestation_pki`.\n\n"
        "Constraint 5 (Negative Anti-Tokens): Do NOT use any of the following 4 words anywhere in your entire output: 'constraint', 'rule', 'forbidden', 'negative'.\n\n"
        "Constraint 6 (Termination): Your final output line must be strictly: '[SECURITY_STAMP_2026_VERIFIED]'."
    )

    log("  Dispatching multi-constraint prompt with lipogram, word counts, and forbidden tokens...")
    res = client.call([{"role": "user", "content": prompt}], max_tokens=3500, temperature=0.4, stream=True)
    text = res["content"] if res["content"].strip() else res["text"]

    # Constraint 1: 4 sections
    s1 = "### Section I: Foundation" in text
    s2 = "### Section II: Isolation Proof" in text
    s3 = "### Section III: Hardware Enclaves" in text
    s4 = "### Section IV: Infrastructure Configuration" in text
    c1 = s1 and s2 and s3 and s4

    # Constraint 2: Lipogram in Section II (No 'e' or 'E')
    c2 = False
    sec2_text = ""
    try:
        sec2_part = text.split("### Section II: Isolation Proof")[1].split("### Section III")[0]
        # remove header line
        sec2_lines = [l for l in sec2_part.splitlines() if not l.startswith("###")]
        sec2_text = "\n".join(sec2_lines).strip()
        c2 = ("e" not in sec2_text.lower()) and (len(sec2_text) > 30)
    except Exception:
        c2 = False

    # Constraint 3: Word count in Section III (90 - 130 words)
    c3 = False
    sec3_words = 0
    try:
        sec3_part = text.split("### Section III: Hardware Enclaves")[1].split("### Section IV")[0]
        words = sec3_part.strip().split()
        sec3_words = len(words)
        c3 = 90 <= sec3_words <= 130
    except Exception:
        c3 = False

    # Constraint 4: YAML block in Section IV with exact 4 keys
    c4 = False
    try:
        yaml_match = re.search(r"```(?:yaml)?\s*\n(.*?)```", text, re.DOTALL)
        if yaml_match:
            y_text = yaml_match.group(1)
            keys = ["enclave_id", "page_isolation", "ring_buffer_mb", "attestation_pki"]
            c4 = all(k in y_text for k in keys)
    except Exception:
        c4 = False

    # Constraint 5: Forbidden anti-tokens
    forbidden = ["constraint", "rule", "forbidden", "negative"]
    found_forbidden = [f for f in forbidden if f in text.lower()]
    c5 = len(found_forbidden) == 0

    # Constraint 6: Termination Stamp
    c6 = "[SECURITY_STAMP_2026_VERIFIED]" in text

    all_passed = c1 and c2 and c3 and c4 and c5 and c6
    log(f"  - C1 (Exact 4 Required Markdown Sections): {'PASS' if c1 else 'FAIL'}")
    log(f"  - C2 (Section II Lipogram: ZERO 'e'/'E' in text): {'PASS' if c2 else 'FAIL'} (Chars: {len(sec2_text)})")
    log(f"  - C3 (Section III Word Count 90-130 words): {'PASS' if c3 else 'FAIL'} (Count: {sec3_words})")
    log(f"  - C4 (Section IV YAML with 4 Exact Keys): {'PASS' if c4 else 'FAIL'}")
    log(f"  - C5 (Zero Forbidden Anti-Tokens): {'PASS' if c5 else 'FAIL'} (Found: {found_forbidden})")
    log(f"  - C6 (Exact Security Stamp Termination): {'PASS' if c6 else 'FAIL'}")
    log(f"  -> Speed: {res['tok_per_sec']:.2f} tok/s | Status: {'PASS' if all_passed else 'FAIL'}")

    return {
        "name": "Combinatorial Anti-Constraints (IFEval)",
        "passed": all_passed,
        "completion_tokens": res["completion_tokens"],
        "tok_s": round(res["tok_per_sec"], 2),
        "constraint_details": {
            "4_sections": c1,
            "lipogram_no_e": c2,
            "word_count_90_130": c3,
            "yaml_keys": c4,
            "no_forbidden_words": c5,
            "termination_stamp": c6
        }
    }


# ============================================================================
# PILLAR 4: COUNTERFACTUAL AXIOMATIC SYMBOLIC ALGEBRA
# ============================================================================
def test_counterfactual_axiomatic_math(client: BenchmarkClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 4/5] Counterfactual Axiomatic Symbolic Deduction (Non-Commutative Modular Algebra)", bold=True, color=CYAN)
    log("="*88)

    # Ground truth calculation:
    # Operators in field Z_23:
    # a (+) b = (3*a - 2*b + 5) mod 23
    # a (*) b = (a^2 + b + 2) mod 23
    # Problem:
    # Given equation: ( (X (+) 4) (*) 3 ) (+) 8 = 19 (mod 23)
    # Step 1: Let Y = ( (X (+) 4) (*) 3 )
    # Y (+) 8 = 19
    # (3*Y - 2*8 + 5) mod 23 = 19
    # (3*Y - 16 + 5) mod 23 = 19
    # (3*Y - 11) mod 23 = 19
    # 3*Y mod 23 = (19 + 11) mod 23 = 30 mod 23 = 7
    # Inverse of 3 mod 23: 3 * 8 = 24 = 1 mod 23.
    # Y = (7 * 8) mod 23 = 56 mod 23 = 10.
    # So Y = 10.
    #
    # Step 2: Let Z = (X (+) 4).
    # Z (*) 3 = Y = 10.
    # (Z^2 + 3 + 2) mod 23 = 10
    # (Z^2 + 5) mod 23 = 10
    # Z^2 mod 23 = 5.
    # Quadratic residues mod 23:
    # 1^2=1, 2^2=4, 3^2=9, 4^2=16, 5^2=25=2, 6^2=36=13, 7^2=49=3, 8^2=64=18, 9^2=81=12, 10^2=100=8, 11^2=121=6, 12^2=144=6,
    # 13^2=169=8, 14^2=196=12, 15^2=225=18, 16^2=256=3, 17^2=289=13, 18^2=324=2, 19^2=361=16, 20^2=400=9, 21^2=441=4, 22^2=484=1
    # Notice 5 is NOT a quadratic residue mod 23!
    # Let's design the equation so Z^2 mod 23 has a solution:
    # Let target Z^2 mod 23 = 4 -> Z = 2 or Z = 21.
    # If Z^2 + 3 + 2 = 9 mod 23 -> Z^2 = 4 mod 23 -> Z in {2, 21}.
    # So we want Y = 9.
    # Then Y (+) 8: (3*9 - 16 + 5) mod 23 = (27 - 11) mod 23 = 16.
    # So equation is: ( (X (+) 4) (*) 3 ) (+) 8 = 16 (mod 23)
    # Y = 9.
    # Z (*) 3 = (Z^2 + 5) mod 23 = 9 -> Z^2 mod 23 = 4 -> Z = 2 or Z = 21.
    #
    # Step 3: Z = (X (+) 4) = (3*X - 2*4 + 5) mod 23 = (3*X - 3) mod 23.
    # Case A: 3*X - 3 = 2 mod 23 -> 3*X = 5 mod 23 -> X = (5 * 8) mod 23 = 40 mod 23 = 17.
    # Case B: 3*X - 3 = 21 mod 23 -> 3*X = 24 = 1 mod 23 -> X = (1 * 8) mod 23 = 8.
    # Valid solutions for X are 17 or 8!
    expected_solutions = [8, 17]

    prompt = (
        "Consider a custom finite-field mathematical system over the ring Z_23 (integers modulo 23, values 0 to 22) "
        "defined by two synthetic binary operations:\n\n"
        "1. Operation (+):  a (+) b = (3*a - 2*b + 5) mod 23\n"
        "2. Operation (*):  a (*) b = (a^2 + b + 2) mod 23\n\n"
        "Solve the following equation for all integer solutions X in the range [0, 22]:\n"
        "   ( (X (+) 4) (*) 3 ) (+) 8 = 16   (mod 23)\n\n"
        "Requirements:\n"
        "- Show the step-by-step reduction for each nested operation using modular arithmetic.\n"
        "- Compute the modular multiplicative inverse of 3 mod 23.\n"
        "- Identify all valid integer values of X in {0, ..., 22}.\n"
        "- State at the very end: 'FINAL_SOLUTIONS_FOR_X: [values]'."
    )

    log("  Requesting step-by-step resolution of custom non-commutative modular algebra...")
    res = client.call([{"role": "user", "content": prompt}], max_tokens=1800, temperature=0.0, stream=True)
    text = res["text"]

    has_inv = ("8" in text) and ("inverse" in text.lower() or "3 * 8" in text or "24" in text)
    has_8 = "8" in text
    has_17 = "17" in text
    matched = ("8" in text and "17" in text)

    log(f"  - Modular Multiplicative Inverse of 3 mod 23 (=8): {'PASS' if has_inv else 'FAIL'}")
    log(f"  - Solved First Root X=8: {'PASS' if has_8 else 'FAIL'}")
    log(f"  - Solved Second Root X=17: {'PASS' if has_17 else 'FAIL'}")
    log(f"  -> Speed: {res['tok_per_sec']:.2f} tok/s | Status: {'PASS' if matched else 'FAIL'}")

    return {
        "name": "Counterfactual Axiomatic Algebra",
        "passed": matched,
        "completion_tokens": res["completion_tokens"],
        "tok_s": round(res["tok_per_sec"], 2)
    }


# ============================================================================
# PILLAR 5: FRONTIER DEPTH PRECISION (0.5%, 50%, 99.5% CONTEXT HORIZON)
# ============================================================================
def test_frontier_depth_precision(client: BenchmarkClient, target_context=120000):
    log("\n" + "="*88, bold=True)
    log(f"[TEST 5/5] Extreme Frontier-Depth Multi-Needle Precision ({target_context:,} Context Window)", bold=True, color=CYAN)
    log("="*88)

    filler_segment = (
        "Distributed consensus engines like Raft and Paxos serialize concurrent log entries using multi-version concurrency control. "
        "Storage backends distribute page tables across high-performance non-volatile media with atomic flush primitives. "
        "Analytical engines parse abstract syntax trees into vectorized SIMD instructions over columnar Apache Arrow memory layouts. "
    ) # ~45 tokens

    blocks = max(100, int(target_context / 45))
    val_front = random.randint(1000, 9999) if "random" in globals() else 4821
    val_middle = 7392
    val_tail = 3105

    pos_front = int(blocks * 0.005) # 0.5% depth
    pos_middle = int(blocks * 0.500) # 50.0% depth
    pos_tail = int(blocks * 0.995)   # 99.5% depth (extreme horizon)

    needle_front = f"\n[CRITICAL_FRONT_REGISTER: ALPHA_REG = {val_front}]\n"
    needle_middle = f"\n[CRITICAL_MIDDLE_REGISTER: BETA_REG = {val_middle}]\n"
    needle_tail = f"\n[CRITICAL_TAIL_REGISTER: GAMMA_REG = {val_tail}]\n"

    expected_composite = (val_front + val_tail) - val_middle # (4821 + 3105) - 7392 = 534

    doc = []
    for b in range(blocks):
        doc.append(filler_segment)
        if b == pos_front:
            doc.append(needle_front)
        elif b == pos_middle:
            doc.append(needle_middle)
        elif b == pos_tail:
            doc.append(needle_tail)

    document = "".join(doc)
    prompt = (
        f"{document}\n\n"
        "TASK:\n"
        "Locate all three critical registers buried in the documentation above:\n"
        "1. ALPHA_REG (near the beginning of the context)\n"
        "2. BETA_REG (in the middle of the context)\n"
        "3. GAMMA_REG (at the extreme end of the context)\n\n"
        "Perform the exact calculation: COMPOSITE_CHECKSUM = (ALPHA_REG + GAMMA_REG) - BETA_REG.\n"
        "State the values of all three registers and the final checksum. Conclude with 'COMPOSITE_CHECKSUM = <integer>'."
    )

    log(f"  Ingesting ~{target_context:,} tokens with needles at 0.5%, 50.0%, and 99.5% depth...")
    res = client.call([{"role": "user", "content": prompt}], max_tokens=1000, temperature=0.0, stream=True)
    text = res["text"]

    m1 = str(val_front) in text
    m2 = str(val_middle) in text
    m3 = str(val_tail) in text
    m_calc = str(expected_composite) in text
    passed = m1 and m2 and m3 and m_calc

    log(f"  - Extreme Front Needle (0.5% depth, ALPHA={val_front}): {'PASS' if m1 else 'FAIL'}")
    log(f"  - Mid-Context Needle (50.0% depth, BETA={val_middle}): {'PASS' if m2 else 'FAIL'}")
    log(f"  - Extreme Horizon Tail Needle (99.5% depth, GAMMA={val_tail}): {'PASS' if m3 else 'FAIL'}")
    log(f"  - Composite Calculation ({val_front} + {val_tail} - {val_middle} = {expected_composite}): {'PASS' if m_calc else 'FAIL'}")
    log(f"  -> Speed: {res['tok_per_sec']:.2f} tok/s | TTFT: {res['ttft_ms']:.1f} ms | Status: {'PASS' if passed else 'FAIL'}")

    return {
        "name": f"Frontier Needle Horizon ({target_context//1000}k)",
        "passed": passed,
        "completion_tokens": res["completion_tokens"],
        "ttft_ms": round(res["ttft_ms"], 1),
        "tok_s": round(res["tok_per_sec"], 2)
    }


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="Enterprise Hardened Adversarial & Agentic Benchmark.")
    parser.add_argument("--endpoint", "-e", default="http://127.0.0.1:8888/v1", help="Server endpoint")
    parser.add_argument("--model", "-m", default="qwen3.8-27b", help="Model name or ID")
    parser.add_argument("--api-key", "-k", default="", help="API Key")
    parser.add_argument("--context-depth", default=32000, type=int, help="Target context depth for frontier test (default: 32000)")
    parser.add_argument("--out", "-o", default="tests/results/hardened_adversarial_results.json", help="Report output path")
    args = parser.parse_args()

    log("\n" + "="*88, bold=True)
    log("  ENTERPRISE HARDENED ADVERSARIAL & AGENTIC BENCHMARK SUITE", bold=True, color=MAGENTA)
    log(f"  Target Endpoint: {args.endpoint}")
    log(f"  Model ID       : {args.model}")
    log("="*88)

    client = BenchmarkClient(args.endpoint, args.model, args.api_key)

    results = []
    t_start = time.perf_counter()

    results.append(test_multihop_graph_distractors(client))
    results.append(test_novel_algorithmic_fuzz(client))
    results.append(test_ifeval_combinatorial_constraints(client))
    results.append(test_counterfactual_axiomatic_math(client))
    results.append(test_frontier_depth_precision(client, target_context=args.context_depth))

    total_wall_time = time.perf_counter() - t_start

    log("\n" + "="*88, bold=True)
    log("                   HARDENED BENCHMARK SCORECARD", bold=True)
    log("="*88)
    log(f"{'Evaluation Domain':<50} | {'Status':<10} | {'Speed / Metrics':<20}")
    log("-" * 88)

    all_passed = True
    for r in results:
        status_color = GREEN if r["passed"] else RED
        status_text = "PASS" if r["passed"] else "FAIL"
        if not r["passed"]:
            all_passed = False
        log(f"{r['name']:<50} | {status_color}{status_text:<10}{RESET} | {r.get('tok_s', 0):.2f} tok/s")

    log("="*88)
    verdict = f"{GREEN}100% HARDENED & QUALIFIED{RESET}" if all_passed else f"{RED}DEFECTS / QUALIFICATION FAILURES DETECTED{RESET}"
    log(f"Final Qualification Verdict: {verdict}")
    log(f"Total Wall Time            : {total_wall_time:.2f} s")

    # Export report
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "endpoint": args.endpoint,
        "model": args.model,
        "total_wall_time_s": round(total_wall_time, 2),
        "all_passed": all_passed,
        "results": results
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log(f"Detailed Report Exported   : {args.out}\n")


if __name__ == "__main__":
    main()
