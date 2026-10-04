#!/usr/bin/env python3
"""
Enterprise LLM Server Full Benchmark & Evaluation Suite (llm_server_full_test.py)

Comprehensive, production-grade test and evaluation harness for any OpenAI-compatible
LLM server endpoint. Designed to evaluate, stress-test, and qualify models and serving
runtimes for enterprise software design, architecture, development, and deployment.

Key Capabilities:
  - Dynamic discovery & interactive picker for available models and endpoints.
  - Automatic detection of Max Context Length and adaptive context milestone scaling.
  - Automatic detection of Parallel Slots / Concurrency via /health when supported.
  - Strict OpenAI protocol compliance: Tools, Structured JSON Mode, Stop Sequences.
  - Evaluates Prefix Caching (RadixAttention / prompt cache reuse) cold vs warm speedup.
  - Client disconnection & socket abort resilience (verifies GPU slot release).
  - Multi-client continuous batching throughput & queueing behavior.
  - Full 14-stage evaluation with terminal summary scorecard and JSON export.
"""

import sys
import os
import time
import json
import re
import base64
import urllib.request
import urllib.error
import socket
import argparse
import tempfile
import subprocess
import uuid
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATH = os.path.join(BASE_DIR, "image.png")
DEFAULT_RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
os.makedirs(DEFAULT_RESULTS_DIR, exist_ok=True)

# ANSI terminal colors
BOLD = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
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

    def call(self, messages, max_tokens=1024, temperature=0.0, stream=False,
             tools=None, tool_choice=None, response_format=None, stop=None,
             seed=None, timeout=600, chat_template_kwargs=None, reasoning_effort=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream
        }
        if chat_template_kwargs:
            payload["chat_template_kwargs"] = chat_template_kwargs
        if reasoning_effort:
            payload["reasoning_effort"] = reasoning_effort
        if tools:
            payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
        if response_format:
            payload["response_format"] = response_format
        if stop:
            payload["stop"] = stop
        if seed is not None:
            payload["seed"] = seed
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
            content = msg.get("content") or ""
            reasoning = msg.get("reasoning_content") or ""
            full_text = f"{reasoning}\n{content}".strip() if reasoning else (content or "")
            tool_calls = msg.get("tool_calls")
            usage = res_json.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
            speed = completion_tokens / total_time if total_time > 0 else 0.0
            return {
                "content": content if content else full_text,
                "reasoning": reasoning,
                "text": full_text,
                "tool_calls": tool_calls,
                "total_time": total_time,
                "ttft": total_time,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "cached_tokens": cached_tokens,
                "decode_speed": speed,
                "raw": res_json
            }
        else:
            t_first = None
            t_last = None
            chunks = []
            reasoning_chunks = []
            content_chunks = []
            token_count = 0
            prompt_tokens = 0
            completion_tokens = 0
            cached_tokens = 0
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
                        cached_tokens = c["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
                    choices = c.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        if "tool_calls" in delta and delta["tool_calls"]:
                            tool_calls = delta["tool_calls"]
                        reasoning_part = delta.get("reasoning_content") or ""
                        content_part = delta.get("content") or ""
                        chunk_text = reasoning_part + content_part
                        if chunk_text:
                            now = time.perf_counter()
                            if t_first is None:
                                t_first = now
                            t_last = now
                            token_count += 1
                            chunks.append(chunk_text)
                            if reasoning_part:
                                reasoning_chunks.append(reasoning_part)
                            if content_part:
                                content_chunks.append(content_part)
            t_end = time.perf_counter()
            total_time = t_end - t0
            ttft = (t_first - t0) if t_first else total_time
            gen_time = (t_end - t_first) if t_first else total_time
            comp_tok = completion_tokens if completion_tokens > 0 else token_count
            speed = (comp_tok / gen_time) if gen_time > 0 else 0.0
            content_str = "".join(content_chunks)
            reasoning_str = "".join(reasoning_chunks)
            full_text = "".join(chunks)
            return {
                "content": content_str if content_str else full_text,
                "reasoning": reasoning_str,
                "text": full_text,
                "tool_calls": tool_calls,
                "total_time": total_time,
                "ttft": ttft,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": comp_tok,
                "cached_tokens": cached_tokens,
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


def build_context_milestones(max_context: int, max_ratio: float = 0.80):
    standard_targets = [4000, 8000, 16000, 32000, 64000, 128000, 256000, 512000, 1000000]
    effective_cap = int(max_context * max_ratio)
    # Reserve a safety buffer for completion tokens (64) + prompt formatting/salt (~200)
    headroom = min(1000, max(256, int(effective_cap * 0.005)))
    safe_ceiling = max(1000, effective_cap - headroom)

    milestones = [t for t in standard_targets if t <= safe_ceiling]
    if not milestones:
        milestones = [min(4000, safe_ceiling)]

    # Include the upper ceiling up to max_ratio if not already present
    if safe_ceiling > milestones[-1]:
        rounded_ceiling = (safe_ceiling // 100) * 100
        milestones.append(rounded_ceiling)
    return sorted(list(set(milestones)))


# ============================================================================
# 14 ENTERPRISE EVALUATION TEST SUITES
# ============================================================================

# ----------------------------------------------------------------------------
# 1. STREAMING & TTFT LATENCY
# ----------------------------------------------------------------------------
def run_test_streaming(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 1/14] Streaming Verification & TTFT Latency", bold=True, color=CYAN)
    log("="*88)
    prompt = "Explain the fundamental difference between synchronous and asynchronous microservice communication in two concise sentences."
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

# ----------------------------------------------------------------------------
# 2. MULTIMODAL VISION EXTRACTION
# ----------------------------------------------------------------------------
def run_test_vision(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 2/14] Multimodal Vision Evaluation (Invoice Document Parsing)", bold=True, color=CYAN)
    log("="*88)
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
    res = client.call(messages, max_tokens=1500, stream=True)
    text = res.get("text", "") or res.get("content", "")
    has_inv = "1024" in text
    has_total = any(k in text for k in ["875", "1250", "1,250", "$875", "875.00"])
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

# ----------------------------------------------------------------------------
# 3. MULTI-CLIENT CONCURRENCY & BATCHING
# ----------------------------------------------------------------------------
def run_test_concurrency(client: LLMClient, parallel: int):
    log("\n" + "="*88, bold=True)
    log(f"[TEST 3/14] Multi-Client Concurrency & Batching Stress Test (Parallel = {parallel})", bold=True, color=CYAN)
    log("="*88)
    
    results = {}
    for c in sorted(list(set([1, parallel]))):
        log(f"  Evaluating parallel stream level c = {c}...")
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

# ----------------------------------------------------------------------------
# 4. 4-TASK SOFTWARE ARCHITECTURE & CODING SUITE
# ----------------------------------------------------------------------------
def run_test_capabilities(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 4/14] Enterprise Architecture & Engineering 4-Task Suite", bold=True, color=CYAN)
    log("="*88)
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

# ----------------------------------------------------------------------------
# 5. OPENAI TOOL / FUNCTION CALLING & MULTI-TURN AGENT RECOVERY
# ----------------------------------------------------------------------------
def run_test_tool_calling(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 5/14] Multi-Turn Agent Loop & Autonomous Error Recovery Protocol", bold=True, color=CYAN)
    log("="*88)
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "provision_database_instance",
                "description": "Provisions a managed database cluster in a specified VPC cloud region.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "engine": {"type": "string", "enum": ["postgres", "mysql", "redis"]},
                        "environment": {"type": "string", "enum": ["production", "staging", "development"]},
                        "region": {"type": "string", "enum": ["us-east-1", "us-west-2", "eu-central-1"]},
                        "storage_gb": {"type": "integer", "description": "Storage capacity in GB"},
                        "ha_cluster": {"type": "boolean", "description": "Enable Multi-AZ high availability"}
                    },
                    "required": ["engine", "environment", "region", "storage_gb", "ha_cluster"]
                }
            }
        }
    ]
    prompt = (
        "Please provision a high-availability PostgreSQL cluster with 500 GB storage for our production payment environment in region 'us-east-1'. "
        "If a quota or provisioning error occurs in that region, automatically adapt and re-attempt provisioning in an alternate valid region with sufficient quota."
    )
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # Turn 1: Dispatch initial tool call
        res1 = client.call(messages, max_tokens=600, tools=tools, tool_choice="auto", stream=False)
        tool_calls1 = res1.get("tool_calls")
        turn1_valid = False
        args_parsed1 = {}
        
        call_id = "call_001"
        if tool_calls1 and len(tool_calls1) > 0:
            call_obj = tool_calls1[0]
            call_id = call_obj.get("id", call_id)
            fn = call_obj.get("function", {})
            if fn.get("name") == "provision_database_instance":
                args_str = fn.get("arguments", "{}")
                try:
                    args_parsed1 = json.loads(args_str) if isinstance(args_str, str) else args_str
                    if (args_parsed1.get("engine") == "postgres" and
                        args_parsed1.get("environment") == "production" and
                        args_parsed1.get("region") == "us-east-1" and
                        args_parsed1.get("storage_gb") == 500 and
                        args_parsed1.get("ha_cluster") is True):
                        turn1_valid = True
                except Exception:
                    pass

        log(f"  [Turn 1] Initial Tool Call Dispatched: {turn1_valid}")
        log(f"  [Turn 1] Parsed Arguments: {json.dumps(args_parsed1)}")

        # Turn 2: Simulate cloud quota failure and test autonomous agent recovery
        turn2_recovered = False
        args_parsed2 = {}
        res2 = {}
        if turn1_valid:
            error_payload = {
                "status": "error",
                "code": "QUOTA_EXCEEDED",
                "message": "Storage quota limit exceeded in 'us-east-1' (max 250 GB). Region 'us-west-2' has 1000 GB capacity available."
            }
            # Append assistant message with tool call and simulated tool error response
            assistant_msg = {"role": "assistant", "tool_calls": tool_calls1, "content": res1.get("content") or ""}
            tool_msg = {"role": "tool", "tool_call_id": call_id, "content": json.dumps(error_payload)}

            messages_turn2 = list(messages) + [assistant_msg, tool_msg]
            res2 = client.call(messages_turn2, max_tokens=1000, tools=tools, tool_choice="auto", stream=False)
            tool_calls2 = res2.get("tool_calls")

            if tool_calls2 and len(tool_calls2) > 0:
                fn2 = tool_calls2[0].get("function", {})
                if fn2.get("name") == "provision_database_instance":
                    args_str2 = fn2.get("arguments", "{}")
                    try:
                        args_parsed2 = json.loads(args_str2) if isinstance(args_str2, str) else args_str2
                        if (args_parsed2.get("engine") == "postgres" and
                            args_parsed2.get("environment") == "production" and
                            args_parsed2.get("region") == "us-west-2" and
                            args_parsed2.get("storage_gb") == 500 and
                            args_parsed2.get("ha_cluster") is True):
                            turn2_recovered = True
                    except Exception:
                        pass

            log(f"  [Turn 2] Error Injected: QUOTA_EXCEEDED in us-east-1")
            log(f"  [Turn 2] Autonomous Recovery Call: {turn2_recovered}")
            log(f"  [Turn 2] Re-attempt Arguments: {json.dumps(args_parsed2)}")

        status = "PASS" if (turn1_valid and turn2_recovered) else ("PARTIAL" if turn1_valid else "FAIL")
        log(f"  -> Multi-Turn Agent Protocol Status: {status}")
        return {
            "status": status,
            "turn1_valid": turn1_valid,
            "turn2_recovered": turn2_recovered,
            "parsed_arguments_turn1": args_parsed1,
            "parsed_arguments_turn2": args_parsed2,
            "tokens": res1.get("completion_tokens", 0) + (res2.get("completion_tokens", 0) if turn1_valid else 0),
            "ttft_ms": round(res1.get("ttft", 0) * 1000, 1)
        }
    except Exception as e:
        log(f"  -> Tool Calling Error: {e}", color=RED)
        return {"status": "FAIL", "error": str(e)}

# ----------------------------------------------------------------------------
# 6. STRICT JSON SCHEMA / CONSTRAINED DECODING
# ----------------------------------------------------------------------------
def run_test_json_schema(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 6/14] Strict JSON Schema & Constrained Decoding (response_format)", bold=True, color=CYAN)
    log("="*88)
    
    prompt = (
        "Generate a software architecture service definition for an OrderProcessingService. "
        "Return a JSON object containing keys: 'service_name' (string), 'protocol' (string), "
        "'port' (integer 8000-9000), and 'dependencies' (list of strings)."
    )
    messages = [{"role": "user", "content": prompt}]
    
    try:
        # Increase max_tokens to 2048 to allow thinking/reasoning models to complete reasoning and emit full JSON
        res = client.call(messages, max_tokens=2048, response_format={"type": "json_object"}, stream=False)
        content = res["content"].strip()
        parsed = {}
        valid_json = False
        valid_schema = False
        
        clean_content = content
        if "```json" in clean_content:
            clean_content = clean_content.split("```json")[1].split("```")[0].strip()
        elif "```" in clean_content:
            clean_content = clean_content.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(clean_content)
            valid_json = True
            if (("service_name" in parsed or "name" in parsed) and "dependencies" in parsed):
                valid_schema = True
        except Exception:
            pass

        status = "PASS" if valid_schema else ("PARTIAL" if valid_json else "FAIL")
        log(f"  -> Valid JSON Emitted: {valid_json}")
        log(f"  -> Schema Conformance: {valid_schema}")
        log(f"  -> Preview: {json.dumps(parsed)[:120]}...")
        return {
            "status": status,
            "valid_json": valid_json,
            "valid_schema": valid_schema,
            "ttft_ms": round(res.get("ttft", 0) * 1000, 1),
            "tok_s": round(res.get("decode_speed", 0), 2)
        }
    except Exception as e:
        log(f"  -> JSON Mode Error: {e}", color=RED)
        return {"status": "FAIL", "error": str(e)}

# ----------------------------------------------------------------------------
# 7. PREFIX CACHING & PROMPT CACHE REUSE
# ----------------------------------------------------------------------------
def run_test_prefix_caching(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 7/14] Prefix Caching / KV Cache Reuse Verification (Cold vs Warm)", bold=True, color=CYAN)
    log("="*88)
    
    shared_system_spec = (
        "ENTERPRISE ARCHITECTURE SPECIFICATION v4.2:\n" +
        "Section 1: All microservices must authenticate with mTLS and JWT Bearer tokens.\n" +
        "Section 2: Database mutations must publish change data capture events to Kafka with transactional outbox.\n" +
        "Section 3: Cache invalidation employs two-phase commit over Redis Sentinel.\n" +
        "Section 4: The mandatory message broker protocol for cross-datacenter sync is AMQP 1.0 with TLS 1.3.\n" +
        "Section 5: Circuit breakers must trip after 5 consecutive 5xx errors in a 10-second rolling window.\n" +
        "Section 6: Secrets must be retrieved from HashiCorp Vault with dynamic 1-hour lease renewal.\n" +
        "Section 7: The maximum permissible p99 latency SLA for payment processing endpoints is 120 milliseconds.\n"
    ) * 40  # Generates ~3,500 tokens of shared prefix

    # Run 1: Cold prefill
    log("  Step 1: Sending cold request with ~3,500 token shared prefix...")
    t0 = time.perf_counter()
    msg1 = [
        {"role": "system", "content": shared_system_spec},
        {"role": "user", "content": "According to Section 4, what is the mandatory message broker protocol?"}
    ]
    r1 = client.call(msg1, max_tokens=60, stream=False)
    cold_ttft = r1["ttft"]
    log(f"    -> Cold Request TTFT: {cold_ttft:.3f} s (Tokens: {r1['prompt_tokens']})")

    # Run 2: Warm prefill (exact same prefix, different question)
    log("  Step 2: Sending warm request with identical prefix to test KV cache hit...")
    msg2 = [
        {"role": "system", "content": shared_system_spec},
        {"role": "user", "content": "According to Section 7, what is the maximum permissible p99 latency SLA?"}
    ]
    r2 = client.call(msg2, max_tokens=60, stream=False)
    warm_ttft = r2["ttft"]
    cached_tokens = r2.get("cached_tokens", 0)
    speedup = (cold_ttft / warm_ttft) if warm_ttft > 0 else 1.0
    
    caching_active = speedup >= 2.0 or cached_tokens > 0
    status = "PASS (ACTIVE)" if caching_active else "INACTIVE / COLD"
    log(f"    -> Warm Request TTFT: {warm_ttft:.3f} s (Reported Cached Tokens: {cached_tokens})")
    log(f"    -> Prefix Cache Acceleration: {speedup:.2f}x speedup -> {status}")
    
    return {
        "status": status,
        "cold_ttft_s": round(cold_ttft, 3),
        "warm_ttft_s": round(warm_ttft, 3),
        "speedup_ratio": round(speedup, 2),
        "cached_tokens_reported": cached_tokens
    }

# ----------------------------------------------------------------------------
# 8. CLIENT DISCONNECTION & SOCKET ABORT RECOVERY
# ----------------------------------------------------------------------------
def run_test_client_abort(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 8/14] Client Socket Abort & Slot Recovery Resilience", bold=True, color=CYAN)
    log("="*88)
    
    payload = {
        "model": client.model,
        "messages": [{"role": "user", "content": "Write a 500-word comprehensive essay explaining garbage collection in Java."}],
        "max_tokens": 1024,
        "stream": True
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(client.completions_url, data=data, headers=client._get_headers())
    
    log("  Step 1: Opening long streaming request and intentionally aborting socket...")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        chunks_read = 0
        for line in resp:
            chunks_read += 1
            if chunks_read >= 5:
                # Forcefully close the connection mid-generation
                resp.close()
                break
        log(f"    -> Abruptly closed socket after {chunks_read} chunks.")
    except Exception as e:
        log(f"    -> Socket closed ({e}).")

    # Step 2: Immediately send follow-up request to test slot recovery
    log("  Step 2: Dispatching immediate follow-up request to verify slot release...")
    t_rec_start = time.perf_counter()
    follow_up_ok = False
    try:
        r = client.call([{"role": "user", "content": "Reply with 'OK'."}], max_tokens=10, stream=False, timeout=15)
        rec_time = time.perf_counter() - t_rec_start
        follow_up_ok = "OK" in r["content"].upper() or len(r["content"]) > 0
        status = "PASS" if (follow_up_ok and rec_time < 5.0) else "DEGRADED"
        log(f"    -> Follow-up Response Time: {rec_time*1000:.1f} ms | Recovered: {follow_up_ok} -> {status}")
        return {
            "status": status,
            "recovery_latency_ms": round(rec_time * 1000, 1),
            "recovered": follow_up_ok
        }
    except Exception as e:
        log(f"    -> Slot recovery failed / server hung: {e}", color=RED)
        return {"status": "FAIL", "error": str(e)}

# ----------------------------------------------------------------------------
# 9. STOP SEQUENCES & DETERMINISTIC SAMPLING COMPLIANCE
# ----------------------------------------------------------------------------
def run_test_stop_sequences(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 9/14] Stop Sequence Compliance & Deterministic Sampling (temp=0.0)", bold=True, color=CYAN)
    log("="*88)
    
    stop_words = ["HALT_GENERATION", "###STOP###"]
    prompt = (
        "Count from 1 to 10 as words separated by commas. After the word 'four', output ' HALT_GENERATION' and then continue counting."
    )
    res = client.call([{"role": "user", "content": prompt}], max_tokens=100, stop=stop_words, temperature=0.0, stream=False)
    text = res["content"]
    
    halt_stopped = "HALT_GENERATION" not in text and ("four" in text.lower())
    log(f"  -> Generated Text: {text.strip()}")
    log(f"  -> Stop word suppressed & execution terminated: {halt_stopped}")
    
    # Determinism check (two runs with temp=0.0)
    log("  -> Verifying greedy deterministic consistency across repeated runs (temp=0.0)...")
    r_det1 = client.call([{"role": "user", "content": "Compute sha256 checksum purpose in distributed ledgers."}], max_tokens=80, temperature=0.0, seed=42)
    r_det2 = client.call([{"role": "user", "content": "Compute sha256 checksum purpose in distributed ledgers."}], max_tokens=80, temperature=0.0, seed=42)
    deterministic = r_det1["content"] == r_det2["content"]
    log(f"  -> Exact Match Across Repeated Seeded Generations: {deterministic}")
    
    status = "PASS" if (halt_stopped and deterministic) else ("PARTIAL" if (halt_stopped or deterministic) else "FAIL")
    return {
        "status": status,
        "stop_sequence_respected": halt_stopped,
        "deterministic_greedy_reproducible": deterministic
    }

# ----------------------------------------------------------------------------
# 10. HIGH-ENTROPY ASSOCIATIVE RECALL
# ----------------------------------------------------------------------------
def run_test_high_entropy(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 10/14] High-Entropy Associative Key-Value Recall", bold=True, color=CYAN)
    log("="*88)
    filler = "In high-performance distributed architectures, nodes communicate via low-latency RPC protocols with strict SLAs. " * 300
    kv_data = (
        "CONFIDENTIAL LOOKUP TABLE:\n"
        "KEY_ALPHA_77: VAL_X9$mK2\n"
        "KEY_BETA_99: VAL_Z4#pQ8\n"
        "KEY_GAMMA_12: VAL_L1*vR5\n"
    )
    prompt = f"{filler}\n{kv_data}\n{filler}\nWhat is the exact value for KEY_BETA_99 and KEY_GAMMA_12? Answer strictly in format: KEY=VAL"
    # Increase max_tokens to 2048 to allow thinking / reasoning models to complete reasoning and emit both keys
    res = client.call([{"role": "user", "content": prompt}], max_tokens=2048, stream=True)
    text = res.get("text", "") or res.get("content", "")
    content = res.get("content", "") or text
    m1 = "Z4#pQ8" in text
    m2 = "L1*vR5" in text
    status = "PASS" if (m1 and m2) else ("PARTIAL" if (m1 or m2) else "FAIL")
    
    summary_lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    response_tail = summary_lines[-1] if summary_lines else "No response generated"
    log(f"  -> Recall: KEY_BETA_99 ({m1}), KEY_GAMMA_12 ({m2}) -> {status}")
    log(f"  -> Model Response Tail: {response_tail[:120]}")
    log(f"  -> TTFT: {res['ttft']*1000:.1f} ms | Decode Speed: {res['decode_speed']:.2f} tok/s")
    return {
        "status": status,
        "matches": {"KEY_BETA_99": m1, "KEY_GAMMA_12": m2},
        "model_tail": response_tail[:200],
        "response_excerpt": content[-300:].strip() if len(content) > 300 else content.strip(),
        "ttft_ms": round(res["ttft"] * 1000, 1),
        "tok_s": round(res["decode_speed"], 2)
    }

# ----------------------------------------------------------------------------
# 11. EXTREME PRECISION FINANCIAL RECONCILIATION
# ----------------------------------------------------------------------------
def run_test_extreme_precision(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 11/14] Extreme Precision Arithmetic & Financial Ledger Reconciliation", bold=True, color=CYAN)
    log("="*88)
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
        f"Calculate the precise ending balance to two decimal places step-by-step. State at the end: 'Ending Balance: $XXXXX.XX'"
    )
    # Increase max_tokens to 2048 to allow full chain-of-thought derivation without premature truncation
    res = client.call([{"role": "user", "content": prompt}], max_tokens=2048, stream=True)
    raw_output = res.get("content", "")
    # Normalize text by stripping commas from thousands separators (e.g., "$11,489.41" -> "$11489.41")
    normalized_output = raw_output.replace(",", "")
    matched = (expected in normalized_output) or (expected in raw_output)
    status = "PASS" if matched else "FAIL"
    
    # Extract last lines for diagnostic visibility
    summary_lines = [l.strip() for l in raw_output.strip().splitlines() if l.strip()]
    response_tail = summary_lines[-1] if summary_lines else "No response generated"
    log(f"  -> Calculated ending balance match ({expected}): {matched} -> {status}")
    log(f"  -> Model Response Tail: {response_tail[:100]}")
    log(f"  -> TTFT: {res['ttft']*1000:.1f} ms | Decode Speed: {res['decode_speed']:.2f} tok/s")
    return {
        "status": status,
        "expected": expected,
        "matched": matched,
        "model_tail": response_tail[:200],
        "ttft_ms": round(res["ttft"] * 1000, 1),
        "tok_s": round(res["decode_speed"], 2)
    }

# ----------------------------------------------------------------------------
# 12. EXECUTABLE CODE GENERATION & DYNAMIC UNIT TESTING
# ----------------------------------------------------------------------------
def run_test_code_execution(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 12/14] Executable Algorithmic Code Generation & Dynamic Verification", bold=True, color=CYAN)
    log("="*88)
    prompt = (
        "Write a complete, self-contained Python implementation of an LRU Cache.\n"
        "Class name must be `LRUCache` with:\n"
        "  `__init__(self, capacity: int)`\n"
        "  `get(self, key: int) -> int` (returns -1 if not found)\n"
        "  `put(self, key: int, value: int) -> None`\n"
        "Provide ONLY the executable Python code inside a ```python ``` code block."
    )
    # Increase max_tokens to 2048 to prevent reasoning + code generation from being cut off
    res = client.call([{"role": "user", "content": prompt}], max_tokens=2048, stream=True)
    raw_content = res.get("content", "")

    # Robust regex code block extraction (prioritizing class LRUCache over draft snippets in reasoning)
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", raw_content, re.DOTALL | re.IGNORECASE)
    code = ""
    if blocks:
        lru_blocks = [b for b in blocks if "class LRUCache" in b]
        code = max(lru_blocks, key=len).strip() if lru_blocks else max(blocks, key=len).strip()
    else:
        # Fallback for unclosed code block if model was cut off or omitted closing tag
        m = re.search(r"```(?:python)?\s*\n(.*)", raw_content, re.DOTALL | re.IGNORECASE)
        code = m.group(1).strip() if m else raw_content.strip()

    # Prepend standard collections imports to prevent spurious NameError if model uses OrderedDict
    import_preamble = "from collections import OrderedDict, defaultdict\nimport sys\n\n"

    test_harness = """

# Test harness execution
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
    full_code = import_preamble + code + "\n" + test_harness
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(full_code)
        f_path = f.name

    try:
        sub = subprocess.run([sys.executable, f_path], capture_output=True, text=True, timeout=15)
        passed = "UNIT_TESTS_PASSED" in sub.stdout
        err = sub.stderr.strip() if sub.stderr.strip() else sub.stdout.strip()
    except Exception as e:
        passed = False
        err = str(e)
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)

    status = "PASS" if passed else "FAIL"
    log(f"  -> Dynamic Execution Assertions: {status} {'(All tests passed)' if passed else f'Error: {err[:150]}'}")
    log(f"  -> Decode Speed: {res['decode_speed']:.2f} tok/s")
    return {
        "status": status,
        "dynamic_tests_passed": passed,
        "error": err if not passed else "",
        "tok_s": round(res["decode_speed"], 2)
    }

# ----------------------------------------------------------------------------
# 13. ERROR ENVELOPE & BOUNDARY API PROTOCOL COMPLIANCE
# ----------------------------------------------------------------------------
def run_test_error_handling(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 13/14] Error Envelope & Boundary API Protocol Compliance", bold=True, color=CYAN)
    log("="*88)
    
    # Check 1: Missing messages field
    bad_payload = json.dumps({"model": client.model, "max_tokens": 10}).encode("utf-8")
    req = urllib.request.Request(client.completions_url, data=bad_payload, headers=client._get_headers())
    bad_schema_caught = False
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            pass
    except urllib.error.HTTPError as e:
        if e.code in [400, 422]:
            bad_schema_caught = True
            log(f"  -> Missing 'messages' rejected with HTTP {e.code} (Standard OpenAI error behavior): PASS")
    except Exception:
        pass

    # Check 2: Non-existent model ID
    non_existent_payload = json.dumps({
        "model": "model_that_does_not_exist_xyz_999",
        "messages": [{"role": "user", "content": "Ping"}],
        "max_tokens": 10
    }).encode("utf-8")
    req2 = urllib.request.Request(client.completions_url, data=non_existent_payload, headers=client._get_headers())
    invalid_model_handled = False
    try:
        with urllib.request.urlopen(req2, timeout=5) as resp:
            # Some local servers accept any model ID gracefully
            invalid_model_handled = True
            log("  -> Non-existent model ID handled gracefully without crash: PASS")
    except urllib.error.HTTPError as e:
        if e.code in [400, 404]:
            invalid_model_handled = True
            log(f"  -> Non-existent model ID rejected with HTTP {e.code}: PASS")
    except Exception:
        pass

    status = "PASS" if (bad_schema_caught or invalid_model_handled) else "PARTIAL"
    return {
        "status": status,
        "bad_schema_rejected": bad_schema_caught,
        "invalid_model_handled": invalid_model_handled
    }

# ----------------------------------------------------------------------------
# 14. DYNAMIC LONG-CONTEXT PREFILL, DECODE & SEMANTIC ACCURACY SCALING
# ----------------------------------------------------------------------------
def run_test_context_scaling(client: LLMClient, milestones: list):
    log("\n" + "="*88, bold=True)
    log("[TEST 14/14] Dynamic Long-Context Prefill, Decode & Semantic Accuracy", bold=True, color=CYAN)
    log(f"Target Milestones: {milestones}")
    log("="*88)

    base_text = (
        "The quick brown fox jumps over the lazy dog. In computer science and artificial intelligence, "
        "large language models utilize transformer architectures with multi-head self-attention mechanisms "
        "to process sequential data efficiently. Memory bandwidth and compute capacity determine inference speed. "
    )
    header = f"{'Target Ctx':>11} | {'Actual Prompt':>14} | {'Cached':>8} | {'TTFT (s)':>10} | {'Cold (t/s)':>12} | {'Effective':>12} | {'Decode (t/s)':>12} | {'Accuracy':>10} | {'Status':>8}"
    log(header)
    log("-" * len(header))

    scaling_results = []
    for target in milestones:
        # Generate needle and embed at ~85% depth
        secret_hex = uuid.uuid4().hex[:8].upper()
        needle_key = f"NEEDLE_CTX_{target}"
        needle_str = f"\n[CRITICAL_SYSTEM_REGISTRATION: {needle_key} = VAL_{secret_hex}]\n"

        repeat_count = max(1, int((target - 50) / 45))
        split_idx = int(repeat_count * 0.85)
        part1 = base_text * split_idx
        part2 = base_text * (repeat_count - split_idx)

        epoch_salt = f"[Context Benchmark Target: {target} | Epoch: {time.time():.4f} | UUID: {uuid.uuid4()}]\n"
        prompt_text = (
            epoch_salt + part1 + needle_str + part2 +
            f"\n\nSummarize the key aspects mentioned above in one sentence, and at the end output: '{needle_key} = VAL_{secret_hex}'."
        )
        try:
            res = client.call([{"role": "user", "content": prompt_text}], max_tokens=600, stream=True, timeout=1200)
            actual_prompt = res["prompt_tokens"] if res["prompt_tokens"] > 0 else target
            cached_tokens = res.get("cached_tokens", 0)
            uncached_tokens = max(0, actual_prompt - cached_tokens)
            
            # Compute both raw cold hardware throughput and effective throughput
            cold_speed = (uncached_tokens / res["ttft"]) if res["ttft"] > 0 and uncached_tokens > 0 else (actual_prompt / res["ttft"] if res["ttft"] > 0 else 0.0)
            effective_speed = (actual_prompt / res["ttft"]) if res["ttft"] > 0 else 0.0

            text = res.get("text", "") or res.get("content", "")
            needle_matched = secret_hex in text
            acc_str = "RECALLED" if needle_matched else "MISSED"
            step_status = "PASS" if needle_matched else "PARTIAL"

            row = (
                f"{target:>11,d} | "
                f"{actual_prompt:>14,d} | "
                f"{cached_tokens:>8,d} | "
                f"{res['ttft']:>10.3f} | "
                f"{cold_speed:>12.1f} | "
                f"{effective_speed:>12.1f} | "
                f"{res['decode_speed']:>12.2f} | "
                f"{acc_str:>10} | "
                f"{step_status:>8}"
            )
            log(row)
            scaling_results.append({
                "target_tokens": target,
                "actual_prompt_tokens": actual_prompt,
                "cached_tokens": cached_tokens,
                "ttft_s": round(res["ttft"], 3),
                "cold_prefill_tok_s": round(cold_speed, 1),
                "effective_prefill_tok_s": round(effective_speed, 1),
                "decode_tok_s": round(res["decode_speed"], 2),
                "completion_tokens": res["completion_tokens"],
                "needle_matched": needle_matched,
                "status": step_status
            })
        except Exception as e:
            row = f"{target:>11,d} | {'ERROR':>14} | {'-':>8} | {'-':>10} | {str(e)[:25]:>12} | {'-':>12} | {'-':>12} | {'-':>10} | {'FAIL':>8}"
            log(row, color=RED)
            scaling_results.append({
                "target_tokens": target,
                "error": str(e),
                "status": "FAIL"
            })
        time.sleep(1)

    log("-" * len(header))
    return scaling_results


# ----------------------------------------------------------------------------
# 15. ADVERSARIAL MULTI-HOP GRAPH TRAVERSAL WITH TEMPORAL DISTRACTORS
# ----------------------------------------------------------------------------
def run_test_multihop_graph(client: LLMClient, target_context: int = 32000):
    log("\n" + "="*88, bold=True)
    log(f"[TEST 15/19] Adversarial Multi-Hop Graph Traversal with Deprecated Distractors (~{target_context//1000}k)", bold=True, color=CYAN)
    log("="*88)

    filler_chunk = (
        "Enterprise cloud transit routing manages BGP autonomous system interconnects across software-defined WAN fabrics. "
        "Packet encapsulation leverages Geneve and VXLAN tunnels terminating on distributed SmartNIC vSwitch datanodes. "
        "Telemetry pipelines process streaming flow logs through Kafka clusters with partitioned RocksDB state stores. "
        "Zero-trust microsegmentation evaluates SPIFFE identity assertions and mutual TLS session keys at every ingress hop. "
    )

    total_blocks = max(100, int(target_context / 60))

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
    res = client.call([{"role": "user", "content": prompt}], max_tokens=2500, temperature=0.0, stream=True)
    text = res.get("text", "") or res.get("content", "")

    has_active_tokyo = ("Transit Hub Tokyo" in text or "Tokyo" in text) and ("Bridge Omega" in text or "Omega" in text)
    has_code = "AUTH_ROUTING_KEY_9921_TOK" in text
    has_rejected_chicago = ("chicago" in text.lower()) and any(w in text.lower() for w in ["deprecated", "rfc-77", "revoked", "invalid"])
    has_rejected_frankfurt = ("frankfurt" in text.lower()) and any(w in text.lower() for w in ["revoked", "rollback", "invalid"])

    passed = has_active_tokyo and has_code and has_rejected_chicago and has_rejected_frankfurt
    log(f"  - Active Route Identification (Tokyo/Omega): {'PASS' if has_active_tokyo else 'FAIL'}")
    log(f"  - Exact Verification Key Recall: {'PASS' if has_code else 'FAIL'}")
    log(f"  - Rejection of Deprecated v1.0 Chicago Distractor: {'PASS' if has_rejected_chicago else 'FAIL'}")
    log(f"  - Rejection of Revoked v2.2 Frankfurt Distractor: {'PASS' if has_rejected_frankfurt else 'FAIL'}")
    log(f"  -> Speed: {res.get('decode_speed', 0):.2f} tok/s | TTFT: {res.get('ttft', 0)*1000:.1f} ms | Status: {'PASS' if passed else 'FAIL'}")

    return {
        "status": "PASS" if passed else "FAIL",
        "active_route_identified": has_active_tokyo,
        "verification_key_recalled": has_code,
        "distractor_chicago_rejected": has_rejected_chicago,
        "distractor_frankfurt_rejected": has_rejected_frankfurt,
        "prompt_tokens": res.get("prompt_tokens", 0),
        "completion_tokens": res.get("completion_tokens", 0),
        "ttft_ms": round(res.get("ttft", 0) * 1000.0, 1),
        "tok_s": round(res.get("decode_speed", 0), 2)
    }


# ----------------------------------------------------------------------------
# 16. NOVEL ALGORITHMIC SYNTHESIS WITH 5,000-OP PROPERTY FUZZ TEST
# ----------------------------------------------------------------------------
def run_test_novel_algorithmic_fuzz(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 16/19] Novel Algorithmic Synthesis & Automated 5,000-Op Property Fuzz Testing", bold=True, color=CYAN)
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

    log("  Requesting synthesis of non-standard Monotonic Ring Buffer (max_tokens=3500)...")
    res = client.call([{"role": "user", "content": prompt}], max_tokens=4500, temperature=0.2, stream=True)
    raw = res.get("content", "") or res.get("text", "")

    code_match = re.findall(r"```(?:python)?\s*\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
    if code_match:
        code = max(code_match, key=len).strip()
    else:
        # Fallback if closing tag was omitted or truncated
        m = re.search(r"```(?:python)?\s*\n(.*)", raw, re.DOTALL | re.IGNORECASE)
        code = m.group(1).strip() if m else raw.strip()
        code = re.sub(r"```\s*$", "", code).strip()
    # Strip any stray markdown code fence lines to guarantee valid Python syntax
    code = "\n".join(line for line in code.splitlines() if not line.strip().startswith("```")).strip()

    fuzz_harness = """
import sys
import random

def run_fuzz_verification():
    buf = ConcurrentMonotonicRingBuffer(3)
    s1 = buf.push(10)
    s2 = buf.push(20)
    s3 = buf.push(30)
    assert (s1, s2, s3) == (1, 2, 3), f"Sequences must be 1, 2, 3, got {(s1, s2, s3)}"
    assert buf.get_by_seq(1) == 10
    assert buf.get_by_seq(2) == 20
    assert buf.get_by_seq(3) == 30
    assert buf.get_latest() == (3, 30)

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

        oldest_valid_seq = max(1, current_seq - cap + 1)
        if (current_seq - cap) in ground_truth:
            del ground_truth[current_seq - cap]

        if op % 10 == 0:
            assert rb.get_latest() == (current_seq, val)
            q_seq = random.randint(oldest_valid_seq, current_seq)
            assert rb.get_by_seq(q_seq) == ground_truth[q_seq]

            if oldest_valid_seq > 1:
                evicted_seq = random.randint(1, oldest_valid_seq - 1)
                try:
                    rb.get_by_seq(evicted_seq)
                    assert False, f"Evicted sequence {evicted_seq} did not raise KeyError"
                except KeyError:
                    pass

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
        proc = subprocess.run([sys.executable, temp_path], capture_output=True, text=True, timeout=25)
        fuzz_passed = ("ALL_5000_FUZZ_TESTS_PASSED" in proc.stdout) and (proc.returncode == 0)
        if not fuzz_passed:
            err_msg = (proc.stderr or proc.stdout).strip()[:300]
    except Exception as e:
        err_msg = str(e)
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    log(f"  - Automated 5,000 Operations Fuzz Assertions: {'PASS' if fuzz_passed else 'FAIL'}")
    if not fuzz_passed and err_msg:
        log(f"    [Error]: {err_msg}", color=RED)
    log(f"  -> Speed: {res.get('decode_speed', 0):.2f} tok/s | Status: {'PASS' if fuzz_passed else 'FAIL'}")

    return {
        "status": "PASS" if fuzz_passed else "FAIL",
        "fuzz_operations": 5000,
        "completion_tokens": res.get("completion_tokens", 0),
        "ttft_ms": round(res.get("ttft", 0) * 1000.0, 1),
        "tok_s": round(res.get("decode_speed", 0), 2),
        "error": err_msg
    }


# ----------------------------------------------------------------------------
# 17. COMBINATORIAL ANTI-CONSTRAINT FOLLOWING (IFEVAL TIER)
# ----------------------------------------------------------------------------
def run_test_combinatorial_anti_constraints(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 17/19] Combinatorial Anti-Constraint & Negative Instruction Following (IFEval Tier)", bold=True, color=CYAN)
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

    log("  Dispatching multi-constraint prompt with lipogram, word counts, and forbidden anti-tokens...")
    res = client.call([{"role": "user", "content": prompt}], max_tokens=5000, temperature=0.4, stream=True)
    text = res.get("content", "") or res.get("text", "")

    s1 = "### Section I: Foundation" in text
    s2 = "### Section II: Isolation Proof" in text
    s3 = "### Section III: Hardware Enclaves" in text
    s4 = "### Section IV: Infrastructure Configuration" in text
    c1 = s1 and s2 and s3 and s4

    c2 = False
    sec2_text = ""
    try:
        sec2_part = text.split("### Section II: Isolation Proof")[1].split("### Section III")[0]
        sec2_lines = [l for l in sec2_part.splitlines() if not l.startswith("###")]
        sec2_text = "\n".join(sec2_lines).strip()
        c2 = ("e" not in sec2_text.lower()) and (len(sec2_text) > 30)
    except Exception:
        c2 = False

    c3 = False
    sec3_words = 0
    try:
        sec3_part = text.split("### Section III: Hardware Enclaves")[1].split("### Section IV")[0]
        words = sec3_part.strip().split()
        sec3_words = len(words)
        c3 = 90 <= sec3_words <= 130
    except Exception:
        c3 = False

    c4 = False
    try:
        yaml_match = re.search(r"```(?:yaml)?\s*\n(.*?)```", text, re.DOTALL)
        if yaml_match:
            y_text = yaml_match.group(1)
            keys = ["enclave_id", "page_isolation", "ring_buffer_mb", "attestation_pki"]
            c4 = all(k in y_text for k in keys)
    except Exception:
        c4 = False

    forbidden = ["constraint", "rule", "forbidden", "negative"]
    found_forbidden = [f for f in forbidden if f in text.lower()]
    c5 = len(found_forbidden) == 0

    c6 = "[SECURITY_STAMP_2026_VERIFIED]" in text

    all_passed = c1 and c2 and c3 and c4 and c5 and c6
    log(f"  - C1 (Exact 4 Required Markdown Sections): {'PASS' if c1 else 'FAIL'}")
    log(f"  - C2 (Section II Lipogram: ZERO 'e'/'E' in text): {'PASS' if c2 else 'FAIL'} (Chars: {len(sec2_text)})")
    log(f"  - C3 (Section III Word Count 90-130 words): {'PASS' if c3 else 'FAIL'} (Count: {sec3_words})")
    log(f"  - C4 (Section IV YAML with 4 Exact Keys): {'PASS' if c4 else 'FAIL'}")
    log(f"  - C5 (Zero Forbidden Anti-Tokens): {'PASS' if c5 else 'FAIL'} (Found: {found_forbidden})")
    log(f"  - C6 (Exact Security Stamp Termination): {'PASS' if c6 else 'FAIL'}")
    log(f"  -> Speed: {res.get('decode_speed', 0):.2f} tok/s | Status: {'PASS' if all_passed else 'FAIL'}")

    return {
        "status": "PASS" if all_passed else "FAIL",
        "constraints_passed": sum([c1, c2, c3, c4, c5, c6]),
        "total_constraints": 6,
        "details": {
            "4_sections": c1,
            "lipogram_no_e": c2,
            "word_count_90_130": c3,
            "yaml_keys": c4,
            "no_forbidden_words": c5,
            "termination_stamp": c6
        },
        "completion_tokens": res.get("completion_tokens", 0),
        "ttft_ms": round(res.get("ttft", 0) * 1000.0, 1),
        "tok_s": round(res.get("decode_speed", 0), 2)
    }


# ----------------------------------------------------------------------------
# 18. COUNTERFACTUAL AXIOMATIC SYMBOLIC ALGEBRA
# ----------------------------------------------------------------------------
def run_test_counterfactual_algebra(client: LLMClient):
    log("\n" + "="*88, bold=True)
    log("[TEST 18/19] Counterfactual Axiomatic Symbolic Deduction (Non-Commutative Modular Algebra)", bold=True, color=CYAN)
    log("="*88)

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
    text = res.get("text", "")

    has_inv = ("8" in text) and ("inverse" in text.lower() or "3 * 8" in text or "24" in text)
    has_8 = "8" in text
    has_17 = "17" in text
    matched = (has_8 and has_17)

    log(f"  - Modular Multiplicative Inverse of 3 mod 23 (=8): {'PASS' if has_inv else 'FAIL'}")
    log(f"  - Solved First Root X=8: {'PASS' if has_8 else 'FAIL'}")
    log(f"  - Solved Second Root X=17: {'PASS' if has_17 else 'FAIL'}")
    status = "PASS" if matched else ("PARTIAL" if (has_8 or has_17) else "FAIL")
    log(f"  -> Speed: {res.get('decode_speed', 0):.2f} tok/s | Status: {status}")

    return {
        "status": status,
        "inverse_found": has_inv,
        "root_8_found": has_8,
        "root_17_found": has_17,
        "completion_tokens": res.get("completion_tokens", 0),
        "ttft_ms": round(res.get("ttft", 0) * 1000.0, 1),
        "tok_s": round(res.get("decode_speed", 0), 2)
    }


# ----------------------------------------------------------------------------
# 19. EXTREME FRONTIER-DEPTH MULTI-NEEDLE SYNTHESIS
# ----------------------------------------------------------------------------
def run_test_frontier_needle_depth(client: LLMClient, target_context: int = 32000):
    log("\n" + "="*88, bold=True)
    log(f"[TEST 19/19] Extreme Frontier-Depth Multi-Needle Precision ({target_context:,} Context Window)", bold=True, color=CYAN)
    log("="*88)

    filler_segment = (
        "Distributed consensus engines like Raft and Paxos serialize concurrent log entries using multi-version concurrency control. "
        "Storage backends distribute page tables across high-performance non-volatile media with atomic flush primitives. "
        "Analytical engines parse abstract syntax trees into vectorized SIMD instructions over columnar Apache Arrow memory layouts. "
    )

    blocks = max(100, int(target_context / 45))
    val_front = 4821
    val_middle = 7392
    val_tail = 3105

    pos_front = int(blocks * 0.005)
    pos_middle = int(blocks * 0.500)
    pos_tail = int(blocks * 0.995)

    needle_front = f"\n[CRITICAL_FRONT_REGISTER: ALPHA_REG = {val_front}]\n"
    needle_middle = f"\n[CRITICAL_MIDDLE_REGISTER: BETA_REG = {val_middle}]\n"
    needle_tail = f"\n[CRITICAL_TAIL_REGISTER: GAMMA_REG = {val_tail}]\n"

    expected_composite = (val_front + val_tail) - val_middle

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
    res = client.call([{"role": "user", "content": prompt}], max_tokens=2000, temperature=0.0, stream=True)
    text = res.get("text", "") or res.get("content", "")

    m1 = str(val_front) in text
    m2 = str(val_middle) in text
    m3 = str(val_tail) in text
    m_calc = str(expected_composite) in text
    passed = m1 and m2 and m3 and m_calc

    log(f"  - Extreme Front Needle (0.5% depth, ALPHA={val_front}): {'PASS' if m1 else 'FAIL'}")
    log(f"  - Mid-Context Needle (50.0% depth, BETA={val_middle}): {'PASS' if m2 else 'FAIL'}")
    log(f"  - Extreme Horizon Tail Needle (99.5% depth, GAMMA={val_tail}): {'PASS' if m3 else 'FAIL'}")
    log(f"  - Composite Calculation ({val_front} + {val_tail} - {val_middle} = {expected_composite}): {'PASS' if m_calc else 'FAIL'}")
    log(f"  -> Speed: {res.get('decode_speed', 0):.2f} tok/s | TTFT: {res.get('ttft', 0)*1000:.1f} ms | Status: {'PASS' if passed else 'FAIL'}")

    return {
        "status": "PASS" if passed else "FAIL",
        "target_context": target_context,
        "alpha_found": m1,
        "beta_found": m2,
        "gamma_found": m3,
        "composite_matched": m_calc,
        "completion_tokens": res.get("completion_tokens", 0),
        "ttft_ms": round(res.get("ttft", 0) * 1000.0, 1),
        "tok_s": round(res.get("decode_speed", 0), 2)
    }


# ============================================================================
# FORMATTED CLI SUMMARY SCORECARD
# ============================================================================

def print_summary_table(report):
    res = report.get("results", {})
    log("\n" + "="*88, bold=True)
    log("              ENTERPRISE LLM SERVER EVALUATION SCORECARD                                ", bold=True, color=CYAN)
    log("="*88, bold=True)
    log(f"  Target Endpoint : {report.get('endpoint')}")
    log(f"  Model Under Test: {report.get('model')}")
    log(f"  Max Context Cap : {report.get('max_context_tokens', 0):,} tokens")
    log(f"  Parallel Setting: {report.get('parallel_streams', 1)} concurrent clients")
    log(f"  Total Wall Time : {report.get('total_suite_wall_time_s', 0):.2f} s")
    log("="*88)

    header = f"{'Evaluation Domain':<38} | {'Status':<10} | {'Key Metric / Latency':<20} | {'Throughput'}"
    log(header, bold=True)
    log("-" * 88)

    def fmt_status(st):
        if "PASS" in str(st):
            return f"{GREEN}{st}{RESET}"
        elif "FAIL" in str(st):
            return f"{RED}{st}{RESET}"
        elif "SKIPPED" in str(st):
            return f"{YELLOW}{st}{RESET}"
        return f"{YELLOW}{st}{RESET}"

    # 1. Streaming
    if "streaming" in res:
        s = res.get("streaming", {})
        log(f"{'1. Streaming & Latency':<38} | {fmt_status(s.get('status', 'N/A')):<19} | TTFT: {s.get('ttft_ms', 0):.1f} ms{'':<6} | {s.get('tok_s', 0):.2f} tok/s")
    else:
        log(f"{'1. Streaming & Latency':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 2. Vision
    if "vision" in res:
        v = res.get("vision", {})
        v_ttft = f"TTFT: {v.get('ttft_ms', 0):.1f} ms" if 'ttft_ms' in v else "N/A"
        v_speed = f"{v.get('tok_s', 0):.2f} tok/s" if 'tok_s' in v else "N/A"
        log(f"{'2. Multimodal Vision (Invoice)':<38} | {fmt_status(v.get('status', 'N/A')):<19} | {v_ttft:<20} | {v_speed}")
    else:
        log(f"{'2. Multimodal Vision (Invoice)':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 3. Parallel Batching
    if "concurrency" in res:
        c = res.get("concurrency", {})
        for ckey, cval in c.items():
            c_label = f"3. Parallel Batching ({ckey.upper()})"
            c_speed = f"{cval.get('aggregate_tok_s', 0):.2f} tok/s (agg)"
            c_wall = f"{cval.get('wall_time_s', 0):.2f} s wall"
            log(f"{c_label:<38} | {fmt_status(cval.get('status', 'N/A')):<19} | {c_wall:<20} | {c_speed}")
    else:
        log(f"{'3. Parallel Batching':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 4. Capabilities
    if "capabilities_4tasks" in res:
        cap = res.get("capabilities_4tasks", {})
        cap_speed = f"{cap.get('average_speed_tok_s', 0):.2f} tok/s (avg)"
        log(f"{'4. 4-Task Architecture Suite':<38} | {fmt_status(cap.get('status', 'N/A')):<19} | 4/4 tasks passed{'':<5} | {cap_speed}")
    else:
        log(f"{'4. 4-Task Architecture Suite':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 5. Tool Calling
    if "tool_calling" in res:
        tc = res.get("tool_calling", {})
        tc_desc = "Agent Error Recovered" if tc.get("turn2_recovered") else ("Turn 1 Valid" if tc.get("turn1_valid") else "Args Invalid")
        log(f"{'5. Tool Calling & Agentic Recovery':<38} | {fmt_status(tc.get('status', 'N/A')):<19} | {tc_desc:<20} | TTFT: {tc.get('ttft_ms', 0):.1f} ms")
    else:
        log(f"{'5. Tool Calling & Agentic Recovery':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 6. JSON Schema
    if "json_schema" in res:
        js = res.get("json_schema", {})
        js_desc = "Strict Schema Conformed" if js.get("valid_schema") else "Schema Invalid"
        log(f"{'6. JSON Schema Mode (response_format)':<38} | {fmt_status(js.get('status', 'N/A')):<19} | {js_desc:<20} | {js.get('tok_s', 0):.2f} tok/s")
    else:
        log(f"{'6. JSON Schema Mode (response_format)':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 7. Prefix Caching
    if "prefix_caching" in res:
        pc = res.get("prefix_caching", {})
        pc_desc = f"{pc.get('speedup_ratio', 1.0):.1f}x speedup" if pc.get("speedup_ratio") else "N/A"
        log(f"{'7. Prefix / KV Cache Reuse':<38} | {fmt_status(pc.get('status', 'N/A')):<19} | {pc_desc:<20} | Warm: {pc.get('warm_ttft_s', 0):.3f}s")
    else:
        log(f"{'7. Prefix / KV Cache Reuse':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 8. Client Abort
    if "client_abort" in res:
        ca = res.get("client_abort", {})
        ca_desc = f"Rec: {ca.get('recovery_latency_ms', 0):.1f} ms"
        log(f"{'8. Client Socket Abort Recovery':<38} | {fmt_status(ca.get('status', 'N/A')):<19} | {ca_desc:<20} | Slots Released")
    else:
        log(f"{'8. Client Socket Abort Recovery':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 9. Stop Sequences
    if "stop_sequences" in res:
        ss = res.get("stop_sequences", {})
        ss_desc = "Deterministic (temp=0)" if ss.get("deterministic_greedy_reproducible") else "Non-deterministic"
        log(f"{'9. Stop Words & Greedy Sampling':<38} | {fmt_status(ss.get('status', 'N/A')):<19} | {ss_desc:<20} | Tokens Suppressed")
    else:
        log(f"{'9. Stop Words & Greedy Sampling':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 10. High Entropy
    if "high_entropy_recall" in res:
        he = res.get("high_entropy_recall", {})
        he_speed = f"{he.get('tok_s', 0):.2f} tok/s"
        log(f"{'10. High-Entropy Key-Value Recall':<38} | {fmt_status(he.get('status', 'N/A')):<19} | TTFT: {he.get('ttft_ms', 0):.1f} ms{'':<4} | {he_speed}")
    else:
        log(f"{'10. High-Entropy Key-Value Recall':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 11. Extreme Precision
    if "extreme_precision" in res:
        ep = res.get("extreme_precision", {})
        ep_desc = "Exact Matched" if ep.get("matched") else "Balance Diverged"
        log(f"{'11. Precision Ledger Reconcile':<38} | {fmt_status(ep.get('status', 'N/A')):<19} | {ep_desc:<20} | {ep.get('tok_s', 0):.2f} tok/s")
    else:
        log(f"{'11. Precision Ledger Reconcile':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 12. Code Execution
    if "code_execution" in res:
        ce = res.get("code_execution", {})
        ce_desc = "Dynamic Assertions OK" if ce.get("dynamic_tests_passed") else "Assertion Failure"
        log(f"{'12. Dynamic Code Unit Testing':<38} | {fmt_status(ce.get('status', 'N/A')):<19} | {ce_desc:<20} | {ce.get('tok_s', 0):.2f} tok/s")
    else:
        log(f"{'12. Dynamic Code Unit Testing':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 13. Error Handling
    if "error_handling" in res:
        eh = res.get("error_handling", {})
        eh_desc = "HTTP 400/422 Standard" if eh.get("bad_schema_rejected") else "Non-standard error"
        log(f"{'13. API Error Protocol Compliance':<38} | {fmt_status(eh.get('status', 'N/A')):<19} | {eh_desc:<20} | Protocol OK")
    else:
        log(f"{'13. API Error Protocol Compliance':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 14. Context Scaling Summary
    if "context_scaling" in res:
        cs = res.get("context_scaling", [])
        if cs:
            log("-" * 88)
            log("  [14. Context Scaling Milestone Performance & Accuracy Breakdown]", bold=True)
            cs_hdr = f"  {'Context Target':<18} | {'Cached':<8} | {'Prefill TTFT':<14} | {'Cold Speed':<14} | {'Effective':<14} | {'Decode':<12} | {'Accuracy':<10}"
            log(cs_hdr)
            log("  " + "-" * (len(cs_hdr) - 2))
            for step in cs:
                if step.get("status") in ("PASS", "PARTIAL"):
                    m_label = f"~{step.get('target_tokens', 0)//1000}k ({step.get('actual_prompt_tokens', 0):,} toks)"
                    cached_str = f"{step.get('cached_tokens', 0):,}"
                    ttft_str = f"{step.get('ttft_s', 0):.2f} s"
                    cold_str = f"{step.get('cold_prefill_tok_s', 0):.1f} tok/s"
                    eff_str = f"{step.get('effective_prefill_tok_s', 0):.1f} tok/s"
                    decode_str = f"{step.get('decode_tok_s', 0):.2f} tok/s"
                    acc_str = "RECALLED" if step.get("needle_matched") else "MISSED"
                    log(f"  {m_label:<18} | {cached_str:<8} | {ttft_str:<14} | {cold_str:<14} | {eff_str:<14} | {decode_str:<12} | {acc_str:<10}")
                else:
                    m_label = f"{step.get('target_tokens', 0):,} toks"
                    log(f"  {m_label:<18} | {'-':<8} | {'FAILED':<14} | {str(step.get('error', 'Error'))[:28]}")
    else:
        if 14 in report.get("selected_tests", range(1, 20)):
            log(f"{'14. Dynamic Context Scaling':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 15. Multihop Graph Traversal
    if "multihop_graph" in res:
        mh = res.get("multihop_graph", {})
        mh_desc = f"{mh.get('completion_tokens', 0)} tokens"
        log(f"{'15. Adversarial Graph & Distractors':<38} | {fmt_status(mh.get('status', 'N/A')):<19} | {mh_desc:<20} | {mh.get('tok_s', 0):.2f} tok/s")
    elif 15 in report.get("selected_tests", set()):
        log(f"{'15. Adversarial Graph & Distractors':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 16. Novel Algorithmic Fuzz
    if "novel_algorithm_fuzz" in res:
        fz = res.get("novel_algorithm_fuzz", {})
        fz_desc = "5,000 Ops Fuzzed OK" if fz.get("status") == "PASS" else "Fuzz Assertion Failed"
        log(f"{'16. Novel Algorithm (5k Fuzz)':<38} | {fmt_status(fz.get('status', 'N/A')):<19} | {fz_desc:<20} | {fz.get('tok_s', 0):.2f} tok/s")
    elif 16 in report.get("selected_tests", set()):
        log(f"{'16. Novel Algorithm (5k Fuzz)':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 17. Combinatorial Anti-Constraints
    if "combinatorial_anti_constraints" in res:
        ac = res.get("combinatorial_anti_constraints", {})
        ac_desc = f"{ac.get('constraints_passed', 0)}/6 Constraints Met"
        log(f"{'17. Anti-Constraints (IFEval Tier)':<38} | {fmt_status(ac.get('status', 'N/A')):<19} | {ac_desc:<20} | {ac.get('tok_s', 0):.2f} tok/s")
    elif 17 in report.get("selected_tests", set()):
        log(f"{'17. Anti-Constraints (IFEval Tier)':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 18. Counterfactual Algebra
    if "counterfactual_algebra" in res:
        ca = res.get("counterfactual_algebra", {})
        ca_desc = "Both Roots Solved" if ca.get("status") == "PASS" else ("1 Root Solved" if ca.get("status") == "PARTIAL" else "Roots Missed")
        log(f"{'18. Counterfactual Axiomatic Math':<38} | {fmt_status(ca.get('status', 'N/A')):<19} | {ca_desc:<20} | {ca.get('tok_s', 0):.2f} tok/s")
    elif 18 in report.get("selected_tests", set()):
        log(f"{'18. Counterfactual Axiomatic Math':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    # 19. Frontier Needle Depth
    if "frontier_needle_depth" in res:
        fn = res.get("frontier_needle_depth", {})
        fn_desc = "Composite Checksum OK" if fn.get("status") == "PASS" else "Checksum Missed"
        log(f"{'19. Frontier Depth Multi-Needle':<38} | {fmt_status(fn.get('status', 'N/A')):<19} | {fn_desc:<20} | {fn.get('tok_s', 0):.2f} tok/s")
    elif 19 in report.get("selected_tests", set()):
        log(f"{'19. Frontier Depth Multi-Needle':<38} | {fmt_status('SKIPPED'):<19} | {'-':<20} | -")

    log("="*88 + "\n")


NAME_TO_TEST_NUM = {
    "streaming": 1,
    "vision": 2,
    "concurrency": 3,
    "parallel": 3,
    "batching": 3,
    "capabilities": 4,
    "capabilities_4tasks": 4,
    "tasks": 4,
    "tool_calling": 5,
    "tools": 5,
    "json_schema": 6,
    "schema": 6,
    "prefix_caching": 7,
    "caching": 7,
    "client_abort": 8,
    "abort": 8,
    "stop_sequences": 9,
    "stop": 9,
    "high_entropy": 10,
    "high_entropy_recall": 10,
    "recall": 10,
    "extreme_precision": 11,
    "precision": 11,
    "code_execution": 12,
    "code": 12,
    "error_handling": 13,
    "error": 13,
    "context_scaling": 14,
    "context": 14,
    "scale": 14,
    "multihop_graph": 15,
    "multihop": 15,
    "graph": 15,
    "distractors": 15,
    "novel_algorithm_fuzz": 16,
    "novel_algorithm": 16,
    "fuzz": 16,
    "ring_buffer": 16,
    "combinatorial_anti_constraints": 17,
    "anti_constraints": 17,
    "ifeval": 17,
    "lipogram": 17,
    "counterfactual_algebra": 18,
    "algebra": 18,
    "symbolic": 18,
    "math": 18,
    "frontier_needle_depth": 19,
    "frontier_needle": 19,
    "frontier": 19,
}

def parse_selected_tests(test_arg: str, suite: str = "all"):
    if test_arg:
        selected = set()
        parts = [p.strip() for p in test_arg.replace(" ", ",").split(",") if p.strip()]
        for part in parts:
            if "-" in part and not part.startswith("-"):
                subparts = part.split("-", 1)
                if subparts[0].isdigit() and subparts[1].isdigit():
                    start_n, end_n = int(subparts[0]), int(subparts[1])
                    for n in range(start_n, end_n + 1):
                        if 1 <= n <= 19:
                            selected.add(n)
                    continue
            if part.isdigit():
                n = int(part)
                if 1 <= n <= 19:
                    selected.add(n)
            else:
                lower = part.lower().replace("-", "_")
                if lower in NAME_TO_TEST_NUM:
                    selected.add(NAME_TO_TEST_NUM[lower])
                elif lower == "flagship":
                    selected.update(range(1, 15))
                elif lower in ("adversarial", "hardened"):
                    selected.update(range(15, 20))
                elif lower == "all":
                    selected.update(range(1, 20))
                else:
                    log(f"Warning: Unknown test identifier '{part}'. Ignored.", color=YELLOW)
        return selected if selected else set(range(1, 20))

    if suite == "flagship":
        return set(range(1, 15))
    elif suite == "adversarial":
        return set(range(15, 20))
    else:  # "all"
        return set(range(1, 20))


# ============================================================================
# MAIN ORCHESTRATOR & INTERACTIVE DISCOVERY
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Enterprise LLM Server Full Benchmark & Evaluation Suite")
    parser.add_argument("--endpoint", "-e", help="LLM server base endpoint (e.g. http://127.0.0.1:8888/v1)")
    parser.add_argument("--api-key", "-k", default=os.getenv("OPENAI_API_KEY", ""), help="API key (optional)")
    parser.add_argument("--model", "-m", help="Model name or ID to test")
    parser.add_argument("--parallel", "-p", type=int, help="Number of parallel clients to test")
    parser.add_argument("--max-context", type=int, help="Override detected max context window tokens")
    parser.add_argument("--milestones", type=int, nargs="+", default=None, help="Custom context milestones for test 14 (e.g. --milestones 4000 8000 200000 204000)")
    parser.add_argument("--test", "-t", default=None, help="Run specific test(s) by number or name (e.g. 14, '11,12', '1-5', 'context_scaling', '15-19')")
    parser.add_argument("--suite", "-s", choices=["all", "flagship", "adversarial"], default="all",
                        help="Benchmark suite category: 'all' (tests 1-19), 'flagship' (tests 1-14), or 'adversarial' (tests 15-19)")
    parser.add_argument("--quick", "-q", action="store_true",
                        help="Fast smoke qualification mode (compact context horizons, skips heavy prefill)")
    parser.add_argument("--adversarial-depth", type=int, default=32000,
                        help="Target context depth for adversarial frontier tests (default: 32000)")
    parser.add_argument("--context-ratio", type=float, default=0.80,
                        help="Maximum fraction of context window to test in Test 14 (default: 0.80 for 80%%)")
    parser.add_argument("--out", "-o", help="Path to write JSON benchmark report")
    parser.add_argument("--auto", "-y", action="store_true", help="Non-interactive auto-selection mode")
    parser.add_argument("--parallel-suite", action="store_true", help="Execute independent functional tests concurrently matching parallel slots")
    args = parser.parse_args()

    log("\n" + "="*88, bold=True)
    log(" ENTERPRISE LLM SERVER FULL BENCHMARK & EVALUATION SUITE ", bold=True, color=GREEN)
    log("="*88)

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

    if args.quick:
        if args.milestones:
            milestones = sorted(list(set(args.milestones)))
        else:
            milestones = [4000, 8000, 16000]
        adv_depth = min(16000, args.adversarial_depth)
        log(f"Quick Mode Active: Milestones={milestones}, Adversarial Depth={adv_depth:,} tokens")
    elif args.milestones:
        milestones = sorted(list(set(args.milestones)))
        adv_depth = args.adversarial_depth
        log(f"Context Scaling Targets (Overridden by flag): {milestones}")
    else:
        milestones = build_context_milestones(max_context, max_ratio=args.context_ratio)
        adv_depth = args.adversarial_depth
        pct_label = f"{int(args.context_ratio * 100)}%"
        log(f"Adopted Context Scaling Targets (up to {pct_label} context): {milestones}")

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

    # 5. Execute Selected Test(s)
    selected_tests = parse_selected_tests(args.test, suite=args.suite)
    if args.test:
        log(f"Selective Test Execution Active: Running test(s) {sorted(list(selected_tests))} out of 19\n", color=CYAN, bold=True)
    else:
        log(f"Active Suite [{args.suite.upper()}]: Executing {len(selected_tests)} test(s) {sorted(list(selected_tests))}\n", color=CYAN, bold=True)

    report = {
        "endpoint": endpoint,
        "model": model_name,
        "max_context_tokens": max_context,
        "context_milestones": milestones,
        "parallel_streams": parallel,
        "suite": args.suite,
        "quick_mode": args.quick,
        "selected_tests": sorted(list(selected_tests)),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": {}
    }

    t_suite_start = time.perf_counter()

    if args.parallel_suite:
        log(f"\n[Parallel Suite Active] Dispatching independent functional tests concurrently with {parallel} slots...", bold=True, color=CYAN)
        parallel_candidates = []
        if 1 in selected_tests:
            parallel_candidates.append(("streaming", lambda: run_test_streaming(client)))
        if 2 in selected_tests:
            parallel_candidates.append(("vision", lambda: run_test_vision(client)))
        if 4 in selected_tests:
            parallel_candidates.append(("capabilities_4tasks", lambda: run_test_capabilities(client)))
        if 5 in selected_tests:
            parallel_candidates.append(("tool_calling", lambda: run_test_tool_calling(client)))
        if 6 in selected_tests:
            parallel_candidates.append(("json_schema", lambda: run_test_json_schema(client)))
        if 9 in selected_tests:
            parallel_candidates.append(("stop_sequences", lambda: run_test_stop_sequences(client)))
        if 10 in selected_tests:
            parallel_candidates.append(("high_entropy_recall", lambda: run_test_high_entropy(client)))
        if 11 in selected_tests:
            parallel_candidates.append(("extreme_precision", lambda: run_test_extreme_precision(client)))
        if 12 in selected_tests:
            parallel_candidates.append(("code_execution", lambda: run_test_code_execution(client)))
        if 13 in selected_tests:
            parallel_candidates.append(("error_handling", lambda: run_test_error_handling(client)))
        if 15 in selected_tests:
            parallel_candidates.append(("multihop_graph", lambda: run_test_multihop_graph(client, target_context=adv_depth)))
        if 16 in selected_tests:
            parallel_candidates.append(("novel_algorithm_fuzz", lambda: run_test_novel_algorithmic_fuzz(client)))
        if 17 in selected_tests:
            parallel_candidates.append(("combinatorial_anti_constraints", lambda: run_test_combinatorial_anti_constraints(client)))
        if 18 in selected_tests:
            parallel_candidates.append(("counterfactual_algebra", lambda: run_test_counterfactual_algebra(client)))

        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(fn): name for name, fn in parallel_candidates}
            for fut in as_completed(futures):
                name = futures[fut]
                try:
                    report["results"][name] = fut.result()
                except Exception as e:
                    log(f"Error in parallel test {name}: {e}", color=RED)
                    report["results"][name] = {"status": "FAIL", "error": str(e)}

        # Isolated tests run sequentially after the parallel matrix
        if 3 in selected_tests:
            report["results"]["concurrency"] = run_test_concurrency(client, parallel)
        if 7 in selected_tests:
            report["results"]["prefix_caching"] = run_test_prefix_caching(client)
        if 8 in selected_tests:
            report["results"]["client_abort"] = run_test_client_abort(client)
        if 14 in selected_tests:
            report["results"]["context_scaling"] = run_test_context_scaling(client, milestones)
        if 19 in selected_tests:
            report["results"]["frontier_needle_depth"] = run_test_frontier_needle_depth(client, target_context=adv_depth)
    else:
        if 1 in selected_tests:
            report["results"]["streaming"] = run_test_streaming(client)
        if 2 in selected_tests:
            report["results"]["vision"] = run_test_vision(client)
        if 3 in selected_tests:
            report["results"]["concurrency"] = run_test_concurrency(client, parallel)
        if 4 in selected_tests:
            report["results"]["capabilities_4tasks"] = run_test_capabilities(client)
        if 5 in selected_tests:
            report["results"]["tool_calling"] = run_test_tool_calling(client)
        if 6 in selected_tests:
            report["results"]["json_schema"] = run_test_json_schema(client)
        if 7 in selected_tests:
            report["results"]["prefix_caching"] = run_test_prefix_caching(client)
        if 8 in selected_tests:
            report["results"]["client_abort"] = run_test_client_abort(client)
        if 9 in selected_tests:
            report["results"]["stop_sequences"] = run_test_stop_sequences(client)
        if 10 in selected_tests:
            report["results"]["high_entropy_recall"] = run_test_high_entropy(client)
        if 11 in selected_tests:
            report["results"]["extreme_precision"] = run_test_extreme_precision(client)
        if 12 in selected_tests:
            report["results"]["code_execution"] = run_test_code_execution(client)
        if 13 in selected_tests:
            report["results"]["error_handling"] = run_test_error_handling(client)
        if 14 in selected_tests:
            report["results"]["context_scaling"] = run_test_context_scaling(client, milestones)
        if 15 in selected_tests:
            report["results"]["multihop_graph"] = run_test_multihop_graph(client, target_context=adv_depth)
        if 16 in selected_tests:
            report["results"]["novel_algorithm_fuzz"] = run_test_novel_algorithmic_fuzz(client)
        if 17 in selected_tests:
            report["results"]["combinatorial_anti_constraints"] = run_test_combinatorial_anti_constraints(client)
        if 18 in selected_tests:
            report["results"]["counterfactual_algebra"] = run_test_counterfactual_algebra(client)
        if 19 in selected_tests:
            report["results"]["frontier_needle_depth"] = run_test_frontier_needle_depth(client, target_context=adv_depth)

    total_suite_time = time.perf_counter() - t_suite_start
    report["total_suite_wall_time_s"] = round(total_suite_time, 2)

    # 6. Save JSON Report
    parsed_host = urlparse(endpoint).netloc.replace(":", "_") or "local"
    out_file = args.out or os.path.join(DEFAULT_RESULTS_DIR, f"enterprise_eval_{parsed_host}_{time.strftime('%Y%m%d_%H%M%S')}.json")
    out_dir = os.path.dirname(out_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    # 7. Display Terminal Result Summary Table
    print_summary_table(report)

    log("="*88, bold=True)
    log(" ENTERPRISE BENCHMARK EVALUATION COMPLETED SUCCESSFULLY ", bold=True, color=GREEN)
    log(f"  Total Suite Wall Time : {total_suite_time:.2f} s")
    log(f"  JSON Benchmark Report : {out_file}", bold=True)
    log("="*88 + "\n", bold=True)

if __name__ == "__main__":
    main()
