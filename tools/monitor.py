#!/usr/bin/env python3
"""
Live terminal monitor for Qwen3.8 / simplex serving metrics.
Usage:
    .venv/bin/python tools/monitor.py
"""
import time
import json
import urllib.request
import sys
import os

HEALTH_URL = "http://127.0.0.1:8888/health"

def fetch_health():
    try:
        req = urllib.request.Request(HEALTH_URL)
        with urllib.request.urlopen(req, timeout=1.5) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return {"error": str(e)}

def main():
    print("\033[?25l", end="")  # hide cursor
    last_prompt = 0
    last_completion = 0
    last_time = time.perf_counter()

    try:
        while True:
            now = time.perf_counter()
            dt = now - last_time
            last_time = now

            data = fetch_health()
            if "error" in data:
                sys.stdout.write(f"\r\033[KConnecting to {HEALTH_URL}... ({data['error']})")
                sys.stdout.flush()
                time.sleep(1.0)
                continue

            active = data.get("active_jobs", 0)
            gpu_active = data.get("gpu_active_jobs", active)
            gpu_pending = data.get("gpu_pending_jobs", 0)
            parallel = data.get("parallel", 2)
            busy = data.get("busy", False)
            p_total = data.get("prompt_tokens_total", 0)
            c_total = data.get("completion_tokens_total", 0)

            # Calculate delta tok/s
            c_diff = c_total - last_completion
            tok_per_sec = (c_diff / dt) if (dt > 0 and last_completion > 0) else 0.0
            if last_completion == 0:
                tok_per_sec = 0.0

            last_prompt = p_total
            last_completion = c_total

            status_str = f"\033[92mACTIVE\033[0m" if (active > 0) else "\033[90mIDLE\033[0m"
            slots_str = f"[{'#' * gpu_active}{'.' * (parallel - gpu_active)}]"

            # Build single-line or multi-line dashboard
            sys.stdout.write(
                f"\r\033[K"
                f"[{status_str}] "
                f"GPU Slots: {slots_str} ({gpu_active}/{parallel}) "
                f"| Queued: {gpu_pending} "
                f"| Live Speed: \033[1;36m{tok_per_sec:5.1f} tok/s\033[0m "
                f"| Completed: {c_total:,} toks "
                f"| Prompts: {p_total:,} toks"
            )
            sys.stdout.flush()
            time.sleep(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        print("\033[?25h")  # restore cursor

if __name__ == "__main__":
    main()
