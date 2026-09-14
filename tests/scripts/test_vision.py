#!/usr/bin/env python3
import base64
import json
import time
import urllib.request
import sys
import os
import argparse

IMAGE_PATH = os.path.join(os.path.dirname(__file__), "image.png")

def main():
    parser = argparse.ArgumentParser(description="Test multimodal vision endpoint.")
    parser.add_argument("--url", default="http://127.0.0.1:8888/v1/chat/completions", help="Endpoint URL")
    parser.add_argument("--model", default="qwen3.8-27b-exl3-3.0bpw", help="Model name")
    parser.add_argument("--image", default=IMAGE_PATH, help="Path to test image")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "vision_result.json"), help="Output JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: {args.image} not found")
        sys.exit(1)

    with open(args.image, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please summarize this document image in detail, extracting key information such as invoice number, dates, parties involved, line items, rates, hours, and total amount."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.0,
        "stream": True
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(args.url, data=req_data, headers={"Content-Type": "application/json"})

    print(f"Sending multimodal vision request to {args.url} (model: {args.model})...")
    start_time = time.perf_counter()
    first_token_time = None
    response_text = []
    token_count = 0

    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line_str = line.decode("utf-8").strip()
            if not line_str or line_str == "data: [DONE]":
                continue
            if line_str.startswith("data: "):
                try:
                    data = json.loads(line_str[6:])
                    delta = data["choices"][0].get("delta", {}).get("content", "")
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        token_count += 1
                        response_text.append(delta)
                        print(delta, end="", flush=True)
                except Exception:
                    pass

    end_time = time.perf_counter()
    print("\n" + "="*60)
    ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else 0.0
    decode_duration = end_time - first_token_time if first_token_time else 0.0
    decode_speed = (token_count - 1) / decode_duration if decode_duration > 0 and token_count > 1 else 0.0

    print(f"Total time: {end_time - start_time:.2f}s")
    print(f"TTFT (Prefill + Vision Encoding): {ttft_ms:.2f} ms")
    print(f"Tokens Generated: {token_count}")
    print(f"Decode Speed: {decode_speed:.2f} tok/s")

    result = {
        "url": args.url,
        "model": args.model,
        "ttft_ms": round(ttft_ms, 2),
        "tokens": token_count,
        "decode_tok_s": round(decode_speed, 2),
        "total_time_s": round(end_time - start_time, 2),
        "response": "".join(response_text)
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved vision benchmark result to {args.out}")

if __name__ == "__main__":
    main()
