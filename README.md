# Qwen3.8-27B for 16GB NVIDIA GPUs (Extreme 204.8k Context & Continuous Batching)

> **Repository:** [`saas-home/Qwen3.8-27B-NV-16GB`](https://github.com/saas-home/Qwen3.8-27B-NV-16GB)  
> **Upstream Project:** Forked from [`MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install`](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install) by [Mia's AI Lab](https://x.com/MiaAI_lab).  
> *(For Windows 1-click desktop guides, general multi-GPU setups, and upstream release archives, please refer directly to the [upstream repository](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install).)*

---

## Overview

This repository is a production-hardened, high-performance serving kit for **[Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)** in turboderp's EXL3 format, optimized specifically for **Ubuntu Linux** (tested on **Ubuntu 26.04 / 24.04**) and consumer **16 GB NVIDIA GPUs** (benchmarked on the **NVIDIA GeForce RTX 4070 Ti SUPER 16 GB**).

While standard desktop baselines truncate context to ~117k tokens and serialize multi-client requests, this fork unlocks an unprecedented **204,800 token context window (~800 pages)**, **2-slot concurrent continuous batching**, and an **automated 14-stage enterprise qualification suite** on a single 16 GB graphics card.

---

## Key Benefits: What's New in This Fork

1. **Full 204,800 Token Context Ceiling on 16GB GPU**:
   - Ingests and reasons over entire codebases and massive documentation repositories on a single 16 GB card using 3.0 bpw EXL3 weights.
   - Empirically certified up to **203,852 tokens (99.54% of hardware ceiling)** with **zero memory fragmentation crashes or OOMs**.
2. **Asymmetric KV Cache Quantization (`CACHE_QUANT=4,3`)**:
   - Quantizes Keys to 4-bit (preserving attentional steering) and Values to 3-bit.
   - Slashes VRAM consumption by **600 MiB** at 200k context, recovering **>1 GB of critical safety headroom** on 16 GB cards.
3. **2-Slot Parallel Continuous Batching (`PARALLEL=2`)**:
   - Replaced serial locks with an asynchronous `BatchWorker` dispatcher and per-job queues.
   - Allows multiple coding agents (e.g. Claude Code, Cline, Cursor, Aider) to stream responses simultaneously on GPU at **~70.25 tok/s aggregate throughput (+58% gain)** instead of serial blocking.
4. **Dynamic Context Headroom Clamping**:
   - Eliminates ExLlamaV3 `AssertionError: Job requires X pages (only Y available)` when deep conversational turns receive large token requests, dynamically clamping generation tokens to available cache headroom.
5. **Zero-VRAM Multimodal Vision (`EXL3_VISION_PINNED=1`)**:
   - Pins the vision tower in host DDR5 RAM, saving ~0.87 GB of VRAM directly for the text KV cache while maintaining fast 1.06s invoice/document parsing.
6. **24 GB DDR5 Host Prompt Cache (`CPU_CACHE_GB=24`)**:
   - Yields **sub-second TTFT (0.91s at 200k context)** on iterative conversational turns with shared prompt prefixes.
7. **AMD Ryzen 9 7950X3D CPU Affinity (`AFFINITY=0-7,16-23`)**:
   - Pins worker execution to CCD0 3D V-Cache, eliminating cross-CCD bus latency penalties.
8. **14-Stage Enterprise Qualification Suite (`tests/scripts/llm_server_full_test.py`)**:
   - Automated end-to-end certification across Streaming, Vision, Concurrency, 4-Task SWE Architecture, Tool Calling, Structured JSON Schema, Prefix Caching, Socket Abort Recovery, Determinism, Needle Recall, 25-step Precision Ledger, Dynamic Code Unit Testing, and Context Scaling up to 204.8k.

---

## Target Hardware & Environment

- **Operating System:** Ubuntu Linux (tested on **Ubuntu 26.04 LTS / 24.04 LTS Server & Desktop**)
- **GPU:** **NVIDIA GeForce RTX 4070 Ti SUPER (16 GB GDDR6X)** or any Turing+ 16 GB NVIDIA GPU
- **NVIDIA Driver:** 570 or newer (default build targets CUDA 12.8 / 13.2)
- **Host Memory:** 32 GB – 64 GB DDR5 (64 GB recommended for 24 GB host prompt cache)
- **Python:** 64-bit Python 3.11 – 3.14 (the kit creates and manages its own isolated `.venv/`)
- **Node.js (Optional):** Node 22.19+ (only required if you want to use the DeepSeek Harness chat UI)

---

## Step-by-Step: Configure and Start ExLlamaV3 Server

### Step 1: Clone the Repository
```bash
git clone https://github.com/saas-home/Qwen3.8-27B-NV-16GB.git
cd Qwen3.8-27B-NV-16GB
```

### Step 2: Run Automated Setup
Run the Linux setup script. It inspects your GPU, verifies driver compatibility, creates the isolated `.venv`, installs PyTorch cu128 and ExLlamaV3 v1.5.0, and downloads the 3.0 bpw model weights into `models/`:
```bash
./linux/setup.sh
```
*(Alternatively, use the unified CLI: `./linux/simplex setup`)*

### Step 3: Configure `.env` for 16GB GPU (RTX 4070 Ti SUPER)
The setup script generates a `.env` configuration. Ensure the following production-optimized settings are present:

```ini
# Model Directory & Context Window
MODEL_DIR=models/turboderp_Qwen3.8-27B-exl3_3.00bpw
CONTEXT_SIZE=204800

# Asymmetric KV Cache (4-bit Keys / 3-bit Values — saves 600 MiB VRAM)
CACHE_QUANT=4,3

# 2-Slot Parallel Continuous Batching
PARALLEL=2

# Headless 16 GB VRAM Allocation & Zero-VRAM Vision Offload
GPU_MEM_GB=15.7
EXL3_VISION_PINNED=1
CPU_CACHE_GB=24

# Generation Budget & Multi-Turn Reasoning Hygiene
MAX_TOKENS=16384
REASONING_EFFORT=medium
NO_REASONING_PRESERVE=1

# Networking (OpenAI-compatible API)
PORT=8888
HOST=0.0.0.0

# Chat UI (browser, server, or no)
UI=browser
SIMPLEX_HARNESS_PORT=3080
```

### Step 4: Start the Server
Start the OpenAI-compatible server:

```bash
# Foreground execution (displays real-time startup & generation logs)
./linux/start.sh

# Or start in background mode (daemon)
./linux/start.sh -b

# Or start without opening the browser chat UI (serve API only)
./linux/simplex start --no-harness
```

To stop the background server at any time:
```bash
./linux/stop.sh
```

---

## Connecting Clients & Coding Agents

The server exposes a standard **OpenAI-compatible `/v1` endpoint**:

- **API Base URL:** `http://127.0.0.1:8888/v1`
- **Model Name:** `qwen3.8-27b-exl3-3.0bpw` (or `Qwen3.8-27B`)
- **API Key:** `local` (or any string; authentication is disabled by default)
- **DeepSeek Harness Web UI:** `http://127.0.0.1:3080/`

### Claude Code / Cline / Cursor / Aider Configuration
In your client or agent settings:
```json
{
  "api_base": "http://127.0.0.1:8888/v1",
  "model": "qwen3.8-27b-exl3-3.0bpw",
  "api_key": "local"
}
```

---

## Automated Enterprise Benchmark Suite

This repository includes a production evaluation test harness in [`tests/scripts/`](tests/scripts):

```bash
# Activate the environment
source .venv/bin/activate

# 1. Run the full 14-stage qualification suite non-interactively
python3 tests/scripts/llm_server_full_test.py --endpoint http://127.0.0.1:8888/v1 --auto

# 2. Run only Context Scaling benchmark (Test 14)
python3 tests/scripts/llm_server_full_test.py --test 14

# 3. Test specific context milestones (e.g. verify 200k and 204k ceiling directly)
python3 tests/scripts/llm_server_full_test.py --test 14 --milestones 200000 203800

# 4. Run precision ledger and dynamic code execution tests
python3 tests/scripts/llm_server_full_test.py --test precision,code
```

### Empirical 204K Context Scaling Benchmark (RTX 4070 Ti SUPER 16 GB)

Empirical cold-prefill performance (prompt salting enabled to isolate hardware throughput from cache hits):

| Context Milestone | Ingested Prompt | Cold Prefill TTFT | Cold Prefill Rate | Sustained Decode Speed | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **~4k** | 4,047 toks | 2.89 s | **1,401.5 tok/s** | **44.47 tok/s** | **PASS** |
| **~8k** | 8,055 toks | 5.57 s | **1,447.4 tok/s** | **44.63 tok/s** | **PASS** |
| **~16k** | 16,064 toks | 11.50 s | **1,396.7 tok/s** | **43.59 tok/s** | **PASS** |
| **~32k** | 32,035 toks | 25.12 s | **1,275.5 tok/s** | **41.84 tok/s** | **PASS** |
| **~64k** | 64,037 toks | 60.45 s (1.0 min) | **1,059.3 tok/s** | **38.62 tok/s** | **PASS** |
| **~128k** | 128,074 toks | 162.38 s (2.7 min) | **788.7 tok/s** | **33.51 tok/s** | **PASS** |
| **~200k** | 200,067 toks | 327.56 s (5.5 min) | **610.8 tok/s** | **29.30 tok/s** | **PASS** |
| **~204k (Hardware Cap)** | **203,852 toks** | **337.69 s (5.6 min)** | **603.7 tok/s** | **29.12 tok/s** | **PASS** |

---

## Technical Documentation & In-Depth Whitepaper

For in-depth mathematical derivations, asymmetric quantization accuracy proofs across 16-hop confusable pointer chasing, and head-to-head empirical evaluations against `llama.cpp` (`llama-server`), see:

📄 **[Qwen3.8-27B-NV-16GB-Optimization-Whitepaper.md](Qwen3.8-27B-NV-16GB-Optimization-Whitepaper.md)**

---

## Upstream & Acknowledgements

- **Base Project:** [`MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install`](https://github.com/MiaAI-Lab/Qwen3.8-27B-16gb-NVIDIA-GPUs-one-click-install) by [Mia's AI Lab](https://x.com/MiaAI_lab) (Windows/Linux setup scripts, harness launcher, and unified CLI architecture).
- **Inference Engine:** [ExLlamaV3](https://github.com/turboderp-org/exllamav3) by [turboderp](https://huggingface.co/turboderp).
- **Model:** [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by the Qwen Team, Alibaba Cloud.
- **Chat UI:** [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`) by DeepSeek AI.

## License

- Base model weights and architecture: **Apache-2.0**
- Serving kit scripts and optimizations: **MIT**
