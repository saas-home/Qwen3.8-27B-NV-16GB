# AI Agent Instructions for Qwen3.8-27B One-Click Install Kit

## Project Overview

This is a serving kit for **Qwen/Qwen3.8-27B** in turboderp's EXL3 quants, designed to run on one consumer NVIDIA GPU (12–32 GB VRAM). It picks a quant that fits the card, installs its own Python environment, downloads weights, serves an **OpenAI-compatible** endpoint, and opens a chat UI. Windows and Linux have identical behavior through a unified `simplex` CLI.

## Key Architecture

- **Unified CLI**: `tools/cli.py` implements the `simplex` command used on both systems
- **Setup flow**: `tools/setup_core.py` contains the setup pipeline; front ends are `linux/setup.sh` and `windows/START-HERE.bat` (web UI)
- **Start flow**: `linux/start.sh` and `windows/start.bat` load models and serve
- **Profile picker**: `tools/profiles.py` detects GPU VRAM and selects the best quant/context/KV-cache combination
- **Server**: `tools/serve_openai.py` serves the OpenAI-compatible API on `/v1/chat/completions`
- **Chat UI**: [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness) (`dsh`) runs as a second Node process on port 3080
- **Engine**: ExLlamaV3 v1.4.4, prebuilt wheels when available; source build fallback

## Directory Structure

```
.
├── linux/              # Linux scripts: setup.sh, start.sh, stop.sh, simplex
├── windows/            # Windows scripts: START-HERE.bat, start.bat, stop.bat, simplex.bat
├── tools/              # Core Python implementation
│   ├── cli.py          # simplex CLI (shared across OS)
│   ├── profiles.py     # GPU VRAM profile picker
│   ├── setup_core.py   # Setup pipeline steps
│   ├── setup_web.py    # Web setup front-end
│   ├── serve_openai.py # OpenAI-compatible server
│   ├── dsh.py          # DeepSeek Harness launcher/configurer
│   ├── downloader.py   # Resumable weight downloads
│   └── wheels.py       # PyTorch/engine wheel resolver
├── models/             # Downloaded EXL3 quant models (gitignored)
├── .env.example        # Configuration template
├── wheels/             # Prebuilt wheel documentation
└── README.md           # Full user documentation
```

## Common Commands

### Running the kit
```bash
# Linux
./linux/setup.sh                # install environment and fetch a model
./linux/start.sh                # load a model and serve
./linux/start.sh -b             # run in background
./linux/stop.sh                 # stop both server and UI

# Windows
windows\START-HERE.bat          # install, then start
windows\start.bat               # start an installed model
windows\stop.bat                # stop both

# Unified CLI (from kit root)
./linux/simplex setup           # install the environment and fetch a model
./linux/simplex start           # load a model and serve it
./linux/simplex start --no-harness  # serve /v1 only
./linux/simplex start -b        # run in background
./linux/simplex stop            # stop both
./linux/simplex status          # what is running
./linux/simplex logs -f         # follow the launcher log
./linux/simplex models          # what is on disk
./linux/simplex doctor          # check this machine
```

## Configuration

- `.env` is created from `.env.example` on first run
- Key knobs: `MODEL_DIR`, `PROFILE`, `CONTEXT_SIZE`, `CACHE_QUANT`, `GPU_MEM_GB`, `VISION`, `DRAFT`, `PORT`, `HOST`, `UI`
- `.env` is **data, never shell** — parse it as key=value pairs, never `source` or `exec` it
- CRLF endings are tolerated; strip `\r` when reading

## Profile Selection (`tools/profiles.py`)

The profile picker:
1. Reads GPU VRAM via `nvidia-smi`
2. Computes what fits under budget: `VRAM - max(1.3 GB, 8%)`
3. Uses memory model: `need = weights + kv_per_token * context * 17/16 + vision_tower + 2.6 overhead`
4. Offers choices (best quality vs longest context)
5. Writes choice into `.env` (`PROFILE`, `MODEL_DIR`, `HF_TARGET_REPO`, `HF_REVISION`, `MODEL_ID`, `CONTEXT_SIZE`, `CACHE_QUANT`, `GPU_MEM_GB`, `VISION`)

KV cache is **int4** on every profile. The 2.0 bpw baseline uses `CACHE_QUANT=8,4` (int8 K / int4 V).

## Important Pitfalls

1. **Never `source .env`** — it contains values with spaces (e.g., GPU names) and Windows CRLF endings. Parse as key=value.
2. **Windows VRAM preflight** — Windows silently spills to system RAM when VRAM is insufficient, causing extremely slow performance. The preflight check is critical on Windows.
3. **No free-VRAM preflight on Linux** — Linux fails with `Insufficient VRAM in split for model and cache`; lower `CONTEXT_SIZE` or `GPU_MEM_GB`.
4. **Prebuilt wheels** — The engine arrives as prebuilt wheels (cu128/cu132). Source compilation is the fallback for platforms no wheel covers (e.g., aarch64/GB10).
5. **Node requirement** — DeepSeek Harness requires Node 22.19+. Without Node, `/v1` still serves but the chat UI fails.
6. **Python 3.11+ required** — 64-bit only. The kit installs its own `.venv`.
7. **Model downloads are resumable** — closing the window or losing connection costs nothing.

## Testing and Verification

- `./linux/simplex doctor` — checks Python, venv, engine version, driver, card, Node, ports, `.env` values, weight completeness, free disk
- `./linux/simplex status --json` — machine-readable status for scripts
- `python tools/profiles.py --list --vram 16` — simulate profile choices for a given VRAM
- Check `logs/simplex-*.log` for crash transcripts

## Conventions

- **Cross-platform parity**: `simplex` CLI behavior must be identical on Windows and Linux
- **No system modifications**: Everything stays inside `.venv/`, `models/`, `logs/`, `apps/`, `.dsh/`
- **Callbacks over prints**: `setup_core.py` functions report through callbacks; no printing, exiting, or blocking on input
- **Resumable downloads**: `downloader.py` tracks progress and resumes interrupted downloads
- **Configuration-driven**: `tools/dsh.py` queries the running server for actual model capabilities and configures the harness accordingly

## Related Documentation

- [README.md](README.md) — Complete user guide for Windows and Linux
- [Qwen3.8-27B-NV-16GB-Optimization-Whitepaper.md](Qwen3.8-27B-NV-16GB-Optimization-Whitepaper.md) — Technical whitepaper on 204.8k context optimization & empirical benchmarks
- [.env.example](.env.example) — Configuration template with detailed comments
- [tools/wheels.py](tools/wheels.py) — PyTorch and engine wheel resolution
