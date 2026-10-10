#!/usr/bin/env python3
"""
Drafting-mode benchmark for this kit's model.

Compares three speculative-decoding setups on the same prompt:
  none   - no drafting (baseline)
  mtp    - MTP head inside the checkpoint (default kit mode)
  ngram  - n-gram drafting, analogous to llama.cpp SPEC_TYPE=ngram-mod
           with SPEC_DRAFT_N_MIN=2 / SPEC_DRAFT_N_MAX=4
           (ngram_match_min=2, num_draft_tokens=4)

Usage:
  .venv/bin/python tools/ngram_bench.py --mode none|mtp|ngram
"""
import argparse
import sys
import time

MODEL = "models/Qwen3.8-27B-EXL3-3.0bpw"
N_TOKENS = 300

# Templated/code-style prompt: n-gram drafting does best on repetitive structure.
PROMPT = (
    "Write a Python class implementing a binary search tree with insert, "
    "delete, and search methods. Include a docstring for each method, "
    "type hints, and a main block that exercises all three methods with "
    "example values and prints the results."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["none", "mtp", "ngram"], required=True)
    ap.add_argument("--cache-size", type=int, default=32768)
    ap.add_argument("--ngram-min", type=int, default=2)
    ap.add_argument("--ndt", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=N_TOKENS)
    ap.add_argument("--prompt", type=str, default=PROMPT)
    args = ap.parse_args()

    from exllamav3 import model_init, Generator, Job
    from exllamav3.generator.sampler.presets import ComboSampler

    parser = argparse.ArgumentParser()
    model_init.add_args(parser, add_draft_model_args=True)
    margs = parser.parse_args(["-m", MODEL])
    margs.cache_size = args.cache_size
    margs.cache_quant = "4,3"
    margs.mtp = args.mode == "mtp"
    margs.draft_model_dir = None
    if args.mode == "ngram":
        margs.ngram_match_min = args.ngram_min
        margs.num_draft_tokens = args.ndt
    elif args.mode == "mtp":
        margs.num_draft_tokens = args.ndt

    t0 = time.perf_counter()
    if args.mode == "mtp":
        model, config, cache, tokenizer, draft_model, draft_config, draft_cache = \
            model_init.init(margs, progress=True)
        generator = Generator(
            model, cache, tokenizer,
            draft_model=draft_model, draft_cache=draft_cache,
            num_draft_tokens=args.ndt,
            max_batch_size=1,
            record_draft_stats=True,
        )
    else:
        model, config, cache, tokenizer, _, _, _ = model_init.init(margs, progress=True)
        gkwargs = dict(max_batch_size=1, record_draft_stats=True)
        if args.mode == "ngram":
            gkwargs["ngram_match_min"] = args.ngram_min
            gkwargs["num_draft_tokens"] = args.ndt
        generator = Generator(model, cache, tokenizer, **gkwargs)
    t_load = time.perf_counter() - t0

    if hasattr(tokenizer, "hf_chat_template"):
        input_ids = tokenizer.hf_chat_template(
            [{"role": "user", "content": args.prompt}],
            add_generation_prompt=True,
            enable_thinking=False,
        )
    else:
        input_ids = tokenizer.encode(args.prompt, add_bos=True)
    sampler = ComboSampler(temperature=0.0)
    job = Job(input_ids=input_ids, max_new_tokens=args.tokens,
              stop_conditions=[tokenizer.eos_token_id], sampler=sampler)

    generator.enqueue(job)
    t1 = time.perf_counter()
    text = ""
    while generator.num_remaining_jobs():
        res = generator.iterate()
        for r in res:
            if r.get("stage") == "streaming":
                text += r.get("text", "")
    t_gen = time.perf_counter() - t1

    # draft_stats: (position, window, accepted) per verification round
    stats = getattr(job, "draft_stats", []) or []
    if stats:
        windows = [w for _, w, _ in stats]
        accepted = [a for _, _, a in stats]
        acc_rate = sum(accepted) / max(1, sum(windows))
        tok_per_round = 1 + sum(accepted) / len(stats)
    else:
        acc_rate = tok_per_round = 0.0

    print(f"mode={args.mode} load={t_load:.1f}s "
          f"tokens={args.tokens} time={t_gen:.2f}s tok/s={args.tokens / t_gen:.1f} "
          f"verify_rounds={len(stats)} acc_rate={acc_rate:.3f} "
          f"tok/round={tok_per_round:.2f}")
    print("--- sample ---")
    print(text[:400])


if __name__ == "__main__":
    sys.exit(main())
