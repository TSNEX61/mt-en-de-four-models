"""LLaMA 3.2 1B Instruct (decoder-only): 3-shot in-context prompting (no weight updates).

Runs through a local Ollama server (llama3.2:1b, GGUF Q8_0) because the bf16 Hugging Face model was
too slow on a laptop CPU (~14 s/sentence). Same prompt, same test subset and greedy decoding as the
other models.  Setup:  ollama pull llama3.2:1b   (Ollama: https://ollama.com)
"""
import argparse
import json
import random
import urllib.request

import llm_utils
from common import (CKPT, SEED, TEMP_SUBSET, Timer, clean_llm_output, load_split, save_outputs, save_result,
                    score, set_seed, temperature_sweep, ethics_probe)

MODEL_ID = "llama3.2:1b"  # Ollama build of meta-llama/Llama-3.2-1B-Instruct
OLLAMA = "http://localhost:11434"


def call(path, payload):
    req = urllib.request.Request(OLLAMA + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def translate(srcs, temperature=0.0, shots=(), cache=None):
    """cache: optional jsonl path; finished sentences are appended so an interrupted run can resume."""
    done = [json.loads(line)["hyp"] for line in open(cache, encoding="utf-8")] if cache and cache.exists() else []
    hyps = done[: len(srcs)]
    for s in srcs[len(hyps):]:
        out = call("/api/chat", {"model": MODEL_ID, "stream": False,
                                 "messages": llm_utils.build_messages(s, shots),
                                 # top_k=0 / top_p=1 disable Ollama's default truncation (40 / 0.9) so that
                                 # sampling matches the pure temperature sampling used for the other models
                                 "options": {"temperature": temperature, "seed": SEED, "num_predict": 64,
                                             "top_k": 0, "top_p": 1.0}})
        hyps.append(clean_llm_output(out["message"]["content"]))
        if cache:
            with open(cache, "a", encoding="utf-8") as f:
                f.write(json.dumps({"hyp": hyps[-1]}, ensure_ascii=False) + "\n")
        print(f"{len(hyps)}/{len(srcs)}", flush=True)
    return hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=3)
    ap.add_argument("--temp_subset", type=int, default=TEMP_SUBSET, help="sentences for the temperature sweep")
    ap.add_argument("--sweep_only", action="store_true", help="only (re)run the temperature sweep")
    args = ap.parse_args()
    set_seed()

    info = call("/api/show", {"model": MODEL_ID})
    total_params = info["model_info"]["general.parameter_count"]
    test = load_split("test_subset")
    srcs, refs = [r["en"] for r in test], [r["de"] for r in test]
    # few-shot demonstrations: fixed random training pairs of typical length
    pool = [r for r in load_split("train") if 8 <= len(r["en"].split()) <= 14]
    shots = random.Random(SEED).sample(pool, args.shots)
    n = min(args.temp_subset, TEMP_SUBSET)
    if args.sweep_only:
        temperature_sweep("llama", lambda s, t: translate(s, t, shots=shots), srcs[:n], refs[:n])
        return

    with Timer() as tm:
        hyps = translate(srcs, shots=shots, cache=CKPT / "llama_test_partial.jsonl")
    save_outputs("llama", srcs, refs, hyps)
    save_result("llama", {
        "model": f"{MODEL_ID} (Ollama, {info['details']['quantization_level']})", "architecture": "decoder-only",
        "adaptation": f"{args.shots}-shot in-context prompting", "params_total": total_params,
        "params_trainable": 0, "train_seconds": 0, "infer_seconds": round(tm.seconds, 1),
        "infer_sent_per_s": round(len(srcs) / tm.seconds, 2), **score(hyps, refs), "shots": shots})

    ethics_probe("llama", lambda s: translate(s, shots=shots))
    temperature_sweep("llama", lambda s, t: translate(s, t, shots=shots), srcs[:n], refs[:n])


if __name__ == "__main__":
    main()
