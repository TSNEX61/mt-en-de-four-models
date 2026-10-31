"""Gemma 3 270M (decoder-only): zero-shot prompting baseline, then LoRA instruction fine-tuning."""
import argparse
import math
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import llm_utils
from common import (CKPT, TEMP_SUBSET, Timer, count_params, load_split, save_outputs, save_result,
                    score, set_seed, temperature_sweep, ethics_probe)

MODEL_ID = "unsloth/gemma-3-270m-it"  # ungated mirror of google/gemma-3-270m-it


def encode_example(tok, row, max_len=128):
    """Chat-formatted example; loss only on the assistant (German) tokens."""
    prompt = llm_utils.prompt_text(tok, row["en"])
    full = tok.apply_chat_template(llm_utils.build_messages(row["en"]) +
                                   [{"role": "assistant", "content": row["de"]}], tokenize=False)
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    f_ids = tok(full, add_special_tokens=False)["input_ids"][:max_len]
    labels = [-100] * len(p_ids) + f_ids[len(p_ids):]
    return f_ids, labels[: len(f_ids)]


def collate(tok, examples):
    width = max(len(ids) for ids, _ in examples)
    ids = [x + [tok.pad_token_id] * (width - len(x)) for x, _ in examples]
    lab = [y + [-100] * (width - len(y)) for _, y in examples]
    att = [[1] * len(x) + [0] * (width - len(x)) for x, _ in examples]
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(lab), "attention_mask": torch.tensor(att)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max_minutes", type=float, default=60)
    args = ap.parse_args()
    set_seed()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32, attn_implementation="eager")
    base_params, _ = count_params(model)
    test = load_split("test_subset")
    srcs, refs = [r["en"] for r in test], [r["de"] for r in test]

    # 1) zero-shot prompting baseline (no adaptation)
    with Timer() as tm:
        zs = llm_utils.translate(model, tok, srcs)
    save_outputs("gemma_zeroshot", srcs, refs, zs)
    save_result("gemma_zeroshot", {
        "model": MODEL_ID, "architecture": "decoder-only", "adaptation": "zero-shot prompting",
        "params_total": base_params, "params_trainable": 0, "train_seconds": 0,
        "infer_seconds": round(tm.seconds, 1), "infer_sent_per_s": round(len(srcs) / tm.seconds, 2),
        **score(zs, refs)})

    # 2) LoRA fine-tuning
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    total_params, trainable = count_params(model)
    model.print_trainable_parameters()

    tok.padding_side = "right"
    train = [encode_example(tok, r) for r in load_split("train")[: args.n_train]]
    steps_total = math.ceil(len(train) / args.batch) * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 30, max(0.0, (steps_total - s) / max(1, steps_total - 30))))

    history, step, t0, stop = [], 0, time.perf_counter(), False
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train)).tolist()
        for i in range(0, len(train), args.batch):
            loss = model(**collate(tok, [train[j] for j in perm[i: i + args.batch]])).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}/{steps_total} loss {loss.item():.4f} "
                      f"elapsed {time.perf_counter() - t0:.0f}s", flush=True)
                history.append({"step": step, "loss": round(loss.item(), 4)})
            if time.perf_counter() - t0 > args.max_minutes * 60:
                stop = True
                break
        if stop:
            print("time budget reached")
            break
    train_seconds = time.perf_counter() - t0
    model.save_pretrained(CKPT / "gemma_lora_adapter")

    with Timer() as tm:
        hyps = llm_utils.translate(model, tok, srcs)
    save_outputs("gemma", srcs, refs, hyps)
    save_result("gemma", {
        "model": MODEL_ID, "architecture": "decoder-only", "adaptation": f"LoRA fine-tuning (r={args.rank})",
        "params_total": base_params, "params_trainable": trainable, "params_with_adapter": total_params,
        "train_examples": min(len(train), step * args.batch), "steps": step,
        "train_seconds": round(train_seconds, 1), "infer_seconds": round(tm.seconds, 1),
        "infer_sent_per_s": round(len(srcs) / tm.seconds, 2), **score(hyps, refs),
        "history": history, "config": vars(args)})

    ethics_probe("gemma", lambda s: llm_utils.translate(model, tok, s))
    temperature_sweep("gemma", lambda s, t: llm_utils.translate(model, tok, s, t),
                      srcs[:TEMP_SUBSET], refs[:TEMP_SUBSET])


if __name__ == "__main__":
    main()
