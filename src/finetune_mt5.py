"""Fine-tune mT5-small (encoder-decoder, multilingual span-corruption pretraining) on EN->DE."""
import argparse
import gc
import math
import time

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from common import (INFER_BATCH, TEMP_SUBSET, Timer, count_params, load_split, save_outputs,
                    save_result, score, set_seed, temperature_sweep, ethics_probe)

MODEL_ID = "google/mt5-small"
PREFIX = "translate English to German: "
MAX_LEN = 64


def translate(model, tok, srcs, temperature=0.0):
    model.eval()
    hyps = []
    for i in range(0, len(srcs), INFER_BATCH):
        enc = tok([PREFIX + s for s in srcs[i: i + INFER_BATCH]], return_tensors="pt", padding=True,
                  truncation=True, max_length=MAX_LEN)
        kw = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0) if temperature > 0 \
            else dict(do_sample=False)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_LEN, num_beams=1, **kw)
        hyps += [h.strip() for h in tok.batch_decode(out, skip_special_tokens=True)]
    return hyps


@torch.no_grad()
def val_loss(model, tok, rows, bs=4):
    model.eval()
    total, n = 0.0, 0
    for i in range(0, len(rows), bs):
        batch = make_batch(tok, rows[i: i + bs])
        total += model(**batch).loss.item() * len(rows[i: i + bs])
        n += len(rows[i: i + bs])
    return total / n


def make_batch(tok, rows):
    enc = tok([PREFIX + r["en"] for r in rows], text_target=[r["de"] for r in rows], return_tensors="pt",
              padding=True, truncation=True, max_length=MAX_LEN)
    enc["labels"][enc["labels"] == tok.pad_token_id] = -100
    return enc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max_minutes", type=float, default=60)
    args = ap.parse_args()
    set_seed()

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    # Freeze the 250k-token input embedding and LM head (~256M params): keeps optimizer state
    # small enough for a 16 GB laptop; all encoder/decoder layers are still fine-tuned.
    for p in list(model.shared.parameters()) + list(model.lm_head.parameters()):
        p.requires_grad = False
    total_params, trainable = count_params(model)
    train = load_split("train")[: args.n_train]
    val = load_split("validation")[:100]
    test = load_split("test_subset")

    steps_total = math.ceil(len(train) / args.batch) * args.epochs
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / 50, max(0.0, (steps_total - s) / max(1, steps_total - 50))))

    history, step, t0, stop = [], 0, time.perf_counter(), False
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(len(train)).tolist()
        model.train()
        for i in range(0, len(train), args.batch):
            loss = model(**make_batch(tok, [train[j] for j in perm[i: i + args.batch]])).loss
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 25 == 0:
                print(f"epoch {epoch} step {step}/{steps_total} loss {loss.item():.4f} "
                      f"elapsed {time.perf_counter() - t0:.0f}s", flush=True)
            if time.perf_counter() - t0 > args.max_minutes * 60:
                stop = True
                break
        history.append({"epoch": epoch, "steps": step, "last_train_loss": round(loss.item(), 4),
                        "elapsed_s": round(time.perf_counter() - t0, 1)})
        print(history[-1], flush=True)
        if stop:
            print("time budget reached")
            break
    train_seconds = time.perf_counter() - t0
    # free optimizer state before evaluating: the 250k-vocab logits need the RAM on a 16 GB laptop
    del opt, loss
    gc.collect()
    history[-1]["val_loss"] = round(val_loss(model, tok, val), 4)
    print(history[-1], flush=True)

    srcs, refs = [r["en"] for r in test], [r["de"] for r in test]
    with Timer() as tm:
        hyps = translate(model, tok, srcs)
    save_outputs("mt5", srcs, refs, hyps)
    save_result("mt5", {
        "model": MODEL_ID, "architecture": "encoder-decoder", "adaptation": "fine-tuning (embeddings frozen)",
        "params_total": total_params, "params_trainable": trainable,
        "train_examples": min(len(train), step * args.batch), "steps": step,
        "train_seconds": round(train_seconds, 1), "infer_seconds": round(tm.seconds, 1),
        "infer_sent_per_s": round(len(srcs) / tm.seconds, 2), **score(hyps, refs),
        "history": history, "config": vars(args)})

    ethics_probe("mt5", lambda s: translate(model, tok, s))
    temperature_sweep("mt5", lambda s, t: translate(model, tok, s, t), srcs[:TEMP_SUBSET], refs[:TEMP_SUBSET])


if __name__ == "__main__":
    main()
