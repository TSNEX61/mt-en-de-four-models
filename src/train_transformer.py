"""Basic Transformer (encoder-decoder) trained from scratch on Multi30k EN->DE.

Joint SentencePiece BPE vocabulary, sinusoidal positions, pre-LN nn.Transformer,
label-smoothed cross-entropy, warmup + inverse-sqrt LR schedule, best checkpoint by val loss.
"""
import argparse
import math
import time

import sentencepiece as spm
import torch
import torch.nn as nn

from common import (CKPT, INFER_BATCH, RESULTS, TEMP_SUBSET, Timer, count_params, load_split,
                    save_outputs, save_result, score, set_seed, temperature_sweep)

PAD, UNK, BOS, EOS = 0, 1, 2, 3
MAX_LEN = 64


def train_tokenizer(train, vocab_size):
    corpus = CKPT / "spm_corpus.txt"
    with open(corpus, "w", encoding="utf-8") as f:
        for r in train:
            f.write(r["en"] + "\n" + r["de"] + "\n")
    prefix = str(CKPT / "spm_joint")
    spm.SentencePieceTrainer.train(input=str(corpus), model_prefix=prefix, vocab_size=vocab_size,
                                   model_type="bpe", character_coverage=1.0, pad_id=PAD, unk_id=UNK,
                                   bos_id=BOS, eos_id=EOS, minloglevel=2)
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class Seq2SeqTransformer(nn.Module):
    def __init__(self, vocab, d_model=256, heads=4, layers=3, ff=1024, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(vocab, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)
        self.transformer = nn.Transformer(d_model, heads, layers, layers, ff, dropout,
                                          batch_first=True, norm_first=True)
        self.out = nn.Linear(d_model, vocab)
        self.out.weight = self.tgt_emb.weight  # weight tying
        # std d^-0.5 (as in the original Transformer): with the sqrt(d) input scaling this keeps
        # embeddings ~unit scale and the tied output logits small at initialisation
        for emb in (self.src_emb, self.tgt_emb):
            nn.init.normal_(emb.weight, std=d_model ** -0.5)
            with torch.no_grad():
                emb.weight[PAD].zero_()

    def embed(self, emb, x):
        return self.drop(self.pos(emb(x) * math.sqrt(self.d_model)))

    def encode(self, src):
        return self.transformer.encoder(self.embed(self.src_emb, src), src_key_padding_mask=src == PAD)

    def decode(self, tgt, memory, src_pad):
        mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), dtype=torch.bool)
        h = self.transformer.decoder(self.embed(self.tgt_emb, tgt), memory, tgt_mask=mask,
                                     tgt_key_padding_mask=tgt == PAD, memory_key_padding_mask=src_pad)
        return self.out(h)

    def forward(self, src, tgt_in):
        return self.decode(tgt_in, self.encode(src), src == PAD)

    @torch.no_grad()
    def generate(self, src, temperature=0.0, max_len=MAX_LEN):
        memory, src_pad = self.encode(src), src == PAD
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long)
        done = torch.zeros(src.size(0), dtype=torch.bool)
        for _ in range(max_len):
            logits = self.decode(ys, memory, src_pad)[:, -1]
            if temperature > 0:
                nxt = torch.multinomial(torch.softmax(logits / temperature, -1), 1).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            nxt = nxt.masked_fill(done, PAD)
            ys = torch.cat([ys, nxt.unsqueeze(1)], 1)
            done |= nxt == EOS
            if done.all():
                break
        return ys[:, 1:]


def pad_batch(seqs):
    width = max(len(s) for s in seqs)
    return torch.tensor([s + [PAD] * (width - len(s)) for s in seqs], dtype=torch.long)


def make_batches(pairs, max_tokens, shuffle=True):
    """Length-bucketed batches of (src_ids, tgt_ids) with roughly max_tokens target tokens each."""
    order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][1]))
    batches, cur, cur_max = [], [], 0
    for i in order:
        cur_max = max(cur_max, len(pairs[i][1]))
        if cur and cur_max * (len(cur) + 1) > max_tokens:
            batches.append(cur)
            cur, cur_max = [], len(pairs[i][1])
        cur.append(i)
    if cur:
        batches.append(cur)
    if shuffle:
        perm = torch.randperm(len(batches)).tolist()
        batches = [batches[p] for p in perm]
    return batches


def encode_pairs(sp, rows):
    return [(sp.encode(r["en"])[: MAX_LEN - 1] + [EOS], [BOS] + sp.encode(r["de"])[: MAX_LEN - 2] + [EOS])
            for r in rows]


def run_epoch(model, pairs, crit, max_tokens, opt=None, sched=None):
    model.train(opt is not None)
    total, n_tok = 0.0, 0
    for idx in make_batches(pairs, max_tokens, shuffle=opt is not None):
        src = pad_batch([pairs[i][0] for i in idx])
        tgt = pad_batch([pairs[i][1] for i in idx])
        with torch.set_grad_enabled(opt is not None):
            logits = model(src, tgt[:, :-1])
            loss = crit(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        tokens = (tgt[:, 1:] != PAD).sum().item()
        total += loss.item() * tokens
        n_tok += tokens
    return total / n_tok


def translate(model, sp, srcs, temperature=0.0):
    model.eval()
    hyps = []
    for i in range(0, len(srcs), INFER_BATCH):
        src = pad_batch([sp.encode(s)[: MAX_LEN - 1] + [EOS] for s in srcs[i: i + INFER_BATCH]])
        for row in model.generate(src, temperature).tolist():
            row = row[: row.index(EOS)] if EOS in row else row
            hyps.append(sp.decode([t for t in row if t != PAD]))
    return hyps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--max_minutes", type=float, default=90)
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=7e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--n_train", type=int, default=0, help="use only the first N training pairs (0 = all)")
    args = ap.parse_args()
    set_seed()

    train, val, test = load_split("train"), load_split("validation"), load_split("test_subset")
    if args.n_train:
        train = train[: args.n_train]
    sp = train_tokenizer(train, args.vocab)
    train_p, val_p = encode_pairs(sp, train), encode_pairs(sp, val)

    model = Seq2SeqTransformer(args.vocab)
    total_params, trainable = count_params(model)
    print(f"params: {total_params:,}")
    crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / args.warmup, math.sqrt(args.warmup / (s + 1))))

    history, best, best_epoch = [], float("inf"), 0
    t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_p, crit, args.max_tokens, opt, sched)
        va = run_epoch(model, val_p, crit, args.max_tokens)
        elapsed = time.perf_counter() - t0
        history.append({"epoch": epoch, "train_loss": round(tr, 4), "val_loss": round(va, 4),
                        "elapsed_s": round(elapsed, 1)})
        print(history[-1], flush=True)
        if va < best:
            best, best_epoch = va, epoch
            torch.save(model.state_dict(), CKPT / "transformer_best.pt")
        if elapsed > args.max_minutes * 60:
            print("time budget reached")
            break
    train_seconds = time.perf_counter() - t0
    model.load_state_dict(torch.load(CKPT / "transformer_best.pt"))

    srcs, refs = [r["en"] for r in test], [r["de"] for r in test]
    with Timer() as tm:
        hyps = translate(model, sp, srcs)
    save_outputs("transformer", srcs, refs, hyps)
    result = {"model": "Basic Transformer (from scratch)", "architecture": "encoder-decoder",
              "adaptation": "trained from scratch", "params_total": total_params,
              "params_trainable": trainable, "train_examples": len(train_p),
              "epochs_run": len(history), "best_epoch": best_epoch,
              "train_seconds": round(train_seconds, 1), "infer_seconds": round(tm.seconds, 1),
              "infer_sent_per_s": round(len(srcs) / tm.seconds, 2), **score(hyps, refs),
              "history": history, "config": vars(args)}
    save_result("transformer", result)

    temperature_sweep("transformer", lambda s, t: translate(model, sp, s, t),
                      srcs[:TEMP_SUBSET], refs[:TEMP_SUBSET])


if __name__ == "__main__":
    main()
