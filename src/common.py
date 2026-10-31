"""Shared helpers: paths, data loading, metrics, timing and result I/O."""
import json
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import sacrebleu
import torch

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUTPUTS = ROOT / "outputs"
PLOTS = ROOT / "plots"
CKPT = ROOT / "checkpoints"
for _d in (DATA, RESULTS, OUTPUTS, PLOTS, CKPT):
    _d.mkdir(exist_ok=True)

SRC_LANG, TGT_LANG = "en", "de"
SEED = 42
TEMPERATURES = [0.0, 0.5, 1.0, 1.5]  # 0.0 == greedy decoding
TEMP_SUBSET = 30                           # sentences used for the temperature sweep
INFER_BATCH = 8                            # identical inference batch size for all models

torch.set_num_threads(os.cpu_count())


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_split(name):
    """name in {train, validation, test, test_subset}."""
    return read_jsonl(DATA / f"{name}.jsonl")


def score(hyps, refs):
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2)}


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def clean_llm_output(text: str) -> str:
    """Decoder-only models sometimes add labels, quotes or extra lines; keep the translation only."""
    text = text.strip().split("\n")[0].strip()
    text = re.sub(r"^(German|Deutsch|Translation|Übersetzung)\s*:\s*", "", text, flags=re.I)
    if len(text) > 1 and text[0] in "\"'„“" and text[-1] in "\"'“”":
        text = text[1:-1].strip()
    return text


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def save_outputs(model_key, srcs, refs, hyps, suffix="test"):
    write_jsonl(OUTPUTS / f"{model_key}_{suffix}.jsonl",
                [{"en": s, "ref_de": r, "hyp_de": h} for s, r, h in zip(srcs, refs, hyps)])


def save_result(model_key, result: dict):
    with open(RESULTS / f"{model_key}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


def ethics_probe(model_key, translate_fn):
    """Translate the hand-written probe set (gender, culture, numbers, sensitive content)."""
    rows = read_jsonl(DATA / "ethics_probe.jsonl")
    hyps = translate_fn([r["en"] for r in rows])
    write_jsonl(OUTPUTS / f"{model_key}_ethics_probe.jsonl", [{**r, "hyp_de": h} for r, h in zip(rows, hyps)])


def temperature_sweep(model_key, translate_fn, srcs, refs):
    """translate_fn(list_of_src, temperature) -> list_of_hyp. Saves per-temperature scores."""
    rows = []
    for t in TEMPERATURES:
        set_seed()
        with Timer() as tm:
            hyps = translate_fn(srcs, t)
        s = score(hyps, refs)
        s.update(temperature=t, seconds=round(tm.seconds, 1))
        rows.append(s)
        save_outputs(model_key, srcs, refs, hyps, suffix=f"temp{t}")
        print(f"[{model_key}] T={t}: {s}")
    with open(RESULTS / f"{model_key}_temperature.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    return rows
