"""Download Multi30k (EN->DE), clean it, build splits and dataset statistics.

Source: https://github.com/multi30k/dataset (HF mirror: https://huggingface.co/datasets/bentrevett/multi30k)
"""
import json
import random
import unicodedata

import matplotlib.pyplot as plt
from datasets import load_dataset

from common import DATA, PLOTS, RESULTS, SEED, write_jsonl

TEST_SUBSET = 100   # common test set translated by all four models
MAX_WORDS = 50      # length filter for training pairs
MAX_RATIO = 3.0     # max source/target length ratio for training pairs


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return " ".join(text.split())  # str.split() also collapses non-breaking spaces


def clean(rows, filter_train=False):
    out, seen = [], set()
    dropped = {"empty": 0, "too_long": 0, "bad_ratio": 0, "duplicate": 0}
    for r in rows:
        en, de = normalize(r["en"]), normalize(r["de"])
        if not en or not de:
            dropped["empty"] += 1
            continue
        if filter_train:
            le, ld = len(en.split()), len(de.split())
            if le > MAX_WORDS or ld > MAX_WORDS:
                dropped["too_long"] += 1
                continue
            if max(le, ld) / max(1, min(le, ld)) > MAX_RATIO:
                dropped["bad_ratio"] += 1
                continue
            if (en, de) in seen:
                dropped["duplicate"] += 1
                continue
            seen.add((en, de))
        out.append({"en": en, "de": de})
    return out, dropped


def main():
    ds = load_dataset("bentrevett/multi30k")
    stats = {"source": "Multi30k (Elliott et al., 2016) — https://github.com/multi30k/dataset",
             "hf_mirror": "https://huggingface.co/datasets/bentrevett/multi30k",
             "direction": "English -> German", "splits": {}}
    cleaned = {}
    for split in ("train", "validation", "test"):
        rows, dropped = clean(ds[split], filter_train=(split == "train"))
        cleaned[split] = rows
        write_jsonl(DATA / f"{split}.jsonl", rows)
        en_len = [len(r["en"].split()) for r in rows]
        de_len = [len(r["de"].split()) for r in rows]
        stats["splits"][split] = {
            "raw": len(ds[split]), "kept": len(rows), "dropped": dropped,
            "avg_en_words": round(sum(en_len) / len(rows), 2),
            "avg_de_words": round(sum(de_len) / len(rows), 2),
            "max_en_words": max(en_len), "max_de_words": max(de_len),
            "en_vocab": len({w.lower() for r in rows for w in r["en"].split()}),
            "de_vocab": len({w.lower() for r in rows for w in r["de"].split()}),
        }

    rng = random.Random(SEED)
    subset = rng.sample(cleaned["test"], TEST_SUBSET)
    write_jsonl(DATA / "test_subset.jsonl", subset)
    stats["test_subset"] = TEST_SUBSET

    with open(RESULTS / "dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist([len(r["en"].split()) for r in cleaned["train"]], bins=40, alpha=0.6, label="English")
    ax.hist([len(r["de"].split()) for r in cleaned["train"]], bins=40, alpha=0.6, label="German")
    ax.set(xlabel="Sentence length (words)", ylabel="Count", title="Multi30k training sentence lengths")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "length_distribution.png", dpi=150)


if __name__ == "__main__":
    main()
