"""Aggregate results of all models into tables, plots, translation samples and error indicators."""
import json
import re

import matplotlib.pyplot as plt
import pandas as pd

from common import OUTPUTS, PLOTS, RESULTS, TEMPERATURES, read_jsonl

MAIN = {"transformer": "Transformer (scratch)", "mt5": "mT5-small (FT)",
        "llama": "LLaMA-3.2-1B (3-shot)", "gemma": "Gemma-3-270M (LoRA)"}
EXTRA = {"llama_zeroshot": "LLaMA-3.2-1B (0-shot)", "gemma_zeroshot": "Gemma-3-270M (0-shot)"}
COLORS = {"transformer": "#4C72B0", "mt5": "#55A868", "llama": "#C44E52", "gemma": "#8172B2",
          "llama_zeroshot": "#E8A0A2", "gemma_zeroshot": "#BDB3D9"}
NUM = re.compile(r"\d+(?:[.,:]\d+)*")


def load(key):
    path = RESULTS / f"{key}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def error_indicators(key):
    """Cheap automatic signals of hallucination / under-translation on the test subset."""
    rows = read_jsonl(OUTPUTS / f"{key}_test.jsonl")
    n = len(rows)
    ratio = [len(r["hyp_de"].split()) / max(1, len(r["en"].split())) for r in rows]
    num_mismatch = sum(sorted(NUM.findall(r["en"])) != sorted(NUM.findall(r["hyp_de"])) for r in rows)
    return {
        "empty_%": round(100 * sum(not r["hyp_de"].strip() for r in rows) / n, 1),
        "copied_source_%": round(100 * sum(r["hyp_de"].strip().lower() == r["en"].strip().lower()
                                           for r in rows) / n, 1),
        "too_long_%": round(100 * sum(x > 2.0 for x in ratio) / n, 1),      # likely hallucinated additions
        "too_short_%": round(100 * sum(x < 0.5 for x in ratio) / n, 1),     # likely omissions
        "number_mismatch_%": round(100 * num_mismatch / n, 1),
    }


def main():
    rows = []
    for key, name in {**MAIN, **EXTRA}.items():
        r = load(key)
        if r is None:
            continue
        rows.append({"key": key, "Model": name, "Architecture": r["architecture"], "Adaptation": r["adaptation"],
                     "Total params (M)": round(r["params_total"] / 1e6, 1),
                     "Trainable params (M)": round(r["params_trainable"] / 1e6, 2),
                     "BLEU": r["bleu"], "chrF": r["chrf"],
                     "Train/FT time (min)": round(r["train_seconds"] / 60, 1),
                     "Inference time (s, 100 sents)": r["infer_seconds"],
                     "Sentences/s": r["infer_sent_per_s"], **error_indicators(key)})
    df = pd.DataFrame(rows)
    df.drop(columns="key").to_csv(RESULTS / "summary.csv", index=False)
    (RESULTS / "summary.md").write_text(df.drop(columns="key").to_markdown(index=False), encoding="utf-8")
    print(df.drop(columns="key").to_string(index=False))

    # --- quality bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ("BLEU", "chrF")):
        bars = ax.bar(df["Model"], df[metric], color=[COLORS[k] for k in df["key"]])
        ax.bar_label(bars, fmt="%.1f")
        ax.set(title=f"{metric} on 100-sentence Multi30k test subset", ylabel=metric)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(PLOTS / "quality_bleu_chrf.png", dpi=150)

    # --- cost: params, training time, inference time
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, col, log in zip(axes, ("Total params (M)", "Train/FT time (min)", "Inference time (s, 100 sents)"),
                            (True, False, False)):
        bars = ax.bar(df["Model"], df[col], color=[COLORS[k] for k in df["key"]])
        ax.bar_label(bars, fmt="%.1f")
        ax.set(title=col, yscale="log" if log else "linear")
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(PLOTS / "cost_params_time.png", dpi=150)

    # --- quality vs size
    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in df.iterrows():
        ax.scatter(r["Total params (M)"], r["BLEU"], s=120, color=COLORS[r["key"]])
        ax.annotate(r["Model"], (r["Total params (M)"], r["BLEU"]), textcoords="offset points", xytext=(6, 6))
    ax.set(xscale="log", xlabel="Total parameters (millions, log)", ylabel="BLEU",
           title="Translation quality vs model size")
    fig.tight_layout()
    fig.savefig(PLOTS / "bleu_vs_params.png", dpi=150)

    # --- temperature sweep
    temp = {}
    for key in MAIN:
        p = RESULTS / f"{key}_temperature.json"
        if p.exists():
            temp[key] = json.loads(p.read_text())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    trows = []
    for key, pts in temp.items():
        for metric, ax in zip(("bleu", "chrf"), axes):
            ax.plot([p["temperature"] for p in pts], [p[metric] for p in pts], marker="o",
                    color=COLORS[key], label=MAIN[key])
        trows += [{"Model": MAIN[key], "T": p["temperature"], "BLEU": p["bleu"], "chrF": p["chrf"]} for p in pts]
    for metric, ax in zip(("BLEU", "chrF"), axes):
        ax.set(xlabel="Sampling temperature (0 = greedy)", ylabel=metric, title=f"{metric} vs temperature",
               xticks=TEMPERATURES)
        ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "temperature_sweep.png", dpi=150)
    if trows:
        tdf = pd.DataFrame(trows).pivot_table(index="Model", columns="T", values=["BLEU", "chrF"])
        (RESULTS / "temperature_summary.md").write_text(tdf.round(2).to_markdown(), encoding="utf-8")

    # --- side-by-side samples (test subset + ethics probe)
    outs = {k: read_jsonl(OUTPUTS / f"{k}_test.jsonl") for k in MAIN if (OUTPUTS / f"{k}_test.jsonl").exists()}
    lines = ["# Translation samples (greedy decoding)\n"]
    for i in range(0, len(next(iter(outs.values()))), 10):
        first = next(iter(outs.values()))[i]
        lines += [f"### Test #{i}", f"- **EN:** {first['en']}", f"- **Reference DE:** {first['ref_de']}"]
        lines += [f"- **{MAIN[k]}:** {o[i]['hyp_de']}" for k, o in outs.items()]
        lines.append("")
    (OUTPUTS / "samples.md").write_text("\n".join(lines), encoding="utf-8")

    probes = {k: read_jsonl(OUTPUTS / f"{k}_ethics_probe.jsonl") for k in MAIN
              if (OUTPUTS / f"{k}_ethics_probe.jsonl").exists()}
    lines = ["# Responsible-AI probe translations\n"]
    for j, base in enumerate(next(iter(probes.values()), [])):
        lines += [f"### {base['id']}. [{base['category']}] {base['en']}", f"_What to check: {base['check']}_"]
        lines += [f"- **{MAIN[k]}:** {p[j]['hyp_de']}" for k, p in probes.items()]
        lines.append("")
    (OUTPUTS / "ethics_probe.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
