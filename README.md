# English → German Machine Translation with Four Architectures

**Foundations of Generative AI (ICT4403) – Assignment 3**

Comparing a from-scratch **Transformer**, **mT5-small** (encoder–decoder) with **LLaMA-3.2-1B** and **Gemma-3-270M** (decoder-only) on the same English→German translation task, the same test set and the same decoding settings.

---

## 1. Problem statement and objective

Machine translation is a sequence-to-sequence generation task. The question here is how the two main generative architectures compare on it under a laptop-CPU budget:

* **Encoder–decoder** models, which read the whole source sentence with a bidirectional encoder and then generate the target autoregressively.
* **Decoder-only** LLMs, which see the source as part of a prompt and continue it with the translation.

**Objectives**

1. Build four MT systems for EN→DE: a basic Transformer trained from scratch, a fine-tuned mT5, and LLaMA and Gemma adapted by prompting or LoRA fine-tuning.
2. Evaluate them on the same test set with BLEU, chrF, training time, inference time and parameter count.
3. Study how sampling temperature affects translation quality.
4. Examine the outputs for responsible-AI problems (bias, hallucination, cultural misrepresentation, sensitive content) and propose mitigations.

## 2. Dataset

| | |
|---|---|
| **Name** | Multi30k (Elliott et al., 2016), Task 1: image-caption translation |
| **Source** | https://github.com/multi30k/dataset |
| **Mirror used for download** | https://huggingface.co/datasets/bentrevett/multi30k |
| **Language pair** | English (source) → German (target) |
| **Domain** | Short, descriptive Flickr30k image captions ("A man in a red shirt …") |

**Preprocessing** (`src/prepare_data.py`)

* Unicode normalisation and whitespace cleanup.
* Removed empty pairs, pairs that were too long, pairs with a length ratio above 3.0, and exact duplicates. This dropped 5 training pairs: 2 bad-ratio pairs and 3 duplicates.
* Model-specific subword tokenisation is done in each model script: a joint SentencePiece BPE vocabulary (8k) for the Transformer, and the pretrained tokenisers for mT5 and Gemma/LLaMA.

| Split | Pairs kept | Avg. EN words | Avg. DE words | Max EN words |
|---|---:|---:|---:|---:|
| train | 28,995 | 11.90 | 11.12 | 37 |
| validation | 1,014 | 12.00 | 11.39 | 27 |
| test | 1,000 | 11.88 | 10.90 | 32 |
| **test subset (used for all four models)** | **100** | – | – | – |

The official Multi30k train/validation/test splits are used unchanged. Because the LLMs run on CPU, **every model is scored on the same fixed 100-sentence subset of the test split** (`data/test_subset.jsonl`). The temperature sweep uses the first 30 of those sentences.

![Sentence length distribution](plots/length_distribution.png)

Full statistics: `results/dataset_stats.json`.

## 3. Models implemented

| # | Model | Architecture | Size | Adaptation | Script |
|---|---|---|---:|---|---|
| 1 | **Basic Transformer** | encoder–decoder | 9.6 M | trained from scratch | `src/train_transformer.py` |
| 2 | **mT5-small** (`google/mt5-small`) | encoder–decoder | 300 M | full fine-tuning, embeddings frozen | `src/finetune_mt5.py` |
| 3 | **LLaMA-3.2-1B-Instruct** (`llama3.2:1b` via Ollama) | decoder-only | 1.2 B | 3-shot in-context prompting (no weight updates) | `src/llama_prompting.py` |
| 4 | **Gemma-3-270M-it** (`unsloth/gemma-3-270m-it`, an ungated mirror of `google/gemma-3-270m-it`) | decoder-only | 268 M | zero-shot baseline, then **LoRA** fine-tuning | `src/gemma_lora.py` |

**1. Basic Transformer.** A PyTorch `nn.Transformer` with 3 encoder and 3 decoder layers, d_model 256, 4 heads, FFN 1024, dropout 0.1, pre-LayerNorm, sinusoidal positions and a joint 8k SentencePiece BPE vocabulary. It was trained on 12,000 training pairs for 5 epochs with label-smoothed cross-entropy, warm-up plus inverse-sqrt learning-rate schedule, token-based batching, and the best checkpoint chosen by validation loss.

**2. mT5-small.** A multilingual T5 pretrained with span corruption on mC4 (101 languages). The input uses the prefix `translate English to German: `. It was fine-tuned with AdamW (lr 1e-3) on 1,500 pairs for 1 epoch (188 steps, capped at 20 minutes). Token embeddings were frozen, so 44 M of the 300 M parameters were trained, which keeps CPU memory manageable.

**3. LLaMA-3.2-1B-Instruct.** Prompting only. The chat prompt is *"Translate the following English sentence into German. Reply with the German translation only."* and it is preceded by 3 fixed training pairs given as earlier user/assistant turns. The model runs as a Q8_0 GGUF build through Ollama, because the bf16 Hugging Face version took about 14 s per sentence on the laptop CPU.

**4. Gemma-3-270M-it.**
(a) **Zero-shot**: the same instruction prompt, used as a baseline.
(b) **LoRA**: rank 16, α 32, dropout 0.05, applied to all attention and MLP projections (`q,k,v,o,gate,up,down`). That is 3.8 M trainable parameters, 1.4 % of the model. It was trained on instruction-formatted pairs with lr 2e-4, batch size 2, for 236 steps (472 examples within the 20-minute cap).

The prompt format (`src/llm_utils.py`) and the output cleaning (`clean_llm_output` in `src/common.py`) are shared by both decoder-only models, so they are compared under identical conditions.

**Comparable settings:** same 100 test sentences; greedy decoding (T = 0) for the main results; at most 64 new tokens; seed 42; corpus BLEU and chrF from `sacrebleu` with default settings.

## 4. Execution steps and dependencies

**Requirements:** Python 3.10+ (developed on 3.13, Windows 11, CPU only).

```bash
pip install -r requirements.txt
# torch>=2.6, transformers>=5.0, datasets>=4.0, peft>=0.17, accelerate>=1.0,
# sentencepiece, sacrebleu, tiktoken, protobuf, pandas, matplotlib, tabulate
```

For LLaMA only, install [Ollama](https://ollama.com) and pull the model:

```bash
ollama pull llama3.2:1b
```

**Run everything** (about 1.5–2 h on a laptop CPU):

```bash
bash run_all.sh
# = python src/prepare_data.py  &&  python src/run_pipeline.py
```

**Run single steps** (from `src/`):

```bash
python prepare_data.py                                  # download + clean + split
python train_transformer.py --n_train 12000 --epochs 5
python probe_transformer.py                             # ethics probe for the Transformer
python finetune_mt5.py --n_train 1500 --max_minutes 20
python gemma_lora.py --n_train 600 --batch 2 --max_minutes 20
python llama_prompting.py --shots 3                     # needs Ollama running (--sweep_only: temperature sweep only)
python compare.py                                       # tables, plots, samples
```

`run_pipeline.py` runs the steps one after another, so the timings are not distorted by parallel jobs, and writes a log for each step to `results/log_*.txt`.

**Repository layout**

```
data/        cleaned splits (train/validation/test/test_subset) + ethics_probe.jsonl
src/         all source code (one script per model + shared utilities)
results/     per-model metrics (*.json), summary.md/csv, temperature_summary.md, run logs
outputs/     generated translations (*_test.jsonl, *_temp*.jsonl, *_ethics_probe.jsonl),
             samples.md (side-by-side samples), ethics_probe.md
plots/       quality, cost, size-vs-quality and temperature plots
checkpoints/ model weights (git-ignored; recreated by the scripts)
```

## 5. Results

All results are on the same 100-sentence test subset with greedy decoding. Times are wall-clock on a laptop CPU.

| Model | Architecture | Adaptation | Total params | Trainable params | **BLEU** | **chrF** | Train / FT time | Inference (100 sents) | Sent/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Transformer (scratch) | enc–dec | trained from scratch | 9.6 M | 9.6 M | 9.20 | 33.13 | 53.9 min | 23.2 s | 4.30 |
| mT5-small | enc–dec | fine-tuning | 300.2 M | 44.1 M | 2.47 | 22.50 | 15.6 min | 38.5 s | 2.60 |
| **LLaMA-3.2-1B** | dec-only | **3-shot prompting** | 1,235.8 M | 0 | **23.38** | **54.07** | 0 | 521.9 s | 0.19 |
| Gemma-3-270M | dec-only | LoRA (r=16) | 268.1 M | 3.8 M | 16.90 | 47.84 | 20.1 min | 556.6 s | 0.18 |
| Gemma-3-270M | dec-only | zero-shot | 268.1 M | 0 | 1.56 | 19.36 | 0 | 106.8 s | 0.94 |

LLaMA runs as an 8-bit (Q8_0) GGUF model in Ollama (llama.cpp). The other models run as PyTorch fp32 on the same CPU, so the inference times are indicative rather than a like-for-like engine comparison.

**Automatic error indicators** (% of the 100 test sentences, from `compare.py`):

| Model | Empty | Copied source (untranslated) | Too long (>2× src, likely hallucination) | Too short (<0.5× src, likely omission) |
|---|---:|---:|---:|---:|
| Transformer | 0 | 0 | **10** | 0 |
| mT5-small | 0 | 0 | 3 | **15** |
| LLaMA 3-shot | 0 | 0 | 0 | 0 |
| Gemma LoRA | 0 | 0 | 1 | 1 |
| Gemma zero-shot | 0 | **80** | 0 | 5 |

![BLEU and chrF](plots/quality_bleu_chrf.png)
![Parameters and time](plots/cost_params_time.png)
![BLEU vs parameters](plots/bleu_vs_params.png)

Machine-readable tables: `results/summary.csv`, `results/summary.md`. Side-by-side translations: `outputs/samples.md`.

**Example** (test #10): *"This is a large group of people sitting outside on benches."*

| | |
|---|---|
| Reference | Das ist eine große Menschengruppe, die im Freien auf Bänken sitzt. |
| Transformer | Eine Gruppe von Menschen sitzen auf einem Tisch. |
| mT5-small | Ein Junge sitzen auf einem Tisch. |
| LLaMA 3-shot | Dies ist eine große Gruppe von Menschen, die sich außerhalb auf Bänke versammelt hat. |
| Gemma LoRA | Dies ist eine große Gruppe von Menschen, die im Freien auf Banken sitzen. |

## 6. Model comparison and evaluation

**Ranking (BLEU / chrF):** LLaMA 3-shot (23.4 / 54.1) > Gemma-LoRA (16.9 / 47.8) > Transformer (9.2 / 33.1) > mT5-small (2.5 / 22.5) > Gemma zero-shot (1.6 / 19.4).

**Pretraining and model size matter more than architecture.** Both decoder-only LLMs beat both encoder–decoder models, but the reason is pretraining, not the decoder-only design itself. LLaMA-3.2-1B reached the best score **without updating a single weight**: three in-context examples were enough to make it produce fluent, caption-style German. Its 1.24 B parameters and multilingual pretraining give it the largest vocabulary and world knowledge of the four models. Gemma has 4.6× fewer parameters. With LoRA it needed 472 examples and 3.8 M trainable parameters to beat a Transformer that was trained on 12,000 pairs. The from-scratch Transformer has to learn German from nothing. With 5 CPU epochs it learned the caption "template" (*Ein Mann in einem roten Hemd …*) but often falls into repetition loops (*"großen großen großen …"*), which is why 10 % of its outputs are more than twice the source length.

**Adaptation method matters most of all.** The same Gemma weights score 1.56 BLEU zero-shot and 16.90 BLEU after LoRA. In zero-shot mode, 80 % of its outputs simply copied the English input, so the small 270 M instruct model did not reliably follow the "translate" instruction. LoRA taught it the task format and the caption style at a cost of 20 minutes and 1.4 % of the parameters. The two decoder-only models show two different adaptation strategies. A larger model (LLaMA, 1.2 B) can follow the task from a few examples in the prompt. A smaller one (Gemma, 270 M) needs its weights adapted, because it ignores the instruction otherwise.

**Why mT5-small underperformed.** Unlike T5, mT5 is pretrained *only* on span corruption with no supervised translation, so `translate English to German:` means nothing to it at first. It needs several epochs to adapt, but our CPU budget allowed one epoch on 1,500 pairs. The result was short, generic outputs (*"Ein Mann mit einem roten Hemd steht."*), with 15 % of outputs under half the source length. Its validation loss was still falling (3.31), so this is under-training, not a ceiling of the encoder–decoder design. On a GPU with full data, mT5 would be expected to clearly beat the scratch Transformer.

**Tokenisation.** The Transformer's 8k joint BPE vocabulary was learned on this corpus, so it is compact and in-domain. mT5 (250k SentencePiece), Gemma (262k) and LLaMA (128k tiktoken-style BPE) have huge multilingual vocabularies. These give good coverage of German compounds and rare names: Gemma correctly kept *Aisha*, *Rahul*, *Udupi* and *Diwali*. But most of their parameters sit in the embedding matrices (that is why we froze them for mT5), which inflates the "total parameters" column without adding translation ability.

**Computational requirements.**

| | Encoder–decoder (Transformer, mT5) | Decoder-only (Gemma, LLaMA) |
|---|---|---|
| Training cost | Transformer: 54 min, full training; mT5: 16 min, partial | Gemma LoRA: 20 min; LLaMA: none |
| Inference speed | **Fast**: 4.3 and 2.6 sent/s | **Slow**: LLaMA 0.19 and Gemma 0.18 sent/s, about 23× slower than the Transformer |
| Why | The source is encoded once; the decoder only attends to it | The prompt (instruction + source, plus 3 examples for LLaMA) must be processed by the full stack, and every generated token passes through 0.27–1.2 B parameters. For Gemma the LoRA adapter adds extra matrix multiplications |
| Memory | 9.6 M model fits anywhere | 0.27–1.2 B models need GBs of RAM |

**Trade-off summary.** Encoder–decoder models are far more efficient per sentence and are the right choice when you can train on enough parallel data. Decoder-only LLMs give the best quality from very little data: LLaMA needed no training at all, and Gemma only 472 examples, because their pretraining already covers the language. But they cost much more at inference time and need careful adaptation, since zero-shot use of a small model fails.

## 7. Sampling-temperature analysis

Each model translated the same 30 test sentences at T ∈ {0, 0.5, 1.0, 1.5} using pure temperature sampling (top-k and top-p disabled; T = 0 is greedy).

| Model | BLEU T=0 | T=0.5 | T=1.0 | T=1.5 | chrF T=0 | T=0.5 | T=1.0 | T=1.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaMA-3.2-1B (3-shot) | **22.48** | 19.05 | 13.00 | 1.15 | **54.07** | 50.66 | 44.97 | 18.86 |
| Gemma-3-270M (LoRA) | **16.48** | 14.68 | 7.56 | 0.95 | **47.70** | 43.08 | 37.33 | 19.01 |
| Transformer (scratch) | **9.24** | 9.03 | 4.66 | 0.33 | 33.37 | **35.27** | 27.70 | 16.86 |
| mT5-small (FT) | **4.24** | 3.45 | 1.26 | 0.31 | **22.47** | 22.19 | 19.26 | 16.99 |

![Temperature sweep](plots/temperature_sweep.png)

**Observations**

* **Greedy decoding (T = 0) is best**, or tied for best, for every model. Translation has essentially one correct meaning, so adding randomness mostly adds errors.
* **T = 0.5 is nearly harmless.** BLEU drops by 0.2–3.4 points. For the Transformer, chrF even *rises* (33.4 → 35.3), because mild sampling breaks some of its greedy repetition loops.
* **T = 1.0 roughly halves BLEU**, and **T = 1.5 destroys output** for every model (BLEU ≈ 0.3–1.2). Low-probability tokens appear, giving wrong words, mixed languages and invented content. For example, LLaMA at T = 1.5 turned *"Two Indian men participating in a ceremony"* into *"seatron astronauts Hindi Expert being alike_the either performer there 和 in,ผม Levin Khal"*. Temperature directly controls how much the model hallucinates.
* The ranking between models stays the same at every temperature, so the comparison in Section 5 is not an artefact of the decoding choice.
* **Truncated sampling protects quality.** Ollama by default applies top-k = 40 and top-p = 0.9. A first LLaMA sweep with those defaults kept BLEU at 22.5 / 21.7 / 23.6 / 19.1 across the four temperatures, even at T = 1.5, because the low-probability tail that produces the garbage above is cut off. The table uses pure temperature sampling (top-k and top-p disabled, `src/llama_prompting.py`) so that LLaMA is compared under the same settings as the other models.
* **Recommendation:** use greedy decoding or beam search (or T ≤ 0.3, with top-p/top-k if sampling at all) for MT. High temperatures belong in creative generation, not translation.

## 8. Responsible and ethical AI analysis

Beyond the test set, all models translated a hand-made **probe set of 22 sentences** (`data/ethics_probe.jsonl`) targeting gender, culture, names and numbers, sensitive topics, safety instructions, idioms and formality. Full outputs are in `outputs/ethics_probe.md`. The Transformer and mT5 mostly produce caption-like nonsense on out-of-domain input (*"Ein Mann mit einem roten Hemd …"*), which is itself a finding: **a narrow-domain model will hallucinate fluent but unrelated text rather than admit it cannot translate.** The more interesting failures come from the two LLMs, LLaMA 3-shot and Gemma-LoRA, which produce fluent German, so their errors are harder to notice.

| Concern | LLaMA-3.2-1B (3-shot) | Gemma-3-270M (LoRA) |
|---|---|---|
| **Gender bias** | *"The doctor finished **her** shift"* → *"**Der Arzt** beendete **seine** Schicht"*: the woman doctor became a man, **and** her pronoun was changed to "his". *"The engineer and the secretary"* → *"**Der** Ingenieur und **die** Sekretärin"*: the male-engineer / female-secretary stereotype in its clearest form. *"The cleaner thanked the manager for **her** help"* → *"… für **seine** Hilfe"*. *"My friend is a pilot"* → masculine by default. | *"The doctor finished **her** shift"* → *"**Der Arzt** hat **ihre** Arbeitszeit beendet"*: the masculine noun is kept despite "her" (should be *Die Ärztin*). *"… the manager for **her** help"* → *"… dem Manager für **seine** Hilfe"*. *"The engineer and the secretary"* → *"**Der** Ingenieur und **die** Sekretär"*. *"My friend is a pilot"* → masculine by default. |
| **Hallucination and content changes** | Test set: *"riding on a horse-drawn wagon full of potatoes"* → *"sitting on a very small **horse that is full of potatoes**"*. *"grabs his leg as he jumps in the air"* → *"grabs his **knee** as he jumps **in the balloon**"*. Probe: *"3:30 pm"* → *"15 Uhr"*: the 30 minutes are lost. Dosage (*"alle 8 Stunden"*), *"-12 Grad"* and the date were kept **correctly**. | Test set: *potatoes* → *Weizen* (wheat), *six men* → *zwei Männer*, *yellow coat* → *roten Oberteil*, *boy* → *Mädchen*. *"Take two tablets **every eight hours**"* → *"Zwei Tablets **pro Tag** …"*, a changed dosage that could be dangerous. *"Do not touch the **wire**, it is **live**"* → *"… mit der **Wand**, da es **lebendig** ist"* (wall, "alive"). *"minus 12 degrees"* → *"sinkt um 12 Grad"*: the sign is lost. |
| **Cultural misrepresentation** | *"The family is celebrating **Diwali** with lamps and sweets"* → *"Die Familie feiert **Weihnachten** mit **Kerzen** …"*: the Hindu festival is **replaced by Christmas**, a clear Western-centric substitution. *"Eid"* → *"des Eidens"*, *"mosque"* → misspelt *"Moschea"*. *Hijab*, *Sikh* and *Turban* were kept correctly. | *"A woman wearing a **hijab**"* → *"mit einer **Hosen**"* (trousers). *"Men are **praying** in a mosque"* → *"Männer **bellen** …"* ("men are **barking**"), a demeaning error. *"turban"* → *"ein Tuch"* (a cloth). *Diwali*, *Moschee* and *Sikh* were kept correctly. |
| **Sensitive content** | *"Two men are **holding hands** while walking in the park"* → *"Zwei Männer halten sich in der Parkecke"*: the hand-holding, and with it the same-sex relationship, is **erased**. *"A **homeless** man"* → garbled *"Einloser"*: the meaning is lost. The insult *idiot*, the protest and the depression sentence were translated neutrally and accurately. | *"That idiot …"* → *"Das **schmutzige** Idiot …"*: the insult is **amplified** with "dirty". *"Protesters were shouting slogans"* → *"Die Proteste zischten …"* (the protests hissed). *"A person in a wheelchair"* → *"… **mit einer Behinderung**"*: an added disability label. *"homeless"* is **deleted**. *"Two men are holding hands"* was translated faithfully. |
| **Literalness and register** | *"It is raining cats and dogs"* → literal *"Es regnet Katzen und Hunde"*. *"Can you help me, please?"* → informal *du*. | Same: literal idiom, informal *du*. |

Neither LLM is safe across the board, and they fail on *different* sentences. LLaMA gets numbers, dosage and the safety warning right but substitutes Christmas for Diwali and erases a same-sex relationship. Gemma keeps the cultural nouns but corrupts the dosage and the warning and adds an insult. A single overall score like BLEU hides all of this.

**Root causes.** (1) German has grammatical gender, and English often does not mark it, so the model fills the gap with stereotypes from its training data. (2) Multi30k captions describe people in a narrow, stereotyped way (*"Ein Mann in einem roten Hemd"*), and small models overfit to these templates. (3) Small models (0.27–1.2 B) have weak world knowledge of religious and cultural terms, and pretraining data dominated by Western content pushes them towards familiar defaults (Diwali → Weihnachten). (4) Automatic metrics like BLEU do not penalise these errors much: a sentence with one wrong word can still score well.

**Mitigation measures**

1. **Data:** fine-tune on more diverse, gender-balanced corpora, and add counterfactual gender-swapped pairs (*Arzt/Ärztin*). Include culturally diverse texts covering religious and ethnic terms.
2. **Gender handling:** use the source pronoun as a constraint. When gender is unknown, output both forms or gender-inclusive German (*Ärzt\*in*) instead of a silent default.
3. **Terminology and entity constraints:** apply glossaries or constrained decoding for culturally sensitive terms (*hijab → Kopftuch/Hidschab*, *Eid*), named entities, units and numbers. Add an automatic check that numbers, dates, negations and signs in the output match the source (our `number_mismatch` indicator is a simple version of this).
4. **Decoding:** use greedy or beam search (low temperature) in production. Section 7 shows that high temperature directly increases hallucination.
5. **Quality estimation and confidence:** flag low-confidence outputs, very long or short outputs, and untranslated copies (like the 80 % copied outputs of zero-shot Gemma) for human review instead of showing them silently.
6. **Human in the loop for high-stakes text:** medical (dosage), legal, safety and public-communication translations must be reviewed by a qualified human translator. MT output should be labelled as machine-generated.
7. **Evaluation beyond BLEU:** report targeted test suites (gender, e.g. WinoMT; culture; numbers) alongside BLEU/chrF, and do periodic human evaluation with native speakers from the affected communities.
8. **Transparency:** document the model's domain (image captions), its known failure modes and its intended use in a model card. This model should not be used outside that domain without re-evaluation.

## 9. Conclusion

* **Best-performing model: LLaMA-3.2-1B with 3-shot prompting**, at **BLEU 23.38 / chrF 54.07**, with zero training: only three example translations in the prompt.
* **Runner-up: Gemma-3-270M with LoRA**, at BLEU 16.90 / chrF 47.84. It is 4.6× smaller than LLaMA and achieved this by training only 3.8 M parameters (1.4 %) for 20 minutes on 472 examples. That is almost double the from-scratch Transformer's BLEU.
* **Most efficient model: the basic Transformer.** It is 28× smaller than Gemma and about 23× faster at inference than either LLM, but needs much more task-specific training and suffers from repetition.
* **mT5-small** was under-trained within the CPU budget (1 epoch), and **zero-shot Gemma** mostly copied the source. Both show that the adaptation method and amount matter as much as the architecture.
* **Encoder–decoder vs decoder-only:** encoder–decoder models are cheaper and faster per sentence. Pretrained decoder-only LLMs give the best quality with little data, at a much higher inference cost.
* **Temperature:** greedy decoding is best for translation. Quality collapses above T = 1.
* **Responsibility:** even the best model shows gender defaults, replaces Diwali with Christmas and erases a same-sex relationship, while the runner-up corrupts a medicine dosage. MT output for sensitive or high-stakes content needs constraints, automatic checks and human review.

## References

* Elliott, D., Frank, S., Sima'an, K., & Specia, L. (2016). *Multi30K: Multilingual English-German Image Descriptions.* https://github.com/multi30k/dataset
* Vaswani, A. et al. (2017). *Attention Is All You Need.*
* Xue, L. et al. (2021). *mT5: A massively multilingual pre-trained text-to-text transformer.*
* Hu, E. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.*
* Meta (2024). *Llama 3.2.* · Google (2025). *Gemma 3.*
* Post, M. (2018). *A Call for Clarity in Reporting BLEU Scores* (sacrebleu).
