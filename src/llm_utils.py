"""Prompt construction and batched generation shared by the decoder-only models (LLaMA, Gemma)."""
import torch

from common import INFER_BATCH, clean_llm_output

INSTRUCTION = ("Translate the following English sentence into German. "
               "Reply with the German translation only.\n\nEnglish: {src}")


def build_messages(src, shots=()):
    """Few-shot examples are given as earlier user/assistant turns."""
    msgs = []
    for ex in shots:
        msgs += [{"role": "user", "content": INSTRUCTION.format(src=ex["en"])},
                 {"role": "assistant", "content": ex["de"]}]
    msgs.append({"role": "user", "content": INSTRUCTION.format(src=src)})
    return msgs


def prompt_text(tok, src, shots=()):
    return tok.apply_chat_template(build_messages(src, shots), tokenize=False, add_generation_prompt=True)


def translate(model, tok, srcs, temperature=0.0, shots=(), max_new_tokens=64):
    model.eval()
    tok.padding_side = "left"
    hyps = []
    for i in range(0, len(srcs), INFER_BATCH):
        prompts = [prompt_text(tok, s, shots) for s in srcs[i: i + INFER_BATCH]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        kw = dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0) if temperature > 0 \
            else dict(do_sample=False, temperature=None, top_k=None, top_p=None)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **kw)
        gen = out[:, enc["input_ids"].shape[1]:]
        hyps += [clean_llm_output(t) for t in tok.batch_decode(gen, skip_special_tokens=True)]
    return hyps
