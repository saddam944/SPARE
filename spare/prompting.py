"""Prompt construction, output parsing and label-likelihood scoring for one-shot classification."""
import re
from difflib import get_close_matches

import numpy as np
import torch


def verbalize(label: str) -> str:
    return label.replace("_", " ").replace("-", " ").strip().lower()


def build_label_space(df):
    labels = sorted(df["label"].astype(str).unique())
    verb = {l: verbalize(l) for l in labels}
    rev = {v: k for k, v in verb.items()}
    return labels, verb, rev


def options_block(verb) -> str:
    return "; ".join(sorted(verb.values()))


def make_instruction(prompt_type: str, verb) -> str:
    opts = options_block(verb)
    if prompt_type == "domain":
        return ("The following text is a tweet posted during a disaster or crisis event. "
                f"Classify it into exactly one of the following humanitarian categories: {opts}. "
                "Return only the category name and nothing else.")
    if prompt_type == "task":
        return f"Classify the input text into one of the following categories: {opts}."
    if prompt_type == "generic":
        return f"Pick one category for the following text. The options are: {opts}."
    raise ValueError(prompt_type)


def one_shot_messages(prompt_type, verb, text, exemplars):
    """Instruction first, one labelled example per class in the middle, the query last."""
    instr = make_instruction(prompt_type, verb)
    shots = "\n\n".join(f"Text: {t}\nCategory: {verb[l]}" for t, l in exemplars)
    return [{"role": "user",
             "content": f"{instr}\n\nExamples:\n{shots}\n\nText: {text}\n\nCategory:"}]


def parse_output(gen_text: str, rev):
    """Map a generated answer to a dataset label, or None if no label matches."""
    g = gen_text.strip().lower()
    g = re.sub(r"^(category|answer|label)\s*[:\-]\s*", "", g).strip(" .\"'`\n")
    if g in rev:                                         # verbalized label
        return rev[g]
    canonical = {str(raw).strip().lower(): raw for raw in rev.values()}
    if g in canonical:                                   # canonical dataset label
        return canonical[g]
    for v, raw in rev.items():                           # verbalized label inside longer output
        if v and v in g:
            return raw
    for c, raw in canonical.items():                     # canonical label inside longer output
        if c and c in g:
            return raw
    close = get_close_matches(g, list(rev.keys()), n=1, cutoff=0.90)
    if close:
        return rev[close[0]]
    close = get_close_matches(g, list(canonical.keys()), n=1, cutoff=0.90)
    return canonical[close[0]] if close else None


@torch.no_grad()
def generate_label(model, tokenizer, messages, device, gen_kwargs) -> str:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(**enc, **gen_kwargs)
    return tokenizer.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def label_logprob(model, tokenizer, messages, label_text, device) -> float:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    p_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    a_ids = tokenizer(" " + label_text, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    full = torch.cat([p_ids, a_ids], dim=1)
    logits = model(full).logits
    lp = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    tok_lp = lp.gather(-1, full[:, 1:].unsqueeze(-1)).squeeze(-1)
    return float(tok_lp[:, p_ids.shape[1] - 1:].mean().item())


def class_probabilities(model, tokenizer, messages, labels, verb, device):
    s = np.array([label_logprob(model, tokenizer, messages, verb[l], device) for l in labels])
    s = s - s.max()
    p = np.exp(s)
    return p / p.sum()
