"""Stage 3: FiNE neuron localization for every Ripple member (weights are never modified).

For a misclassified sample, the source object is its current wrong prediction. The contribution
of FFN neuron i in layer l to source token t is (Eq. 3 of FiNE)

    C^l_i(t) = q^l_i * (W_u W^l_out)_{t, i}

where q^l is the input of ``down_proj`` at the last prompt token. The top-k neurons are ranked
globally over all layers except the last ``exclude_last_n_layers``.
"""
import hashlib
import itertools
import json
import math

import pandas as pd
import torch
import torch.nn.functional as F

from .ripple_protect import load_scoped_predictions
from .utils import WRONG_TOKEN, RunContext, gpu_mem, require

LOC_COLS = ["ripple_group_id", "sample_id", "text", "true_label", "predicted_label",
            "source_token", "source_token_id", "layer_id", "neuron_id", "contribution_score", "rank"]


def _norm(s):
    return " ".join(str(s).split()).casefold()


@torch.no_grad()
def weight_digest(model):
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        d = p.detach().double()
        stat = torch.stack([d.sum(), (d ** 2).sum(), d.amin(), d.amax()]).cpu().numpy()
        h.update(n.encode("utf-8"))
        h.update(stat.tobytes())
    return h.hexdigest()


def reconstruct_prompt(tokenizer, exact_messages_json):
    """Return (prompt_str, input_ids) for the exact one-shot prompt used by the baseline."""
    obj = json.loads(exact_messages_json)
    if isinstance(obj, list):
        prompt_str = tokenizer.apply_chat_template(obj, tokenize=False, add_generation_prompt=True)
    else:
        prompt_str = str(obj)
    input_ids = tokenizer(prompt_str, add_special_tokens=False, return_tensors="pt").input_ids
    return prompt_str, input_ids


def contextual_label_ids(tokenizer, prompt_str, label_text):
    """Token ids of ``label_text`` as tokenized after the prompt (never in isolation).

    Returns (ids, prefix_clean); prefix_clean is False if the label merged with the last prompt token.
    """
    prompt_only = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
    full = tokenizer(prompt_str + label_text, add_special_tokens=False)["input_ids"]
    cpl = 0
    for a, b in zip(prompt_only, full):
        if a != b:
            break
        cpl += 1
    return full[cpl:], cpl == len(prompt_only)


class FineLocalizer:
    def __init__(self, ctx: RunContext):
        self.ctx = ctx
        self.tokenizer, self.model = ctx.load_model()
        cfg = self.model.config
        self.layers = self.model.model.layers
        self.d_h, self.d_m, self.n_layers = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers
        assert str(cfg.hidden_act).lower() in ("silu", "swish"), f"unexpected activation {cfg.hidden_act}"
        assert tuple(self.layers[0].mlp.down_proj.weight.shape) == (self.d_h, self.d_m)
        self.W_u = self.model.get_output_embeddings().weight          # [v, d_h]
        self.allowed_layers = list(range(0, self.n_layers - ctx.cfg.exclude_last_n_layers))
        assert self.allowed_layers, "exclude_last_n_layers is too large"
        self._capture = {}
        self._handles = []

    def __enter__(self):
        def make_hook(l):
            def hook(module, args):
                self._capture[l] = args[0][:, -1, :].detach().to(torch.float32)
            return hook
        self._handles = [self.layers[l].mlp.down_proj.register_forward_pre_hook(make_hook(l))
                         for l in self.allowed_layers]
        return self

    def __exit__(self, *exc):
        while self._handles:
            self._handles.pop().remove()

    @torch.no_grad()
    def last_token_activations(self, input_ids):
        self._capture.clear()
        self.model(input_ids=input_ids.to(self.ctx.device))
        return {l: self._capture[l][0] for l in self.allowed_layers}

    @torch.no_grad()
    def verify_q_mapping(self, input_ids):
        """Check that the captured down_proj input equals silu(gate_proj) * up_proj."""
        l0, buf = self.allowed_layers[0], {}
        mlp = self.layers[l0].mlp
        h1 = mlp.gate_proj.register_forward_hook(lambda m, a, o: buf.__setitem__("g", o[:, -1, :].float()))
        h2 = mlp.up_proj.register_forward_hook(lambda m, a, o: buf.__setitem__("u", o[:, -1, :].float()))
        self._capture.clear()
        self.model(input_ids=input_ids.to(self.ctx.device))
        h1.remove()
        h2.remove()
        ok = torch.allclose(F.silu(buf["g"]) * buf["u"], self._capture[l0], atol=1e-2, rtol=1e-2)
        assert ok, "down_proj input != silu(gate_proj) * up_proj"

    @torch.no_grad()
    def topk_neurons_for_tokens(self, q_by_layer, token_ids):
        """Global top-k (layer, neuron, score) per source token, following FiNE Eq. 3."""
        top_k = self.ctx.cfg.top_k
        WU = [self.W_u[t].to(torch.float32) for t in token_ids]
        Cs = [torch.empty(len(self.allowed_layers), self.d_m, dtype=torch.float32, device=WU[0].device)
              for _ in token_ids]
        for li, l in enumerate(self.allowed_layers):
            W_out = self.layers[l].mlp.down_proj.weight.to(torch.float32)   # [d_h, d_m]
            for ti in range(len(token_ids)):
                Cs[ti][li] = q_by_layer[l] * (WU[ti] @ W_out)
            del W_out
        out = []
        for C in Cs:
            vals, idx = torch.topk(C.reshape(-1), min(top_k, C.numel()))
            layer_pos, neuron_ids = (idx // self.d_m).tolist(), (idx % self.d_m).tolist()
            out.append([(self.allowed_layers[p], int(n), float(v))
                        for p, n, v in zip(layer_pos, neuron_ids, vals.tolist())])
        return out


def _exemplar_texts(exemplars_json):
    try:
        exs = json.loads(exemplars_json)
    except Exception:
        return []
    texts = []
    for e in exs if isinstance(exs, list) else []:
        if isinstance(e, dict):
            texts += [str(v) for k, v in e.items() if "text" in k.lower()]
        elif isinstance(e, (list, tuple)) and e:
            texts.append(str(e[0]))
        else:
            texts.append(str(e))
    return texts


def run_localization(ctx: RunContext):
    ripple = pd.read_csv(require(ctx.path("ripple_sets.csv"), "Run the 'groups' stage first."),
                         dtype=str, keep_default_na=False)
    protect = pd.read_csv(ctx.path("protect_sets.csv"), dtype=str, keep_default_na=False)
    assert len(ripple) > 0, "ripple_sets.csv is empty; nothing to localize."
    meta = load_scoped_predictions(ctx).drop_duplicates("sample_id").set_index("sample_id")

    loc = FineLocalizer(ctx)
    digest_before = weight_digest(loc.model)
    rows, failures, source_recon = [], [], {}
    faithfulness_checked = False

    def fail(gid, sid, reason, **extra):
        failures.append({"ripple_group_id": gid, "sample_id": sid, "reason": reason, **extra})

    gpu_mem("FiNE localize: start")
    with loc:
        for _, r in ripple.iterrows():
            sid, gid = str(r["sample_id"]), str(r["ripple_group_id"])
            true_label, pred_label, text = str(r["true_label"]), str(r["predicted_label"]), str(r["text"])
            if pred_label in (WRONG_TOKEN, true_label):
                fail(gid, sid, "not_a_valid_misclassification")
                continue
            if sid not in meta.index or not str(meta.loc[sid, "exact_messages_json"]):
                fail(gid, sid, "missing_exact_messages")
                continue
            prompt_str, input_ids = reconstruct_prompt(loc.tokenizer, meta.loc[sid, "exact_messages_json"])
            if _norm(text) not in _norm(prompt_str):
                fail(gid, sid, "prompt_verification_failed")
                continue

            # Source object = the current wrong prediction, tokenized in context.
            src_ids, prefix_clean = contextual_label_ids(loc.tokenizer, prompt_str, pred_label)
            if not src_ids:
                fail(gid, sid, "empty_source_tokenization")
                continue
            decoded = loc.tokenizer.decode(src_ids)
            if _norm(decoded) != _norm(pred_label):
                fail(gid, sid, "contextual_tokenization_mismatch", decoded=decoded)
                continue
            source_recon[sid] = decoded

            q_by_layer = loc.last_token_activations(input_ids)
            if not faithfulness_checked:
                loc.verify_q_mapping(input_ids)
                faithfulness_checked = True

            for t_id, topk in zip(src_ids, loc.topk_neurons_for_tokens(q_by_layer, src_ids)):
                tok_str = loc.tokenizer.decode([t_id])
                for rank, (lid, nid, score) in enumerate(topk, start=1):
                    if not math.isfinite(score):
                        fail(gid, sid, "non_finite_score", source_token_id=int(t_id))
                        continue
                    rows.append({"ripple_group_id": gid, "sample_id": sid, "text": text,
                                 "true_label": true_label, "predicted_label": pred_label,
                                 "source_token": tok_str, "source_token_id": int(t_id),
                                 "layer_id": int(lid), "neuron_id": int(nid),
                                 "contribution_score": float(score), "rank": int(rank)})
    gpu_mem("FiNE localize: end")

    fine = pd.DataFrame(rows, columns=LOC_COLS)
    fine.to_csv(ctx.path("FiNE_ripple_localization.csv"), index=False)
    pd.DataFrame(failures, columns=["ripple_group_id", "sample_id", "reason"] if not failures else None) \
        .to_csv(ctx.path("FiNE_localization_failures.csv"), index=False)
    _save_group_analysis(ctx, fine)

    assert weight_digest(loc.model) == digest_before, "model weights changed during localization"
    if len(fine):
        loc_ids = set(fine["sample_id"])
        assert loc_ids <= set(ripple["sample_id"]) and loc_ids.isdisjoint(set(protect["sample_id"]))
        assert (fine["predicted_label"] != fine["true_label"]).all()
        assert set(fine["layer_id"]) <= set(loc.allowed_layers)
        assert fine["neuron_id"].between(0, loc.d_m - 1).all()
        for sid in loc_ids:
            pl = fine.loc[fine["sample_id"] == sid, "predicted_label"].iloc[0]
            assert _norm(source_recon[sid]) == _norm(pl)
    print(f"localized {fine['sample_id'].nunique() if len(fine) else 0} samples "
          f"({len(fine)} neuron rows) | failures: {len(failures)}")
    return fine


def _save_group_analysis(ctx, fine):
    """Neuron sharing within each Ripple group (analysis only)."""
    member_neurons = {(gid, sid): set(map(tuple, g[["layer_id", "neuron_id"]].values.tolist()))
                      for (gid, sid), g in fine.groupby(["ripple_group_id", "sample_id"])}
    summary, overlap, layer_rows = [], [], []
    for gid, g in fine.groupby("ripple_group_id"):
        gsize = g["sample_id"].nunique()
        for (lid, nid), gg in g.groupby(["layer_id", "neuron_id"]):
            mem = gg["sample_id"].nunique()
            summary.append({"ripple_group_id": gid, "layer_id": int(lid), "neuron_id": int(nid),
                            "member_count": int(mem), "member_fraction": mem / gsize,
                            "mean_contribution": float(gg["contribution_score"].mean()),
                            "max_contribution": float(gg["contribution_score"].max())})
        for a, b in itertools.combinations(sorted(g["sample_id"].unique()), 2):
            Sa, Sb = member_neurons[(gid, a)], member_neurons[(gid, b)]
            inter, uni = len(Sa & Sb), len(Sa | Sb)
            overlap.append({"ripple_group_id": gid, "sample_id_a": a, "sample_id_b": b,
                            "intersection_size": inter, "union_size": uni,
                            "jaccard": inter / uni if uni else float("nan")})
        for lid, gl in g.groupby("layer_id"):
            layer_rows.append({"ripple_group_id": gid, "layer_id": int(lid),
                               "selected_neuron_count": int(gl["neuron_id"].nunique()),
                               "member_count": int(gl["sample_id"].nunique())})
    pd.DataFrame(summary).to_csv(ctx.path("FiNE_group_neuron_summary.csv"), index=False)
    pd.DataFrame(overlap).to_csv(ctx.path("FiNE_group_overlap.csv"), index=False)
    pd.DataFrame(layer_rows).to_csv(ctx.path("FiNE_layer_distribution.csv"), index=False)
