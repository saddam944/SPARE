"""Stage 4: preservation-aware sequential editing.

1. Each Ripple group gets a neuron mask: the FiNE neurons shared by at least ``min_member_fraction``
   of its members.
2. The editor H_theta is trained over the whole trajectory of groups W0 -> W1 -> ... -> WT
   (one optimizer step per full traversal). For group t the loss is
       sum_{i=t-k..t} mu^(t-i) NLL(group i)  +  lambda * KL(P_{W_{t-1}} || P_{W_t}) on Protect
       +  eta * ||dW_t||^2
   and the group gradient is projected with A-GEM against a reference gradient from the Protect memory.
3. The trained editor is applied group by group to the physical weights without resetting them.
"""
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .editor import LowLevelEditor
from .localization import contextual_label_ids, reconstruct_prompt
from .ripple_protect import load_scoped_predictions
from .utils import WRONG_TOKEN, RunContext, gpu_mem, norm_label, require


class EditingEngine:
    """Prepared examples, Ripple groups and the forward-hook patch used for editing and evaluation."""

    def __init__(self, ctx: RunContext):
        self.ctx, self.cfg = ctx, ctx.cfg
        self.device = ctx.device
        self.tokenizer, self.model = ctx.load_model()
        mcfg = self.model.config
        self.d_h, self.d_m = mcfg.hidden_size, mcfg.intermediate_size
        self.layers = self.model.model.layers
        self._handles = []
        self.editor = None

        self.fine = pd.read_csv(require(ctx.path("FiNE_ripple_localization.csv"),
                                        "Run the 'localize' stage first."), dtype=str, keep_default_na=False)
        for c in ("layer_id", "neuron_id"):
            self.fine[c] = self.fine[c].astype(int)
        self.ripple = pd.read_csv(ctx.path("ripple_sets.csv"), dtype=str, keep_default_na=False)
        self.protect_sets = pd.read_csv(ctx.path("protect_sets.csv"), dtype=str, keep_default_na=False)
        self.meta = load_scoped_predictions(ctx).drop_duplicates("sample_id").set_index("sample_id")

        self.global_protect_ids = [s for s in self.protect_sets["sample_id"].unique() if self._has_meta(s)]
        self.groups = self._build_groups()
        self.all_edited_layers = sorted({l for g in self.groups for l in g["sel_by_layer"]})
        for g in self.groups:
            g["col_mask"] = {}
            for l, ns in g["sel_by_layer"].items():
                assert all(0 <= n < self.d_m for n in ns)
                m = torch.zeros(1, self.d_m, device=self.device, dtype=torch.float32)
                m[0, ns] = 1.0
                g["col_mask"][l] = m
            g["edited_layers"] = sorted(g["sel_by_layer"])

        # Pristine weights of every edited layer, used to restore W0.
        self.W0 = {l: self.layers[l].mlp.down_proj.weight.detach().clone() for l in self.all_edited_layers}

        self.global_protect = [self.build_example(s, self._protect_label(s)) for s in self.global_protect_ids]
        for g in self.groups:
            g["members"] = [self.build_example(s, self._true_label(s)) for s in g["member_ids"]]
            g["protect"] = self.global_protect

        labels = load_scoped_predictions(ctx)["true_label"].dropna().unique()
        self.label_lookup = {norm_label(x): norm_label(x) for x in labels}
        print(f"trajectory: {len(self.groups)} groups | edited layers: {self.all_edited_layers} | "
              f"ripple examples: {sum(len(g['members']) for g in self.groups)} | "
              f"global Protect memory: {len(self.global_protect)}")

    # -- groups and examples -------------------------------------------------------
    def _has_meta(self, sid):
        return sid in self.meta.index and bool(str(self.meta.loc[sid].get("exact_messages_json", "")))

    def _group_mask(self, gid):
        g = self.fine[self.fine["ripple_group_id"] == str(gid)]
        members = [s for s in g["sample_id"].unique() if self._has_meta(s)]
        if not members:
            return {}, members
        per_member = {s: set() for s in members}
        for _, r in g.iterrows():
            if r["sample_id"] in per_member:
                per_member[r["sample_id"]].add((int(r["layer_id"]), int(r["neuron_id"])))
        if self.cfg.fine_mask_mode == "union":
            chosen = set().union(*per_member.values())
        elif self.cfg.fine_mask_mode == "member_fraction":
            cnt = Counter(ln for s in members for ln in per_member[s])
            chosen = {ln for ln, c in cnt.items() if c / len(members) >= self.cfg.min_member_fraction}
        else:
            raise ValueError(f"unknown fine_mask_mode {self.cfg.fine_mask_mode!r}")
        sel = defaultdict(list)
        for l, n in chosen:
            sel[l].append(n)
        return {l: sorted(v) for l, v in sel.items()}, members

    def _build_groups(self):
        gids = sorted(self.fine["ripple_group_id"].unique(),
                      key=lambda x: (int(x) if x.isdigit() else 1_000_000, x))
        groups = []
        for gid in gids:
            sel_by_layer, members = self._group_mask(gid)
            if not members or not sel_by_layer:
                print(f"  [skip] group {gid}: members={len(members)} editable layers={len(sel_by_layer)}")
                continue
            tl = self.ripple[self.ripple["sample_id"] == members[0]]["true_label"]
            groups.append({"gid": gid, "member_ids": members, "sel_by_layer": sel_by_layer,
                           "true_label": tl.iloc[0] if len(tl) else "?"})
        if self.cfg.max_groups is not None:
            groups = groups[:self.cfg.max_groups]
        assert groups, "no editable groups"
        return groups

    def _true_label(self, sid):
        r = self.ripple[self.ripple["sample_id"] == sid]
        return r.iloc[0]["true_label"] if len(r) else "?"

    def _protect_label(self, sid):
        pr = self.protect_sets[self.protect_sets["sample_id"] == sid].iloc[0]
        return pr["true_label"] if pr["true_label"] else pr["predicted_label"]

    def build_example(self, sid, target_label):
        prompt_str, pid = reconstruct_prompt(self.tokenizer, self.meta.loc[sid, "exact_messages_json"])
        lab, _ = contextual_label_ids(self.tokenizer, prompt_str, target_label)
        full = torch.cat([pid, torch.tensor([lab])], dim=1).to(self.device)
        return {"sample_id": sid, "prompt_len": pid.shape[1], "full_ids": full,
                "label_ids": lab, "target": target_label}

    # -- teacher-forced metrics ------------------------------------------------------
    def label_positions(self, prompt_len, n_label):
        return torch.arange(prompt_len - 1, prompt_len - 1 + n_label, device=self.device)

    @torch.no_grad()
    def eval_example(self, ex):
        logits = self.model(input_ids=ex["full_ids"], use_cache=False).logits
        pos = self.label_positions(ex["prompt_len"], len(ex["label_ids"]))
        lp = F.log_softmax(logits[0, pos].float(), -1)
        tgt = torch.tensor(ex["label_ids"], device=self.device)
        hit = lp.argmax(-1) == tgt
        return {"top1": hit.float().mean().item(), "exact": bool(hit.all().item())}

    @torch.no_grad()
    def answer_distribution(self, ex):
        logits = self.model(input_ids=ex["full_ids"][:, :ex["prompt_len"]], use_cache=False).logits
        return F.softmax(logits[0, ex["prompt_len"] - 1].float(), -1)

    def acc_over(self, examples):
        if not examples:
            return {"exact": float("nan"), "top1": float("nan"), "n": 0}
        r = [self.eval_example(e) for e in examples]
        return {"exact": float(np.mean([x["exact"] for x in r])),
                "top1": float(np.mean([x["top1"] for x in r])), "n": len(examples)}

    def group_mean_nll(self, group):
        tot = 0.0
        for ex in group["members"]:
            tot = tot + self._nll(self.model(input_ids=ex["full_ids"], use_cache=False).logits[0], ex)
        return tot / len(group["members"])

    def _nll(self, logits_row, ex):
        pos = self.label_positions(ex["prompt_len"], len(ex["label_ids"]))
        lp = F.log_softmax(logits_row[pos].float(), -1)
        tgt = torch.tensor(ex["label_ids"], dtype=torch.long, device=self.device)
        return -lp[torch.arange(len(tgt), device=self.device), tgt].mean()

    # -- generated classification ----------------------------------------------------
    def parse_generated_label(self, raw):
        raw_norm = norm_label(raw)
        if raw_norm in self.label_lookup:
            return self.label_lookup[raw_norm]
        matches = sorted(((len(k), v) for k, v in self.label_lookup.items() if k in raw_norm), reverse=True)
        return matches[0][1] if matches else WRONG_TOKEN

    @torch.no_grad()
    def generated_prediction(self, ex):
        prompt_ids = ex["full_ids"][:, :ex["prompt_len"]]
        out = self.model.generate(input_ids=prompt_ids, **self.ctx.gen_kwargs())
        raw = self.tokenizer.decode(out[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        pred = self.parse_generated_label(raw)
        true_label = norm_label(self.tokenizer.decode(ex["label_ids"], skip_special_tokens=True))
        return {"sample_id": str(ex["sample_id"]), "true_label": true_label, "predicted_label": pred,
                "raw_output": raw, "correct": pred == true_label}

    def generated_acc_over(self, examples):
        rows = [self.generated_prediction(ex) for ex in examples]
        return {"acc": float(np.mean([r["correct"] for r in rows])) if rows else float("nan"),
                "n": len(rows), "rows": rows}

    # -- weight patch ----------------------------------------------------------------
    def install_patch(self, delta_map):
        """down_proj output -> base output + x @ delta_l^T (delta_l may carry gradients to the editor)."""
        self.remove_patch()
        for l in self.all_edited_layers:
            def make(l):
                def hook(module, args, output):
                    d = delta_map.get(l)
                    if d is None:
                        return output
                    x = args[0]
                    return output + F.linear(x, d.to(dtype=x.dtype, device=x.device), None)
                return hook
            self._handles.append(self.layers[l].mlp.down_proj.register_forward_hook(make(l)))

    def remove_patch(self):
        while self._handles:
            self._handles.pop().remove()

    def restore_W0(self):
        with torch.no_grad():
            for l in self.all_edited_layers:
                self.layers[l].mlp.down_proj.weight.copy_(self.W0[l])

    def compute_uv(self, ex, delta_state, feature_layers):
        """Rank-1 factors (u_l, v_l) at the last prompt position under W0 + delta_state.

        u_l is the input of down_proj and v_l the gradient of the edit loss w.r.t. its output.
        """
        for l in feature_layers:
            self.layers[l].mlp.down_proj.weight.requires_grad_(True)
        inp, out, handles = {}, {}, []
        for l in self.all_edited_layers:
            def make(l, d_l, cap):
                def hook(m, args, output):
                    x, o = args[0], output
                    if d_l is not None:
                        o = o + F.linear(x, d_l.to(dtype=x.dtype, device=x.device), None)
                    if cap:
                        inp[l], out[l] = x, o
                    return o
                return hook
            handles.append(self.layers[l].mlp.down_proj.register_forward_hook(
                make(l, delta_state.get(l), l in feature_layers)))
        try:
            with torch.enable_grad():
                logits = self.model(input_ids=ex["full_ids"], use_cache=False).logits
                loss = self._nll(logits[0], ex)
                grads = torch.autograd.grad(loss, [out[l] for l in feature_layers])
        finally:
            for h in handles:
                h.remove()
            for l in feature_layers:
                self.layers[l].mlp.down_proj.weight.requires_grad_(False)
        ps = ex["prompt_len"] - 1
        return {l: (inp[l][0, ps].detach().float(), g[0, ps].detach().float())
                for l, g in zip(feature_layers, grads)}

    def make_group_delta(self, group, delta_state):
        """Differentiable group update: mean over members of the masked editor outputs.

        Only the selected columns are accumulated ([d_h, S] per member) and scattered into a dense
        [d_h, d_m] matrix at the end, which equals averaging outer(v~, u~) * col_mask.
        """
        n = len(group["members"])
        els = group["edited_layers"]
        sel_idx = {l: group["col_mask"][l][0].nonzero(as_tuple=False).flatten() for l in els}
        acc = {l: None for l in els}
        coef = self.editor.coefficient()
        for ex in group["members"]:
            uv = self.compute_uv(ex, delta_state, els)
            for l in els:
                ut, vt = self.editor(uv[l][0], uv[l][1], l)
                contrib = coef * torch.outer(vt, ut.index_select(0, sel_idx[l]))
                acc[l] = contrib if acc[l] is None else acc[l] + contrib
        return {l: torch.zeros(self.d_h, self.d_m, dtype=acc[l].dtype, device=acc[l].device)
                .index_copy(1, sel_idx[l], acc[l] / n) for l in els}

    def cap_group_delta(self, gdW, group):
        """Optional locality cap ||(W - W0) + alpha dW|| <= edit_cap * ||W0|| on the selected columns."""
        cap = self.cfg.edit_cap
        if cap is None or cap <= 0:
            return gdW
        c2 = d2 = cd = w02 = 0.0
        for l in group["edited_layers"]:
            idx = torch.tensor(group["sel_by_layer"][l], device=self.W0[l].device, dtype=torch.long)
            base = self.W0[l].float().index_select(1, idx)
            C = self.layers[l].mlp.down_proj.weight.detach().float().index_select(1, idx) - base
            D = gdW[l].detach().float().index_select(1, idx)
            c2 += float((C * C).sum())
            d2 += float((D * D).sum())
            cd += float((C * D).sum())
            w02 += float((base * base).sum())
        budget = cap * math.sqrt(max(w02, 0.0))
        if d2 <= 0.0 or c2 + 2 * cd + d2 <= budget ** 2:
            return gdW
        disc = (2 * cd) ** 2 - 4 * d2 * (c2 - budget ** 2)
        alpha = 0.0
        if disc >= 0:
            r1 = (-2 * cd - math.sqrt(disc)) / (2 * d2)
            r2 = (-2 * cd + math.sqrt(disc)) / (2 * d2)
            left, right = max(0.0, min(r1, r2)), min(1.0, max(r1, r2))
            alpha = right if left <= right else 0.0
        return {l: gdW[l] * alpha for l in gdW}

    # -- batched losses (execution only; same objective as per-example loops) -----------
    def _pad_right(self, seqs):
        seqs = [x.reshape(-1).to(self.device) for x in seqs]
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        max_len = max(int(x.numel()) for x in seqs)
        input_ids = torch.full((len(seqs), max_len), int(pad_id), dtype=seqs[0].dtype, device=self.device)
        mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=self.device)
        for b, ids in enumerate(seqs):
            input_ids[b, :ids.numel()] = ids
            mask[b, :ids.numel()] = 1
        return input_ids, mask

    def nll_batch_sum(self, examples):
        """Sum over examples of the per-example mean NLL of the target label tokens."""
        if not examples:
            return torch.zeros((), dtype=torch.float32, device=self.device)
        ids, mask = self._pad_right([ex["full_ids"][0] for ex in examples])
        logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        return torch.stack([self._nll(logits[b], ex) for b, ex in enumerate(examples)]).sum()

    @torch.no_grad()
    def protect_distributions(self, memory):
        out = []
        for batch in _chunks(memory, self.cfg.before_batch):
            ids, mask = self._pad_right([ex["full_ids"][0, :ex["prompt_len"]] for ex in batch])
            logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            out += [F.softmax(logits[b, ex["prompt_len"] - 1].float(), -1).detach()
                    for b, ex in enumerate(batch)]
        return out

    def protect_kl_batch_sum(self, examples, P_before):
        """Sum over examples of KL(P_before || P_after) at the answer position."""
        ids, mask = self._pad_right([ex["full_ids"][0, :ex["prompt_len"]] for ex in examples])
        logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
        return torch.stack([
            F.kl_div(F.log_softmax(logits[b, ex["prompt_len"] - 1].float(), -1), Pb, reduction="sum")
            for b, (ex, Pb) in enumerate(zip(examples, P_before))]).sum()

    # -- training ----------------------------------------------------------------------
    def new_editor(self):
        cfg = self.cfg
        self.editor = LowLevelEditor(self.d_m, self.d_h, self.all_edited_layers, d_r=cfg.editor_rank,
                                     C=cfg.editor_blocks, init_scale=cfg.edit_init_scale,
                                     edit_sign=cfg.edit_sign).to(self.device).float()
        print(f"editor params: {sum(p.numel() for p in self.editor.parameters()):,}")
        return self.editor

    def _checkpoint_keys(self):
        cfg = self.cfg
        return {"train_steps": cfg.train_steps, "lr": cfg.lr, "gamma": cfg.gamma, "mu": cfg.mu,
                "memory_k": cfg.memory_k, "lambda_protect": cfg.lambda_protect, "eta": cfg.eta,
                "use_agem": cfg.use_agem, "agem_ref_batch": cfg.agem_ref_batch,
                "agem_ref_chunk": cfg.agem_ref_chunk, "n_samples": cfg.n_samples, "seed": cfg.seed}

    def train_editor(self):
        cfg, editor = self.cfg, self.editor or self.new_editor()
        params = list(editor.parameters())
        opt = torch.optim.Adam(params, lr=cfg.lr)
        ckpt_path = str(self.ctx.edit_dir / "training_checkpoint.pt")
        history, start_step = [], 0

        if cfg.resume and os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            for k, v in self._checkpoint_keys().items():
                assert ckpt[k] == v, f"checkpoint was made with {k}={ckpt[k]}, current config has {v}"
            editor.load_state_dict(ckpt["editor_state_dict"])
            opt.load_state_dict(ckpt["optimizer_state_dict"])
            history, start_step = list(ckpt["history"]), int(ckpt["next_step"])
            _restore_rng_state(ckpt)
            print(f"[resume] continuing from traversal {start_step}/{cfg.train_steps}")

        def zeros():
            return [torch.zeros_like(p) for p in params]

        def add_(buf, grads):
            with torch.no_grad():
                for b, g in zip(buf, grads):
                    if g is not None:
                        b.add_(g.detach())

        memory = list(self.global_protect)
        assert memory, "No Protect samples found."
        n_protect = len(memory)
        gpu_mem("train_editor: start")

        for step in range(start_step, cfg.train_steps):
            delta_state, trajectory_grad, J, projected = {}, zeros(), 0.0, 0
            for t, group in enumerate(self.groups):
                gt = cfg.gamma ** t

                # Protect distributions under W_{t-1}.
                P_before = None
                if cfg.lambda_protect != 0:
                    self.install_patch(delta_state)
                    P_before = self.protect_distributions(memory)
                    self.remove_patch()

                # Differentiable group action dW_t, applied on top of the detached previous state.
                gdW = self.make_group_delta(group, delta_state)
                self.install_patch(_merge(delta_state, gdW))
                g_cur, L_val = zeros(), 0.0

                # Edit loss with memory backtracking over groups [t-k, t].
                for i in range(max(0, t - cfg.memory_k), t + 1):
                    members = self.groups[i]["members"]
                    w = cfg.mu ** (t - i) / max(1, len(members))
                    for batch in _chunks(members, cfg.nll_batch):
                        c = gt * w * self.nll_batch_sum(batch)
                        add_(g_cur, torch.autograd.grad(c, params, retain_graph=True, allow_unused=True))
                        L_val += float(c.detach())

                # KL to the previous state on the full Protect memory.
                if cfg.lambda_protect != 0:
                    for s in range(0, n_protect, cfg.kl_batch):
                        c = gt * cfg.lambda_protect / n_protect * self.protect_kl_batch_sum(
                            memory[s:s + cfg.kl_batch], P_before[s:s + cfg.kl_batch])
                        add_(g_cur, torch.autograd.grad(c, params, retain_graph=True, allow_unused=True))
                        L_val += float(c.detach())

                # Update-norm regularizer.
                need_ref = cfg.use_agem and len(memory) > 0
                L_upd = gt * cfg.eta * sum((gdW[l] ** 2).sum() for l in gdW)
                add_(g_cur, torch.autograd.grad(L_upd, params, retain_graph=need_ref, allow_unused=True))
                L_val += float(L_upd.detach())

                # A-GEM: project g_cur if it conflicts with the Protect reference gradient.
                if need_ref:
                    ref = memory if len(memory) <= cfg.agem_ref_batch else random.sample(memory, cfg.agem_ref_batch)
                    K, g_ref = len(ref), zeros()
                    chunks = list(_chunks(ref, max(1, cfg.agem_ref_chunk)))
                    for j, chunk in enumerate(chunks):
                        mbs = list(_chunks(chunk, cfg.agem_microbatch))
                        for m, mb in enumerate(mbs):
                            last = j == len(chunks) - 1 and m == len(mbs) - 1
                            add_(g_ref, torch.autograd.grad(self.nll_batch_sum(mb) / K, params,
                                                            retain_graph=not last, allow_unused=True))
                    ref_norm2 = float(sum((x.float() ** 2).sum() for x in g_ref))
                    if ref_norm2 > 0:
                        dot = float(sum((x.float() * y.float()).sum() for x, y in zip(g_cur, g_ref)))
                        if dot < 0:
                            with torch.no_grad():
                                for gc, gr in zip(g_cur, g_ref):
                                    gc.add_(gr, alpha=-dot / ref_norm2)
                            projected += 1
                    del g_ref

                with torch.no_grad():
                    for a, g in zip(trajectory_grad, g_cur):
                        a.add_(g)
                self.remove_patch()

                # Detached transition W_{t-1} -> W_t.
                for l in gdW:
                    delta_state[l] = gdW[l].detach() if l not in delta_state else delta_state[l] + gdW[l].detach()
                J += gt * (-L_val)
                del g_cur, gdW, P_before

            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                for p, g in zip(params, trajectory_grad):
                    p.grad = g.clone()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            opt.step()
            history.append(J)

            _atomic_save({"editor_state_dict": editor.state_dict(), "optimizer_state_dict": opt.state_dict(),
                          "history": history, "next_step": step + 1, **self._checkpoint_keys(),
                          **_capture_rng_state()}, ckpt_path)
            print(f"traversal {step:3d} | J={J:10.4f} | grad_norm={float(grad_norm):.4f} | "
                  f"A-GEM projections={projected}/{len(self.groups)}")
            del trajectory_grad
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        torch.save({"editor_state_dict": editor.state_dict(), "history": history, **self._checkpoint_keys()},
                   self.ctx.edit_dir / "editor_final.pt")
        pd.DataFrame({"traversal": range(len(history)), "reward_J": history}).to_csv(
            self.ctx.edit_dir / "training_history.csv", index=False)
        self.remove_patch()
        self.restore_W0()
        return history

    # -- sequential editing of the physical weights ------------------------------------
    @torch.no_grad()
    def _group_update(self, group):
        """Mean editor update for a group under the current physical weights."""
        gdW = {}
        for ex in group["members"]:
            uv = self.compute_uv(ex, {}, group["edited_layers"])
            for l in group["edited_layers"]:
                d = self.editor.delta_w(uv[l][0], uv[l][1], l, group["col_mask"][l])
                gdW[l] = d if l not in gdW else gdW[l] + d
        return {l: v / len(group["members"]) for l, v in gdW.items()}

    def sequential_edit(self):
        """Apply W0 -> W1 -> ... -> WT with the trained editor and log per-group diagnostics."""
        cfg, ctx = self.cfg, self.ctx
        self.editor.eval()
        self.remove_patch()
        self.restore_W0()

        # Protect accuracy and distributions at the current weights; reused as the next "before".
        prot_tf = self.acc_over(self.global_protect)
        prot_gen = self.generated_acc_over(self.global_protect)
        P_prev = [self.answer_distribution(e) for e in self.global_protect]

        sanity = []
        for t, g in enumerate(self.groups):
            rip = self.generated_acc_over(g["members"])
            sanity.append({"group": t, "group_id": g["gid"], "ripple_generated_W0": rip["acc"],
                           "n_ripple": rip["n"], "protect_generated_W0": prot_gen["acc"],
                           "n_protect": prot_gen["n"]})
        pd.DataFrame(sanity).to_csv(ctx.edit_path("W0_group_generated_sanity.csv"), index=False)

        logs, detail, prev_members = [], [], []
        for t, group in enumerate(self.groups):
            rip_tf_before = self.acc_over(group["members"])
            rip_gen_before = self.generated_acc_over(group["members"])
            prot_tf_before, prot_gen_before, P_before = prot_tf, prot_gen, P_prev

            gdW = self.cap_group_delta(self._group_update(group), group)
            with torch.no_grad():
                for l in group["edited_layers"]:
                    w = self.layers[l].mlp.down_proj.weight
                    w.add_(gdW[l].to(w.dtype))

            rip_tf_after = self.acc_over(group["members"])
            rip_gen_after = self.generated_acc_over(group["members"])
            prot_tf = self.acc_over(self.global_protect)
            prot_gen = self.generated_acc_over(self.global_protect)
            P_prev = [self.answer_distribution(e) for e in self.global_protect]
            kl = float(np.mean([F.kl_div(Pa.clamp_min(1e-12).log(), Pb, reduction="sum").item()
                                for Pa, Pb in zip(P_prev, P_before)])) if P_prev else float("nan")
            retention = self.acc_over(prev_members) if prev_members else \
                {"exact": float("nan"), "top1": float("nan"), "n": 0}

            with torch.no_grad():
                L_edit = float(sum(cfg.mu ** (t - i) * float(self.group_mean_nll(self.groups[i]))
                                   for i in range(max(0, t - cfg.memory_k), t + 1)))
                L_update = float(sum((gdW[l] ** 2).sum() for l in gdW))
            L_t = L_edit + cfg.lambda_protect * (0.0 if math.isnan(kl) else kl) + cfg.eta * L_update

            logs.append({
                "t": t, "group_id": group["gid"], "true_label": group["true_label"],
                "n_members": len(group["members"]), "n_protect": len(self.global_protect),
                "ripple_teacher_exact_before": rip_tf_before["exact"], "ripple_teacher_exact_after": rip_tf_after["exact"],
                "ripple_teacher_top1_before": rip_tf_before["top1"], "ripple_teacher_top1_after": rip_tf_after["top1"],
                "ripple_generated_acc_before": rip_gen_before["acc"], "ripple_generated_acc_after": rip_gen_after["acc"],
                "protect_teacher_exact_before": prot_tf_before["exact"], "protect_teacher_exact_after": prot_tf["exact"],
                "protect_generated_acc_before": prot_gen_before["acc"], "protect_generated_acc_after": prot_gen["acc"],
                "protect_kl": kl,
                "prev_retention_teacher_exact": retention["exact"], "prev_retention_teacher_top1": retention["top1"],
                "prev_groups_n": retention["n"],
                "r_low": -L_t, "L_t": L_t, "L_edit": L_edit, "L_update": L_update,
                "update_norm": math.sqrt(L_update),
                "edited_layers": ",".join(map(str, group["edited_layers"])),
            })
            print(f"[group {t}] {group['true_label']}: ripple generated {rip_gen_before['acc']:.3f} -> "
                  f"{rip_gen_after['acc']:.3f} | protect generated {prot_gen_before['acc']:.3f} -> "
                  f"{prot_gen['acc']:.3f} | KL={kl:.4f} | prev top1={retention['top1']:.3f}")

            before = {x["sample_id"]: x for x in rip_gen_before["rows"] + prot_gen_before["rows"]}
            after = {x["sample_id"]: x for x in rip_gen_after["rows"] + prot_gen["rows"]}
            member_ids = {str(e["sample_id"]) for e in group["members"]}
            for sid in sorted(set(before) & set(after)):
                b, a = before[sid], after[sid]
                detail.append({"t": t, "group_id": group["gid"],
                               "set_type": "ripple" if sid in member_ids else "protect",
                               "sample_id": sid, "true_label": b["true_label"],
                               "pred_before": b["predicted_label"], "pred_after": a["predicted_label"],
                               "correct_before": b["correct"], "correct_after": a["correct"],
                               "raw_before": b["raw_output"], "raw_after": a["raw_output"]})
            prev_members.extend(group["members"])
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        logs_df = pd.DataFrame(logs)
        logs_df.to_csv(ctx.edit_path("seq_edit_generated_diagnostic.csv"), index=False)
        pd.DataFrame(detail).to_csv(ctx.edit_path("seq_edit_generated_per_instance.csv"), index=False)
        print(f"Sequential editing complete: W0 -> ... -> W{len(self.groups)}. The model is now at WT.")
        return logs_df


def run_editing(ctx: RunContext):
    engine = EditingEngine(ctx)
    engine.new_editor()
    engine.train_editor()
    engine.sequential_edit()
    if ctx.cfg.save_edited_model:
        out = ctx.edit_dir / "FINAL_WT_MODEL"
        ctx.model.save_pretrained(out, safe_serialization=True)
        ctx.tokenizer.save_pretrained(out)
        print(f"Saved edited model to {out}")
    return engine


# -- helpers ----------------------------------------------------------------------------
def _chunks(seq, size):
    seq, size = list(seq), max(1, int(size))
    for s in range(0, len(seq), size):
        yield seq[s:s + size]


def _merge(delta_state, gdW):
    out = dict(delta_state)
    for l, cur in gdW.items():
        base = delta_state.get(l)
        out[l] = cur if base is None else base.to(cur.dtype) + cur
    return out


def _capture_rng_state():
    state = {"python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
             "torch_cpu_rng_state": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(ckpt):
    random.setstate(ckpt["python_random_state"])
    np.random.set_state(ckpt["numpy_random_state"])
    torch.set_rng_state(ckpt["torch_cpu_rng_state"])
    if torch.cuda.is_available() and "torch_cuda_rng_state_all" in ckpt:
        torch.cuda.set_rng_state_all(ckpt["torch_cuda_rng_state_all"])


def _atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)
