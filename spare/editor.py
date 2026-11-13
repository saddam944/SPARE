"""Low-level editor H_theta (MEND-style, as used by HiEdit).

Given the rank-1 gradient factors (u, v) of the edit loss with respect to a ``down_proj``
weight, the editor maps them to (u~, v~) and the update is

    dW = edit_sign * exp(log_edit_scale) * outer(v~, u~)  restricted to the FiNE-selected columns.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LowLevelEditor(nn.Module):
    def __init__(self, d_m, d_h, layer_ids, d_r=128, C=2, init_scale=1e-3, edit_sign=-1.0):
        super().__init__()
        self.d_m, self.d_h, self.C = d_m, d_h, C
        self.edit_sign = edit_sign
        d = d_m + d_h
        self.in_norm = nn.LayerNorm(d)
        self.A = nn.ModuleList([nn.Linear(d, d_r, bias=False) for _ in range(C)])
        self.B = nn.ModuleList([nn.Linear(d_r, d, bias=False) for _ in range(C)])
        for b in self.B:
            nn.init.zeros_(b.weight)
        # Layer-specific scale and offset for each residual block.
        self.scale, self.offset = nn.ParameterDict(), nn.ParameterDict()
        for l in layer_ids:
            for c in range(C):
                self.scale[f"{l}_{c}"] = nn.Parameter(torch.ones(d))
                self.offset[f"{l}_{c}"] = nn.Parameter(torch.zeros(d))
        self.log_edit_scale = nn.Parameter(torch.log(torch.tensor(float(init_scale))))

    def forward(self, u, v, layer_id):
        h = self.in_norm(torch.cat([u, v], dim=-1))
        for c in range(self.C):
            z = self.B[c](F.relu(self.A[c](h)))
            h = h + self.scale[f"{layer_id}_{c}"] * z + self.offset[f"{layer_id}_{c}"]
        return h[..., :self.d_m], h[..., self.d_m:]

    def coefficient(self):
        return self.edit_sign * torch.exp(self.log_edit_scale)

    def delta_w(self, u, v, layer_id, col_mask):
        ut, vt = self(u, v, layer_id)
        return self.coefficient() * torch.outer(vt, ut) * col_mask
