#!/usr/bin/env python3
"""
Hierarchical Spawn-and-Prune (HSAP) attention: compute-matched experiment on WikiText-103.

Trains four variants under an identical backbone and wall-clock budget:
  - vanilla            : standard decoder-only Transformer (nn.MultiheadAttention baseline)
  - vanilla_prune      : scalar head gates + L1 sparsity + pruning
  - scalar_prune_spawn : pruning + spawning of child heads
  - hsap_full          : pruning + spawning + Gaussian (RBF) specialization

Note on the baseline: the vanilla variant uses nn.MultiheadAttention while the three gated
variants use the custom monolithic attention below. The custom implementation was numerically
unstable under fp16 on the available hardware, so the reference implementation was used for the
baseline. This is a known, intentional implementation confound that applies only to baseline
comparisons; the inter-variant comparisons all share the custom implementation.
"""
from __future__ import annotations

import os
import math
import time
import json
import argparse
from dataclasses import dataclass, asdict, is_dataclass
from datetime import datetime
from typing import Dict, List, Optional
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler

import matplotlib.pyplot as plt

# Force the numerically stable "math" SDPA kernel for the custom attention.
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

# Minimum total attention heads across the whole model (global, not per-layer).
MIN_TOTAL_HEADS = 32


def _sdpa_compat_mask(attn_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    # nn.MultiheadAttention bool mask: True = block. F.scaled_dot_product_attention: True = allow.
    # We accept MHA-style masks and invert them for SDPA so the two paths agree.
    if attn_mask is None:
        return None
    if attn_mask.dtype == torch.bool:
        return ~attn_mask
    return attn_mask


@dataclass
class HSAPConfig:
    vocab_size: int = 32000
    d_model: int = 512
    d_ff: int = 2048
    num_layers: int = 12
    init_num_heads: int = 8
    max_heads_per_layer: int = 100
    max_seq_len: int = 512
    dropout: float = 0.1
    label_smoothing: float = 0.0

    lambda_sparse: float = 5e-4
    tau_prune: float = 0.3
    tau_spawn: float = 0.9
    num_child_heads: int = 1
    child_perturbation: float = 0.1
    child_gate_init: float = 0.1
    structural_step_interval: int = 10000

    sigma_init: float = 2.0
    sigma_min: float = 0.25
    sigma_shrink: float = 0.5

    weight_decay: float = 0.001
    grad_clip: float = 1.0
    pad_token_id: int = 0


EMBEDDED_TUNED_HPARAMS: Dict[str, Dict[str, float]] = {
    "vanilla": {
        "dropout": 0.004739217854190271, "grad_clip": 1.43, "label_smoothing": 0.14,
        "lr": 0.000984816013061787, "warmup_steps": 6000, "weight_decay": 0.008,
    },
    "vanilla_prune": {
        "dropout": 0.0107, "grad_clip": 0.80, "label_smoothing": 0.16,
        "lambda_sparse": 0.00025, "lr": 0.00096, "structural_step_interval": 24000,
        "tau_prune": 0.59, "warmup_steps": 6000, "weight_decay": 1.5e-05,
    },
    "scalar_prune_spawn": {
        "child_gate_init": 0.059, "child_perturbation": 0.050, "dropout": 0.094,
        "grad_clip": 0.88, "label_smoothing": 0.08, "lambda_sparse": 0.00036,
        "lr": 0.00057, "max_heads_per_layer": 16, "num_child_heads": 1,
        "structural_step_interval": 24000, "tau_prune": 0.59, "tau_spawn": 0.98,
        "warmup_steps": 6000, "weight_decay": 0.0008,
    },
    "hsap_full": {
        "child_gate_init": 0.079, "child_perturbation": 0.108, "dropout": 0.034,
        "grad_clip": 1.89, "label_smoothing": 0.010, "lambda_sparse": 0.00036,
        "lr": 0.00096, "max_heads_per_layer": 24, "num_child_heads": 3,
        "sigma_shrink": 0.69, "structural_step_interval": 16000,
        "tau_prune": 0.59, "tau_spawn": 0.98, "warmup_steps": 8000,
        "weight_decay": 2.6e-05,
    },
}


class ShrinkingScalarGatedMultiHeadSelfAttention(nn.Module):
    """Monolithic attention with per-head scalar gates. Supports pruning and spawning by
    resizing the packed QKV and output projections (real compute change, not masking)."""

    def __init__(self, d_model: int, init_num_heads: int, max_heads: int, dropout: float = 0.0,
                 gate_init: float = 0.5, enable_spawn: bool = False) -> None:
        super().__init__()
        self.d_model = d_model
        self.init_num_heads = init_num_heads
        self.max_heads = max_heads
        self.enable_spawn = enable_spawn
        self.d_head = d_model // init_num_heads
        self.num_heads = init_num_heads

        self.qkv_proj = nn.Linear(d_model, 3 * self.num_heads * self.d_head, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.d_head, d_model, bias=False)
        self.o_bias_per_head = nn.Parameter(torch.empty(self.num_heads, d_model))

        eps = 1e-4
        g = float(np.clip(gate_init, eps, 1.0 - eps))
        logit = math.log(g / (1.0 - g))
        self.gate_logit = nn.Parameter(torch.full((self.num_heads,), logit, dtype=torch.float32))

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        with torch.no_grad():
            bound = 1.0 / math.sqrt(self.d_head)
            self.o_bias_per_head.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.num_heads == 0:
            return torch.zeros_like(x)
        B, T, _ = x.size()
        H, d_h = self.num_heads, self.d_head

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([H * d_h, H * d_h, H * d_h], dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        if causal_mask is not None:
            out_tokens = F.scaled_dot_product_attention(q, k, v, attn_mask=_sdpa_compat_mask(causal_mask), dropout_p=0.0, is_causal=False)
        else:
            out_tokens = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        gate = torch.sigmoid(self.gate_logit).view(1, H, 1, 1)
        out_tokens = out_tokens * gate
        out_tokens = out_tokens.transpose(1, 2).contiguous().view(B, T, H * d_h)
        out_no_bias = self.o_proj(out_tokens)

        gate_coeff = gate.squeeze(-1).permute(0, 2, 1)
        bias_contrib = torch.einsum("bth,hd->btd", gate_coeff, self.o_bias_per_head)
        return self.dropout(out_no_bias + bias_contrib)

    def _rebuild(self, keep_indices: List[int]):
        H_new = len(keep_indices)
        if H_new == self.num_heads:
            return
        device = self.qkv_proj.weight.device
        dtype = self.qkv_proj.weight.dtype
        d_h = self.d_head

        w_qkv = self.qkv_proj.weight.data.view(3, self.num_heads, d_h, self.d_model)[:, keep_indices, :, :]
        b_qkv = self.qkv_proj.bias.data.view(3, self.num_heads, d_h)[:, keep_indices, :]
        w_o = self.o_proj.weight.data.view(self.d_model, self.num_heads, d_h)[:, keep_indices, :]

        new_qkv = nn.Linear(self.d_model, 3 * H_new * d_h, bias=True).to(device=device, dtype=dtype)
        new_qkv.weight.data.copy_(w_qkv.reshape(-1, self.d_model))
        new_qkv.bias.data.copy_(b_qkv.reshape(-1))

        new_o = nn.Linear(H_new * d_h, self.d_model, bias=False).to(device=device, dtype=dtype)
        new_o.weight.data.copy_(w_o.reshape(self.d_model, -1))

        self.qkv_proj = new_qkv
        self.o_proj = new_o
        self.o_bias_per_head = nn.Parameter(self.o_bias_per_head.data[keep_indices])
        self.gate_logit = nn.Parameter(self.gate_logit.data[keep_indices])
        self.num_heads = H_new

    def _append_child(self, parent_idx: int, perturb: float, gate_init: float):
        if self.num_heads >= self.max_heads:
            return False
        H_old = self.num_heads
        H_new = H_old + 1
        d_h = self.d_head
        dev = self.qkv_proj.weight.device

        w_qkv_new = torch.empty(3, H_new, d_h, self.d_model, device=dev)
        b_qkv_new = torch.empty(3, H_new, d_h, device=dev)
        w_o_new = torch.empty(self.d_model, H_new, d_h, device=dev)
        b_o_new = torch.empty(H_new, self.d_model, device=dev)
        g_new = torch.empty(H_new, device=dev)

        w_qkv_old = self.qkv_proj.weight.data.view(3, H_old, d_h, self.d_model)
        w_qkv_new[:, :H_old] = w_qkv_old
        b_qkv_new[:, :H_old] = self.qkv_proj.bias.data.view(3, H_old, d_h)
        w_o_new[:, :H_old] = self.o_proj.weight.data.view(self.d_model, H_old, d_h)
        b_o_new[:H_old] = self.o_bias_per_head.data
        g_new[:H_old] = self.gate_logit.data

        with torch.no_grad():
            w_qkv_new[:, H_old] = w_qkv_old[:, parent_idx] + perturb * torch.randn_like(w_qkv_old[:, parent_idx])
            b_qkv_new[:, H_old] = b_qkv_new[:, parent_idx] + perturb * torch.randn_like(b_qkv_new[:, parent_idx])
            w_o_new[:, H_old] = w_o_new[:, parent_idx] + perturb * torch.randn_like(w_o_new[:, parent_idx])
            b_o_new[H_old] = b_o_new[parent_idx] + perturb * torch.randn_like(b_o_new[parent_idx])
            eps = 1e-4
            gi = float(np.clip(gate_init, eps, 1.0 - eps))
            g_new[H_old] = math.log(gi / (1.0 - gi))

        self.qkv_proj = nn.Linear(self.d_model, 3 * H_new * d_h, bias=True).to(dev)
        self.qkv_proj.weight.data.copy_(w_qkv_new.reshape(-1, self.d_model))
        self.qkv_proj.bias.data.copy_(b_qkv_new.reshape(-1))
        self.o_proj = nn.Linear(H_new * d_h, self.d_model, bias=False).to(dev)
        self.o_proj.weight.data.copy_(w_o_new.reshape(self.d_model, -1))
        self.o_bias_per_head = nn.Parameter(b_o_new)
        self.gate_logit = nn.Parameter(g_new)
        self.num_heads = H_new
        return True

    @torch.no_grad()
    def structural_step(self, tau_prune, tau_spawn, num_child, max_h, perturb, g_init, spawn_budget: Optional[int] = None):
        if self.num_heads > 0:
            gates = torch.sigmoid(self.gate_logit)
            norm_g = gates / (gates.max() + 1e-8)
            keep = (norm_g >= tau_prune).nonzero().squeeze(1).tolist()
            if not keep:
                keep = [int(torch.argmax(gates).item())]
            self._rebuild(keep)

        spawned = 0
        if self.enable_spawn and self.num_heads < max_h and (spawn_budget is None or spawn_budget > 0):
            gates = torch.sigmoid(self.gate_logit)
            norm_g = gates / (gates.max() + 1e-8)
            parents = (norm_g >= tau_spawn).nonzero().squeeze(1).tolist()
            parents.sort(key=lambda i: float(norm_g[i]), reverse=True)
            for pid in parents:
                if self.num_heads >= max_h:
                    break
                if spawn_budget is not None and spawned >= spawn_budget:
                    break
                for _ in range(num_child):
                    if spawn_budget is not None and spawned >= spawn_budget:
                        break
                    if not self._append_child(pid, perturb, g_init):
                        break
                    spawned += 1
        return spawned


class ShrinkingGatedMultiHeadSelfAttention(ShrinkingScalarGatedMultiHeadSelfAttention):
    """Adds a token-dependent Gaussian (RBF) gate in query space on top of the scalar gate."""

    def __init__(self, d_model, init_num_heads, max_heads, dropout=0.0, gate_init=0.5,
                 sigma_init=2.0, sigma_min=0.25, sigma_shrink=0.5):
        super().__init__(d_model, init_num_heads, max_heads, dropout, gate_init, enable_spawn=True)
        self.sigma_min = sigma_min
        self.sigma_shrink = sigma_shrink
        self.center = nn.Parameter(torch.zeros(self.num_heads, self.d_head))
        self.log_sigma = nn.Parameter(torch.full((self.num_heads,), math.log(sigma_init)))

    def forward(self, x, causal_mask=None):
        if self.num_heads == 0:
            return torch.zeros_like(x)
        B, T, _ = x.size()
        H, d_h = self.num_heads, self.d_head

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([H * d_h, H * d_h, H * d_h], dim=-1)
        q = q.view(B, T, H, d_h).transpose(1, 2)
        k = k.view(B, T, H, d_h).transpose(1, 2)
        v = v.view(B, T, H, d_h).transpose(1, 2)

        if causal_mask is not None:
            out_tok = F.scaled_dot_product_attention(q, k, v, attn_mask=_sdpa_compat_mask(causal_mask), dropout_p=0.0, is_causal=False)
        else:
            out_tok = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

        q_cent = q - self.center.view(1, H, 1, d_h)
        dist2 = (q_cent ** 2).mean(dim=-1, keepdim=True)
        sig = torch.exp(self.log_sigma).clamp(min=self.sigma_min).view(1, H, 1, 1)
        rbf = torch.exp(-dist2 / (2 * sig ** 2))

        scalar_g = torch.sigmoid(self.gate_logit).view(1, H, 1, 1)
        combined_g = scalar_g * rbf

        out_tok = out_tok * combined_g
        out_tok = out_tok.transpose(1, 2).contiguous().view(B, T, H * d_h)
        out = self.o_proj(out_tok)

        g_coeff = combined_g.squeeze(-1).permute(0, 2, 1)
        bias = torch.einsum("bth,hd->btd", g_coeff, self.o_bias_per_head)
        return self.dropout(out + bias)

    def _rebuild(self, keep):
        super()._rebuild(keep)
        self.center = nn.Parameter(self.center.data[keep])
        self.log_sigma = nn.Parameter(self.log_sigma.data[keep])

    def _append_child(self, parent_idx, perturb, gate_init):
        if not super()._append_child(parent_idx, perturb, gate_init):
            return False
        H_new = self.num_heads
        H_old = H_new - 1
        dev = self.center.device

        c_new = torch.empty(H_new, self.d_head, device=dev)
        s_new = torch.empty(H_new, device=dev)
        c_new[:H_old] = self.center.data
        s_new[:H_old] = self.log_sigma.data

        with torch.no_grad():
            c_new[H_old] = self.center.data[parent_idx] + perturb * torch.randn_like(self.center.data[parent_idx])
            parent_sig = torch.exp(self.log_sigma.data[parent_idx])
            child_sig = max(parent_sig * self.sigma_shrink, self.sigma_min)
            s_new[H_old] = math.log(child_sig)

        self.center = nn.Parameter(c_new)
        self.log_sigma = nn.Parameter(s_new)
        return True


class VanillaTransformerLM(nn.Module):
    """Baseline decoder-only Transformer built on nn.MultiheadAttention (see module docstring)."""

    def __init__(self, cfg: HSAPConfig):
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        class Block(nn.Module):
            def __init__(self, c):
                super().__init__()
                self.ln1 = nn.LayerNorm(c.d_model)
                self.attn = nn.MultiheadAttention(c.d_model, c.init_num_heads, dropout=c.dropout, batch_first=True)
                self.ln2 = nn.LayerNorm(c.d_model)
                self.ff = nn.Sequential(
                    nn.Linear(c.d_model, c.d_ff), nn.GELU(), nn.Dropout(c.dropout),
                    nn.Linear(c.d_ff, c.d_model), nn.Dropout(c.dropout),
                )

            def forward(self, x, mask=None):
                h = self.ln1(x)
                a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
                x = x + a
                x = x + self.ff(self.ln2(x))
                return x

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        B, T = input_ids.size()
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        mask = torch.triu(torch.ones((T, T), device=input_ids.device, dtype=torch.bool), diagonal=1)
        for b in self.blocks:
            x = b(x, mask)
        return self.lm_head(self.ln_f(x))

    def gate_l1_loss(self):
        return 0.0


class MonolithicHSAPTransformerLM(nn.Module):
    def __init__(self, cfg: HSAPConfig, mode="hsap_full"):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList()
        for _ in range(cfg.num_layers):
            b = nn.Module()
            b.ln1 = nn.LayerNorm(cfg.d_model)
            b.ln2 = nn.LayerNorm(cfg.d_model)
            if mode == "hsap_full":
                b.attn = ShrinkingGatedMultiHeadSelfAttention(
                    cfg.d_model, cfg.init_num_heads, cfg.max_heads_per_layer, cfg.dropout,
                    sigma_init=cfg.sigma_init, sigma_min=cfg.sigma_min, sigma_shrink=cfg.sigma_shrink,
                )
            else:
                spawn = (mode == "scalar_prune_spawn")
                mh = cfg.max_heads_per_layer if spawn else cfg.init_num_heads
                b.attn = ShrinkingScalarGatedMultiHeadSelfAttention(
                    cfg.d_model, cfg.init_num_heads, mh, cfg.dropout, enable_spawn=spawn,
                )
            b.ff = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_ff), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_ff, cfg.d_model), nn.Dropout(cfg.dropout),
            )
            self.blocks.append(b)

        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight

    def forward(self, input_ids):
        B, T = input_ids.size()
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        mask = torch.triu(torch.ones((T, T), device=input_ids.device, dtype=torch.bool), diagonal=1)
        for b in self.blocks:
            x = x + b.attn(b.ln1(x), mask)
            x = x + b.ff(b.ln2(x))
        return self.lm_head(self.ln_f(x))

    def gate_l1_loss(self):
        loss = 0.0
        for b in self.blocks:
            loss += torch.sigmoid(b.attn.gate_logit).sum()
        return loss

    def total_heads(self) -> int:
        return sum(int(b.attn.num_heads) for b in self.blocks)

    def structural_step(self, cfg: HSAPConfig, min_total_heads: int = MIN_TOTAL_HEADS):
        # Prune + optional spawn, enforcing a global minimum total head count (not per-layer).
        heads_before_total = self.total_heads()
        if heads_before_total < min_total_heads:
            raise RuntimeError(
                f"Global head constraint violated before structural step: "
                f"total_heads={heads_before_total} < {min_total_heads}."
            )

        allowed_removals = max(0, heads_before_total - min_total_heads)

        # Pass 1: compute per-layer prune proposals and collect restore candidates.
        removed_total_proposed = 0
        layer_plan = []
        restore_candidates = []
        for li, b in enumerate(self.blocks):
            attn = b.attn
            H_before = int(attn.num_heads)
            if H_before <= 0:
                layer_plan.append({"li": li, "H_before": H_before, "keep_set": set(), "removed": [], "keep": []})
                continue
            with torch.no_grad():
                gates = torch.sigmoid(attn.gate_logit.detach())
                norm_g = gates / (gates.max() + 1e-8)
                keep = (norm_g >= cfg.tau_prune).nonzero().squeeze(1).tolist()
                if not keep:
                    keep = [int(torch.argmax(gates).item())]
            keep_set = set(int(k) for k in keep)
            removed = [h for h in range(H_before) if h not in keep_set]
            removed_total_proposed += len(removed)
            for h in removed:
                restore_candidates.append((float(norm_g[h].item()), li, int(h)))
            layer_plan.append({"li": li, "H_before": H_before, "keep_set": keep_set,
                               "removed": removed, "keep": sorted(keep_set)})

        # Pass 2: if pruning would drop below the global minimum, restore the strongest removed heads.
        if removed_total_proposed > allowed_removals:
            to_restore = removed_total_proposed - allowed_removals
            restore_candidates.sort(key=lambda t: t[0], reverse=True)
            for _, li, h in restore_candidates[:to_restore]:
                layer_plan[li]["keep_set"].add(h)
            removed_total = 0
            for info in layer_plan:
                H_before = info["H_before"]
                if H_before <= 0:
                    info["keep"], info["removed"] = [], []
                    continue
                keep_set = info["keep_set"]
                info["keep"] = sorted(keep_set)
                info["removed"] = [h for h in range(H_before) if h not in keep_set]
                removed_total += len(info["removed"])
        else:
            removed_total = removed_total_proposed

        # Apply pruning.
        for info in layer_plan:
            if info["H_before"] > 0 and len(info["keep"]) != info["H_before"]:
                self.blocks[info["li"]].attn._rebuild(info["keep"])

        # Spawn, budgeted by the number actually removed (keeps total head count balanced).
        spawn_budget = removed_total if self.mode in ("scalar_prune_spawn", "hsap_full") else None
        remaining_budget = spawn_budget
        spawned_total = 0
        for b in self.blocks:
            spawned = b.attn.structural_step(
                tau_prune=0.0,
                tau_spawn=cfg.tau_spawn,
                num_child=cfg.num_child_heads,
                max_h=cfg.max_heads_per_layer,
                perturb=cfg.child_perturbation,
                g_init=cfg.child_gate_init,
                spawn_budget=remaining_budget,
            )
            spawned_total += int(spawned)
            if remaining_budget is not None:
                remaining_budget = max(0, remaining_budget - int(spawned))

        heads_after_total = self.total_heads()
        if heads_after_total < min_total_heads:
            raise RuntimeError(
                f"Global head constraint violated after structural step: total_heads={heads_after_total}."
            )
        print(f"Structural step: removed {removed_total}, spawned {spawned_total}, total heads now {heads_after_total}.")


def get_dataloader(dataset_dir: str, split: str, batch_size: int):
    from datasets import load_from_disk
    from torch.utils.data import DataLoader
    ds = load_from_disk(dataset_dir)
    dataset = ds[split]
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"), drop_last=(split == "train"))


def get_lr(step, base_lr, warmup, t_start, decay_seconds):
    # Step-based warmup, then wall-clock cosine decay over decay_seconds (robust to throughput
    # changes from pruning/spawning). decay_seconds is coupled to the actual training duration.
    if step <= warmup:
        return base_lr * step / max(1, warmup)
    elapsed = time.time() - t_start
    progress = min(max(elapsed / max(1e-9, decay_seconds), 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def ntp_loss(logits, ids, pad_id, smooth):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = ids[:, 1:].contiguous()
    return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1),
                           ignore_index=pad_id, label_smoothing=smooth)


def apply_hparams(cfg, hparams, variant):
    c = deepcopy(cfg)
    for k, v in hparams.items():
        if hasattr(c, k):
            setattr(c, k, v)
    if variant == "vanilla_prune":
        c.tau_spawn = 1.1  # disable spawning
    return c


def make_run_dir(base):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base, f"run_{ts}")
    for s in ["weights", "stats", "plots"]:
        os.makedirs(os.path.join(path, s), exist_ok=True)
    return path


def train_variant(variant, base_cfg, hparams, data_dir, run_dir, hours, device):
    print(f"\n>>> Starting {variant}...")
    cfg = apply_hparams(base_cfg, hparams, variant)

    if variant == "vanilla":
        model = VanillaTransformerLM(cfg).to(device)
        total_heads = int(cfg.init_num_heads) * int(cfg.num_layers)
    else:
        model = MonolithicHSAPTransformerLM(cfg, mode=variant).to(device)
        total_heads = model.total_heads()
    if total_heads < MIN_TOTAL_HEADS:
        raise RuntimeError(f"Global head constraint violated at init for '{variant}': total_heads={total_heads}.")

    opt = torch.optim.AdamW(model.parameters(), lr=hparams["lr"], weight_decay=hparams["weight_decay"])
    scaler = GradScaler("cuda")

    loader = get_dataloader(data_dir, "train", 8)
    val_loader = get_dataloader(data_dir, "validation", 8)
    iter_dl = iter(loader)

    weights_dir = os.path.join(run_dir, "weights")
    stats_dir = os.path.join(run_dir, "stats")

    loss_log_path = os.path.join(stats_dir, f"{variant}_train_val_losses.txt")
    with open(loss_log_path, "w") as f:
        f.write("step\ttrain_loss\tval_loss\telapsed_hours\n")

    best_train_loss, best_train_step = float("inf"), 0
    best_val, best_val_step = float("inf"), None
    last_checkpoint_time = time.time()

    struct_interval = 0 if variant == "vanilla" else hparams.get("structural_step_interval", 0)
    decay_seconds = hours * 3600.0  # cosine decay horizon coupled to actual training duration

    step = 0
    t_start = time.time()
    limit = hours * 3600
    model.train()

    while (time.time() - t_start) < limit:
        step += 1
        try:
            batch = next(iter_dl)
        except StopIteration:
            iter_dl = iter(loader)
            batch = next(iter_dl)
        ids = batch["input_ids"].to(device)

        lr = get_lr(step, hparams["lr"], hparams["warmup_steps"], t_start, decay_seconds)
        for pg in opt.param_groups:
            pg["lr"] = lr

        with autocast("cuda"):
            logits = model(ids)
            if step % 2000 == 0 and (torch.isnan(logits).any() or torch.isinf(logits).any()):
                raise RuntimeError(f"NaN/Inf in logits at step {step}")
            loss = ntp_loss(logits, ids, cfg.pad_token_id, hparams["label_smoothing"])
            if variant != "vanilla":
                loss = loss + hparams.get("lambda_sparse", 0) * model.gate_l1_loss()

        loss_item = float(loss.detach().item())
        if loss_item < best_train_loss:
            best_train_loss, best_train_step = loss_item, step

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), hparams["grad_clip"])
        scaler.step(opt)
        scaler.update()

        if struct_interval > 0 and step > hparams["warmup_steps"] and (step - hparams["warmup_steps"]) % struct_interval == 0:
            print(f"[{variant}] Step {step}: structural update")
            model.structural_step(cfg)
            # Re-init the optimizer because the structural step resized parameter tensors,
            # which invalidates AdamW's stored moment buffers.
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=hparams["weight_decay"])

        now = time.time()
        if (now - last_checkpoint_time) >= 1800.0:
            last_checkpoint_time = now
            cfg_dict = asdict(cfg) if is_dataclass(cfg) else getattr(cfg, "__dict__", {})
            torch.save({
                "variant": variant, "cfg": cfg_dict, "hparams": hparams, "step": step,
                "elapsed_seconds": float(now - t_start),
                "model_state": model.state_dict(), "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
            }, os.path.join(weights_dir, f"{variant}_latest_30min.pt"))

        if step % 100 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Time: {(time.time()-t_start)/60:.1f}m")

        if step % 10000 == 0:
            train_loss_at_eval = loss_item
            model.eval()
            with torch.no_grad():
                v_loss, v_cnt = 0.0, 0
                for vb in val_loader:
                    v_ids = vb["input_ids"].to(device)
                    v_loss += ntp_loss(model(v_ids), v_ids, cfg.pad_token_id, 0.0).item()
                    v_cnt += 1
                avg_val = v_loss / max(1, v_cnt)
                print(f"VALIDATION: {avg_val:.6f} | LR: {lr:.6e}")
                with open(loss_log_path, "a") as f:
                    f.write(f"{step}\t{train_loss_at_eval:.6f}\t{avg_val:.6f}\t{(time.time()-t_start)/3600.0:.4f}\n")
                if avg_val < best_val:
                    best_val, best_val_step = avg_val, step
                    torch.save(model.state_dict(), os.path.join(weights_dir, f"{variant}_best.pt"))
            model.train()

    torch.save(model.state_dict(), os.path.join(weights_dir, f"{variant}_final.pt"))
    with open(os.path.join(stats_dir, "best_train_loss.txt"), "a") as f:
        f.write(f"{variant}\t{best_train_loss:.6f}\tstep={best_train_step}\n")

    history = {"best_val": best_val, "best_val_step": best_val_step}
    del model, opt, scaler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return best_val, history


def main():
    parser = argparse.ArgumentParser(description="HSAP compute-matched experiment on WikiText-103.")
    parser.add_argument("--dataset_dir", default="wikitext103_spm32k_512packed")
    parser.add_argument("--out_dir", default="hsap_results")
    parser.add_argument("--hours_per_model", type=float, default=6.0)
    parser.add_argument("--variants", nargs="+",
                        default=["vanilla", "vanilla_prune", "scalar_prune_spawn", "hsap_full"],
                        help="Which variants to train, in order. Each runs for --hours_per_model.")
    args = parser.parse_args()

    run_dir = make_run_dir(args.out_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_cfg = HSAPConfig()

    results = {}
    for v in args.variants:
        val, hist = train_variant(v, base_cfg, EMBEDDED_TUNED_HPARAMS[v],
                                   args.dataset_dir, run_dir, args.hours_per_model, device)
        results[v] = {"best_val": val, "history": hist}

    with open(os.path.join(run_dir, "stats", "final_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nDone. Results saved to", run_dir)


if __name__ == "__main__":
    main()
