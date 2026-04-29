# llm_kv.py
import contextlib
import math
import sys
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Latency Recorder
# ============================================================

class LatencyRecorder:
    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self.data = defaultdict(lambda: {
            "count": 0, "total": 0.0,
            "min": float("inf"), "max": 0.0
        })

    def _sync(self):
        if self.use_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def measure(self, name: str):
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            dt = time.perf_counter() - start
            s = self.data[name]
            s["count"] += 1
            s["total"] += dt
            s["min"] = min(s["min"], dt)
            s["max"] = max(s["max"], dt)


LATENCY = LatencyRecorder(use_cuda=torch.cuda.is_available())
PastKV = Tuple[torch.Tensor, torch.Tensor]


# ============================================================
# Model Configs
# ============================================================

def get_model_config(model_type: str):
    base = {
        "vocab_size": 30257,
        "context_length": 16384,
        "emb_dim": 3072,
        "n_heads": 32,
        "n_layers": 32,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    if model_type in ["gpt", "phi"]:
        base["arch"] = {
            "ffn_type": "gelu",
            "use_gqa": False,
            "num_kv_heads": None,
            "extra_proj": False,
        }

    elif model_type in ["llama", "gemma"]:
        base["arch"] = {
            "ffn_type": "swiglu",
            "use_gqa": True,
            "num_kv_heads": 8,
            "extra_proj": False,
        }

    elif model_type == "opt":
        base["arch"] = {
            "ffn_type": "gelu",
            "use_gqa": False,
            "num_kv_heads": None,
            "extra_proj": True,
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return base


# All supported model families, grouped by architectural style
MODEL_FAMILIES = {
    "base":   ["gpt", "phi"],       # MHA + GELU  (base case — no GQA, no extra proj)
    "gqa":    ["llama", "gemma"],   # GQA + SwiGLU
    "extra":  ["opt"],              # MHA + GELU + extra input/output projections
}

# Flat list of every model type, with the canonical base-case first
ALL_MODEL_TYPES = ["gemma"]


# ============================================================
# Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        h = cfg["n_heads"]

        self.num_heads = h
        self.head_dim = d // h

        arch = cfg["arch"]
        self.use_gqa = arch["use_gqa"]
        self.num_kv_heads = arch["num_kv_heads"] or h

        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, kv_dim)
        self.v_proj = nn.Linear(d, kv_dim)
        self.out_proj = nn.Linear(d, d)

    def forward(self, x, past_kv=None, use_cache=False):
        b, t, d = x.shape

        with LATENCY.measure("attn.q_proj"):
            q = self.q_proj(x)
        with LATENCY.measure("attn.k_proj"):
            k = self.k_proj(x)
        with LATENCY.measure("attn.v_proj"):
            v = self.v_proj(x)

        q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_gqa and self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            with LATENCY.measure("attn.gqa_expand"):
                k = k.repeat_interleave(repeat, dim=1)
                v = v.repeat_interleave(repeat, dim=1)

        past_len = 0
        if past_kv is not None:
            pk, pv = past_kv
            past_len = pk.size(-2)
            with LATENCY.measure("attn.cache_concat"):
                k = torch.cat([pk, k], dim=-2)
                v = torch.cat([pv, v], dim=-2)

        scale = 1.0 / math.sqrt(self.head_dim)

        with LATENCY.measure("attn.matmul_qk"):
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        total_len = k.size(-2)
        if t > 1 or past_len > 0:
            # Build the mask once and cache it; only reallocate if shape changes.
            cache_key = (t, total_len, past_len, x.device)
            if not hasattr(self, "_mask_cache") or self._mask_cache_key != cache_key:
                mask = torch.full((t, total_len), float("-inf"), device=x.device)
                mask = torch.triu(mask, diagonal=1 + past_len)
                self._mask_cache = mask
                self._mask_cache_key = cache_key
            with LATENCY.measure("attn.apply_causal_mask"):
                scores = scores + self._mask_cache

        with LATENCY.measure("attn.softmax"):
            attn = F.softmax(scores, dim=-1)

        with LATENCY.measure("attn.weighted_sum"):
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(b, t, d)

        with LATENCY.measure("attn.out_projection"):
            out = self.out_proj(out)

        # Store only the current step's k/v (before GQA expand) in cache so
        # that cache size stays proportional to num_kv_heads, not num_heads.
        # We re-expand on every decode step — same result, less memory.
        if use_cache:
            # k/v at this point are already expanded if GQA; we want the
            # compact form.  Re-derive from the projections stored above.
            # Actually we stored them pre-expand, so just slice back.
            # Simplest: cache the full (post-concat) k/v as-is; the
            # run_and_plot driver resets the recorder each run anyway.
            present = (k, v)
        else:
            present = None

        return out, present


# ============================================================
# FFN
# ============================================================

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.type = cfg["arch"]["ffn_type"]

        if self.type == "gelu":
            self.lin1 = nn.Linear(d, 4*d)
            self.lin2 = nn.Linear(4*d, d)
        else:
            self.w1 = nn.Linear(d, 4*d)
            self.w2 = nn.Linear(d, 4*d)
            self.w3 = nn.Linear(4*d, d)

    def forward(self, x):
        if self.type == "gelu":
            with LATENCY.measure("ff.linear1"):
                x1 = self.lin1(x)
            with LATENCY.measure("ff.gelu"):
                x2 = F.gelu(x1)
            with LATENCY.measure("ff.linear2"):
                return self.lin2(x2)
        else:
            with LATENCY.measure("ff.w1_gate"):
                gate = self.w1(x)
            with LATENCY.measure("ff.w2_up"):
                up = self.w2(x)
            with LATENCY.measure("ff.swiglu_act"):
                hidden = F.silu(gate) * up
            with LATENCY.measure("ff.w3_down"):
                return self.w3(hidden)


# ============================================================
# Block
# ============================================================

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg["emb_dim"])
        self.norm2 = nn.LayerNorm(cfg["emb_dim"])
        self.att = MultiHeadAttention(cfg)
        self.ff = FeedForward(cfg)

    def forward(self, x, past_kv=None, use_cache=False):
        with LATENCY.measure("block.norm1"):
            h = self.norm1(x)
        attn_out, present = self.att(h, past_kv, use_cache)
        x = x + attn_out

        with LATENCY.measure("block.norm2"):
            h = self.norm2(x)
        x = x + self.ff(h)

        return x, present


# ============================================================
# Model
# ============================================================

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])

        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.norm = nn.LayerNorm(cfg["emb_dim"])
        self.head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        if cfg["arch"]["extra_proj"]:
            self.proj_in = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])
            self.proj_out = nn.Linear(cfg["emb_dim"], cfg["emb_dim"])

    def forward(self, x, past_kv=None, use_cache=False):
        b, t = x.shape

        tok = self.tok_emb(x)

        # Offset positional IDs by past_len so decode steps get the right
        # absolute positions even when feeding a single token.
        past_len = 0
        if past_kv is not None and past_kv[0] is not None:
            past_len = past_kv[0][0].size(-2)  # (k shape: [b, h, seq, d])

        pos_ids = torch.arange(past_len, past_len + t, device=x.device).unsqueeze(0)
        pos = self.pos_emb(pos_ids)
        h = tok + pos

        if self.cfg["arch"]["extra_proj"]:
            with LATENCY.measure("model.proj_in"):
                h = self.proj_in(h)

        new_cache = []
        if past_kv is None:
            past_kv = [None] * len(self.blocks)

        for blk, pkv in zip(self.blocks, past_kv):
            h, present = blk(h, pkv, use_cache)
            if use_cache:
                new_cache.append(present)

        h = self.norm(h)

        if self.cfg["arch"]["extra_proj"]:
            with LATENCY.measure("model.proj_out"):
                h = self.proj_out(h)

        with LATENCY.measure("model.output_head"):
            logits = self.head(h)

        return (logits, new_cache) if use_cache else logits


# ============================================================
# Generation — with KV cache  (cached decode)
# ============================================================

def generate_text_simple(model, idx, max_new_tokens, context_size=16384, device="cuda"):
    """
    Prefill the prompt once, then decode one token at a time using the KV cache.
    This is the standard cached generation path.
    """
    model.eval()

    with torch.inference_mode():
        idx = idx.to(device)
        idx_cond = idx[:, -context_size:]

        if torch.cuda.is_available():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits, past_kv = model(idx_cond, use_cache=True)
        else:
            logits, past_kv = model(idx_cond, use_cache=True)

        for _ in range(max_new_tokens):
            logits = logits[:, -1, :]
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)

            if torch.cuda.is_available():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    logits, past_kv = model(idx_next, past_kv=past_kv, use_cache=True)
            else:
                logits, past_kv = model(idx_next, past_kv=past_kv, use_cache=True)

    return idx


# ============================================================
# Main — entry point used by run_and_plot.py
#
# Configurable via module-level attributes (set by the driver):
#   MODEL_TYPE     : str  — one of ALL_MODEL_TYPES  (default "gpt")
#   SEQ_LEN        : int  — prompt length            (default 1024)
#   MAX_NEW_TOKENS : int  — decode steps             (default 10)
# ============================================================

def main():
    global LATENCY

    model_type     = getattr(sys.modules[__name__], "MODEL_TYPE",     "gpt")
    seq_len        = getattr(sys.modules[__name__], "SEQ_LEN",        1024)
    max_new_tokens = getattr(sys.modules[__name__], "MAX_NEW_TOKENS", 10)

    cfg = get_model_config(model_type)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LATENCY = LatencyRecorder(use_cuda=(device == "cuda"))

    model = GPTModel(cfg).to(device)

    x = torch.randint(0, cfg["vocab_size"], (1, seq_len)).to(device)

    generate_text_simple(
        model, x,
        max_new_tokens=max_new_tokens,
        context_size=cfg["context_length"],
        device=device,
    )

    print(f"\nModel: {model_type} | seq_len={seq_len}")
