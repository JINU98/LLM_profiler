import contextlib
import math
import sys
import time
from collections import defaultdict
from typing import List, Optional, Tuple

import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# Simple latency recorder
class LatencyRecorder:
    def __init__(self, use_cuda: bool):
        self.use_cuda = use_cuda
        self.data = defaultdict(lambda: {"count": 0, "total": 0.0, "min": float("inf"), "max": 0.0})

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
            delta = time.perf_counter() - start
            stat = self.data[name]
            stat["count"] += 1
            stat["total"] += delta
            stat["min"] = min(stat["min"], delta)
            stat["max"] = max(stat["max"], delta)

    def report(self):
        lines = ["Latency Report (seconds):"]
        for name, stat in sorted(self.data.items(), key=lambda x: x[0]):
            count = stat["count"]
            total = stat["total"]
            avg = total / count if count > 0 else 0.0
            lines.append(
                f"  {name:40s} | count={count:4d} | total={total:.6f} | avg={avg:.6f} | min={stat['min']:.6f} | max={stat['max']:.6f}"
            )
        return "\n".join(lines)


LATENCY = LatencyRecorder(use_cuda=torch.cuda.is_available())


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(
    txt,
    batch_size=4,
    max_length=256,
    stride=128,
    shuffle=True,
    drop_last=True,
    num_workers=4,
    pin_memory=True,
):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


PastKV = Tuple[torch.Tensor, torch.Tensor]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, dropout, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.qkv = nn.Linear(d_in, d_out * 3, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout_p = dropout

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[PastKV] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PastKV]]:
        b, seq_len, _ = x.shape

        with LATENCY.measure("attn.qkv_projection"):
            qkv = self.qkv(x)
        qkv = qkv.reshape(b, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        past_len = 0
        if past_kv is not None:
            past_k, past_v = past_kv
            past_len = past_k.size(-2)
            with LATENCY.measure("attn.cache_concat"):
                k = torch.cat([past_k, k], dim=-2)
                v = torch.cat([past_v, v], dim=-2)

        scale = 1.0 / math.sqrt(self.head_dim)
        with LATENCY.measure("attn.matmul_qk"):
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale

        total_k_len = k.size(-2)
        if seq_len > 1 or past_len > 0:
            key_positions = torch.arange(total_k_len, device=x.device)
            query_positions = past_len + torch.arange(seq_len, device=x.device)
            causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
            if causal_mask.any():
                with LATENCY.measure("attn.apply_causal_mask"):
                    attn_scores = attn_scores.masked_fill(
                        causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )

        with LATENCY.measure("attn.softmax"):
            attn_weights = F.softmax(attn_scores, dim=-1)

        with LATENCY.measure("attn.weighted_sum"):
            attn_out = torch.matmul(attn_weights, v)

        attn_out = attn_out.permute(0, 2, 1, 3).contiguous()
        attn_out = attn_out.view(b, seq_len, self.d_out)

        with LATENCY.measure("attn.out_projection"):
            out = self.out_proj(attn_out)

        present = (k, v) if use_cache else None
        return out, present


class FeedForward(nn.Module):
    def __init__(self, emb_dim, drop_rate):
        super().__init__()
        self.lin1 = nn.Linear(emb_dim, 4*emb_dim)
        self.act = nn.GELU()
        self.lin2 = nn.Linear(4*emb_dim, emb_dim)
        self.drop_rate = drop_rate

    def forward(self, x):
        with LATENCY.measure("ff.linear1"):
            x1 = self.lin1(x)
        with LATENCY.measure("ff.gelu"):
            x2 = self.act(x1)
        with LATENCY.measure("ff.linear2"):
            x3 = self.lin2(x2)
        return x3


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg["emb_dim"], eps=1e-5)
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
        )
        self.norm2 = nn.LayerNorm(cfg["emb_dim"], eps=1e-5)
        self.ff = FeedForward(cfg["emb_dim"], cfg["drop_rate"])

    def forward(
        self,
        x: torch.Tensor,
        past_kv: Optional[PastKV] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[PastKV]]:
        shortcut = x
        with LATENCY.measure("block.norm1"):
            x = self.norm1(x)
        x, present = self.att(x, past_kv=past_kv, use_cache=use_cache)
        x = x + shortcut

        shortcut = x
        with LATENCY.measure("block.norm2"):
            x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut
        return x, present


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg["n_layers"])] )
        self.final_norm = nn.LayerNorm(cfg["emb_dim"], eps=1e-5)
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(
        self,
        in_idx: torch.Tensor,
        past_kv: Optional[List[Optional[PastKV]]] = None,
        use_cache: bool = False,
    ):
        batch_size, seq_len = in_idx.shape
        num_layers = len(self.trf_blocks)

        if past_kv is None:
            past_kv = [None] * num_layers
            past_length = 0
        else:
            if len(past_kv) != num_layers:
                raise ValueError(f"past_kv must have {num_layers} entries")
            past_length = 0
            for layer_cache in past_kv:
                if layer_cache is not None:
                    past_length = layer_cache[0].size(-2)
                    break

        if past_length + seq_len > self.cfg["context_length"]:
            raise ValueError(
                f"sequence length {past_length + seq_len} exceeds context_length {self.cfg['context_length']}"
            )

        tok_embeds = self.tok_emb(in_idx)
        pos_idx = torch.arange(past_length, past_length + seq_len, device=in_idx.device, dtype=torch.long)
        pos_embeds = self.pos_emb(pos_idx).unsqueeze(0)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        new_past_kv: List[Optional[PastKV]] = []
        for block, layer_past in zip(self.trf_blocks, past_kv):
            x, layer_present = block(x, past_kv=layer_past, use_cache=use_cache)
            if use_cache:
                new_past_kv.append(layer_present)

        x = self.final_norm(x)
        with LATENCY.measure("model.output_head"):
            logits = self.out_head(x)

        if use_cache:
            return logits, new_past_kv
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size, device="cuda"):
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


def main():
    global LATENCY
    GPT_CONFIG_124M = {
        "vocab_size": 30257,
        "context_length": 16348,
        "emb_dim": 3072,
        "n_heads": 32,
        "n_layers": 32,
        "drop_rate": 0.1,
        "qkv_bias": False,
    }

    torch.manual_seed(123)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    LATENCY = LatencyRecorder(use_cuda=(device == "cuda"))

    model = GPTModel(GPT_CONFIG_124M).to(device)
    model.eval()

    import sys
    seq_len = getattr(sys.modules[__name__], "SEQ_LEN", 1024)
    assert seq_len <= GPT_CONFIG_124M["context_length"], \
        f"seq_len {seq_len} exceeds context_length {GPT_CONFIG_124M['context_length']}"

    encoded_tensor = torch.randint(
        0, GPT_CONFIG_124M["vocab_size"], (1, seq_len), dtype=torch.long
    ).to(device)
    print("here", encoded_tensor.shape)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M["context_length"],
        device=device,
    )
    tokenizer = tiktoken.get_encoding("gpt2")
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    # print("Output text:", decoded_text)
    print("\n" + LATENCY.report())


if __name__ == "__main__":
    main()

