# run_and_plot.py
import os
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch._dynamo
torch._dynamo.config.suppress_errors = True

sys.path.insert(0, os.path.abspath("."))

try:
    import llm_kv
except Exception:
    print("Failed to import llm module.")
    traceback.print_exc()
    sys.exit(1)


# ------------------------------------------------------------
# Sequence lengths to benchmark
# ------------------------------------------------------------
SEQ_LENGTHS = [512, 1024, 2048, 4096, 8192]


# ------------------------------------------------------------
# Name mapping
# ------------------------------------------------------------
NAME_MAP = {
    "attn.qkv_projection": "QKV Projection",
    "attn.matmul_qk": "QKᵀ MatMul",
    "attn.apply_causal_mask": "Causal Mask",
    "attn.softmax": "Softmax",
    "attn.weighted_sum": "Attention Weighted Sum",
    "attn.out_projection": "Output Projection",
    "attn.cache_concat": "KV Cache Concat",
    "block.norm1": "LayerNorm (Pre-Attention)",
    "block.norm2": "LayerNorm (Pre-FFN)",
    "ff.linear1": "FFN Linear 1",
    "ff.gelu": "GELU Activation",
    "ff.linear2": "FFN Linear 2",
    "model.output_head": "Output Head Projection",
}


def pretty_name(name: str) -> str:
    return NAME_MAP.get(name, name.replace(".", " ").replace("_", " ").title())


# ------------------------------------------------------------
# Extract latency stats into a DataFrame
# ------------------------------------------------------------
def extract_stats_from_latency(latency_recorder) -> pd.DataFrame:
    rows = []
    for name, stat in latency_recorder.data.items():
        count = int(stat.get("count", 0))
        total = float(stat.get("total", 0.0))
        minv  = float(stat.get("min", 0.0))
        maxv  = float(stat.get("max", 0.0))
        avg   = total / count if count > 0 else 0.0
        rows.append({
            "name":        name,
            "pretty_name": pretty_name(name),
            "count":       count,
            "total":       total,
            "avg":         avg,
            "min":         minv,
            "max":         maxv,
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Per-run plots (log bar + pie) — same as before
# ------------------------------------------------------------
def plot_latency_logscale(df: pd.DataFrame, out_path: str, title_suffix: str = ""):
    df_plot = df.sort_values("avg", ascending=False)
    x = np.arange(len(df_plot))
    avg  = df_plot["avg"].values
    mins = df_plot["min"].values
    maxs = df_plot["max"].values
    lower_err = np.maximum(avg - mins, 1e-12)
    upper_err = np.maximum(maxs - avg, 1e-12)

    plt.figure(figsize=(12, 6))
    plt.bar(x, avg)
    plt.errorbar(x, avg, yerr=[lower_err, upper_err], fmt="none", capsize=4)
    plt.xticks(x, df_plot["pretty_name"], rotation=45, ha="right")
    plt.ylabel("Latency (seconds) — Log Scale")
    plt.yscale("log")
    plt.title(f"Transformer Operation Latency{title_suffix}")
    plt.grid(axis="y", which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"  Saved bar chart  → {out_path}")


def plot_latency_pie(df: pd.DataFrame, out_path: str, title_suffix: str = ""):
    def group_block(name: str) -> str:
        if name.startswith("attn."):       return "Self-Attention"
        elif name.startswith("ff."):       return "Feed Forward Network"
        elif name.startswith("block.norm"): return "Layer Normalization"
        elif name.startswith("model."):    return "Output Head"
        else:                              return "Other"

    df_g = df.copy()
    df_g["block"] = df_g["name"].apply(group_block)
    grouped = df_g.groupby("block")["total"].sum().sort_values(ascending=False)

    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"][:len(grouped)]
    explode = [0.06 if i == 0 else 0 for i in range(len(grouped))]

    plt.figure(figsize=(8, 8))
    wedges, texts, autotexts = plt.pie(
        grouped.values, labels=None, colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90, counterclock=False, explode=explode,
        wedgeprops=dict(edgecolor="white", linewidth=1.2), pctdistance=0.75,
    )
    for at in autotexts:
        at.set_fontsize(11); at.set_fontweight("bold"); at.set_color("white")
    plt.legend(wedges, grouped.index.tolist(), title="Transformer Components",
               loc="center left", bbox_to_anchor=(1, 0.5),
               fontsize=11, title_fontsize=12, frameon=False)
    plt.title(f"Latency Distribution{title_suffix}", fontsize=15,
              fontweight="bold", pad=20)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved pie chart  → {out_path}")

# ------------------------------------------------------------
# Comparison plot — latency share (%) vs sequence length per op
# ------------------------------------------------------------
def plot_comparison_pct(all_dfs: dict, out_dir: str):
    """Line plot: x = seq_len, y = each op's % share of total avg latency."""
    records = []
    for seq_len, df in all_dfs.items():
        total = df["avg"].sum()
        for _, row in df.iterrows():
            records.append({
                "seq_len":     seq_len,
                "pretty_name": row["pretty_name"],
                "pct":         (row["avg"] / total * 100) if total > 0 else 0.0,
            })

    wide = pd.DataFrame(records).pivot(index="seq_len", columns="pretty_name", values="pct")

    plt.figure(figsize=(13, 7))
    for col in wide.columns:
        plt.plot(wide.index, wide[col], marker="o", label=col)

    plt.xlabel("Sequence Length")
    plt.ylabel("% Share of Total Avg Latency")
    plt.title("Latency Share (%) vs Sequence Length per Operation")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    plt.grid(which="both", linestyle="--", alpha=0.4)
    plt.ylim(0, 100)
    plt.tight_layout()

    path = os.path.join(out_dir, "comparison_pct_latency.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved % comparison plot → {path}")
# ------------------------------------------------------------
# Comparison plot — avg latency vs sequence length per op
# ------------------------------------------------------------
def plot_comparison(all_dfs: dict, out_dir: str):
    """Line plot: x = seq_len, y = avg latency, one line per operation."""
    # Build a wide DataFrame: index=seq_len, columns=pretty_name
    records = []
    for seq_len, df in all_dfs.items():
        for _, row in df.iterrows():
            records.append({"seq_len": seq_len, "pretty_name": row["pretty_name"], "avg": row["avg"]})
    wide = pd.DataFrame(records).pivot(index="seq_len", columns="pretty_name", values="avg")

    plt.figure(figsize=(13, 7))
    for col in wide.columns:
        plt.plot(wide.index, wide[col], marker="o", label=col)
    plt.xlabel("Sequence Length")
    plt.ylabel("Avg Latency (seconds) — Log Scale")
    plt.yscale("log")
    plt.title("Avg Latency vs Sequence Length per Operation")
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    plt.grid(which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_avg_latency.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison plot → {path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    out_dir = "latency_results"
    os.makedirs(out_dir, exist_ok=True)

    all_dfs = {}   # seq_len -> DataFrame
   # MODEL_TYPES = ["phi", "llama", "gemma", "opt"]
    MODEL_TYPES = ["gemma", "opt"]
    for model_type in MODEL_TYPES:
        print(f"\n{'#'*60}")
        print(f" MODEL: {model_type.upper()}")
        print(f"{'#'*60}")

        model_dir = os.path.join(out_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)

        all_dfs = {}

        for seq_len in SEQ_LENGTHS:
            print(f"\n--- seq_len = {seq_len} ---")

            llm_kv.LATENCY = llm_kv.LatencyRecorder(
                use_cuda=__import__("torch").cuda.is_available()
            )

            llm_kv.SEQ_LEN = seq_len
            llm_kv.MODEL_TYPE = model_type

            llm_kv.main()

            LATENCY = llm_kv.LATENCY
            df = extract_stats_from_latency(LATENCY)
            all_dfs[seq_len] = df

            df.to_csv(os.path.join(model_dir, f"{model_type}_seq{seq_len}.csv"), index=False)

        if len(all_dfs) > 1:
            plot_comparison(all_dfs, model_dir)
            plot_comparison_pct(all_dfs, model_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
