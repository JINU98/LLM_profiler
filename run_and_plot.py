# run_and_plot.py
"""
Run llm.main(), extract LATENCY measurements, save CSV, and save:
 - log-scaled bar chart
 - pie chart of total latency share

Names are mapped to standard Transformer terminology.
"""

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
# 🔹 STANDARD TRANSFORMER NAME MAPPING
# ------------------------------------------------------------
NAME_MAP = {
    "attn.qkv_projection": "QKV Projection",
    "attn.matmul_qk": "QKᵀ MatMul",
    "attn.apply_causal_mask": "Causal Mask",
    "attn.softmax": "Softmax",
    "attn.weighted_sum": "Attention Weighted Sum",
    "attn.out_projection": "Output Projection",
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
# Extract Latency Data
# ------------------------------------------------------------
def extract_stats_from_latency(latency_recorder) -> pd.DataFrame:
    rows = []
    for name, stat in latency_recorder.data.items():
        count = int(stat.get("count", 0))
        total = float(stat.get("total", 0.0))
        minv = float(stat.get("min", 0.0))
        maxv = float(stat.get("max", 0.0))
        avg = total / count if count > 0 else 0.0

        rows.append({
            "name": name,
            "pretty_name": pretty_name(name),
            "count": count,
            "total": total,
            "avg": avg,
            "min": minv,
            "max": maxv
        })

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Log-Scale Bar Chart
# ------------------------------------------------------------
def plot_latency_logscale(df: pd.DataFrame, out_path: str):
    df_plot = df.sort_values("avg", ascending=False)

    x = np.arange(len(df_plot))
    avg = df_plot["avg"].values
    mins = df_plot["min"].values
    maxs = df_plot["max"].values

    lower_err = np.maximum(avg - mins, 1e-12)
    upper_err = np.maximum(maxs - avg, 1e-12)

    plt.figure(figsize=(12, 6))
    plt.bar(x, avg)
    plt.errorbar(x, avg, yerr=[lower_err, upper_err], fmt='none', capsize=4)

    plt.xticks(x, df_plot["pretty_name"], rotation=45, ha="right")
    plt.ylabel("Latency (seconds) — Log Scale")
    plt.yscale("log")
    plt.title("Transformer Operation Latency")
    plt.grid(axis="y", which="both", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved bar chart → {out_path}")


# ------------------------------------------------------------
# Pie Chart (Total Latency Share)
# ------------------------------------------------------------
def plot_latency_pie(df: pd.DataFrame, out_path: str):
    """
    Publication-quality pie chart grouped by Transformer block:
    - Self-Attention
    - Feed Forward Network (FFN)
    - LayerNorm
    - Output Head
    """

    # --------------------------------------------------
    # 1️⃣  GROUP OPERATIONS INTO HIGH-LEVEL BLOCKS
    # --------------------------------------------------
    def group_block(name: str) -> str:
        if name.startswith("attn."):
            return "Self-Attention"
        elif name.startswith("ff."):
            return "Feed Forward Network"
        elif name.startswith("block.norm"):
            return "Layer Normalization"
        elif name.startswith("model.output_head"):
            return "Output Head"
        else:
            return "Other"

    df_grouped = df.copy()
    df_grouped["block"] = df_grouped["name"].apply(group_block)

    grouped = (
        df_grouped
        .groupby("block")["total"]
        .sum()
        .sort_values(ascending=False)
    )

    sizes = grouped.values
    labels = grouped.index.tolist()

    # --------------------------------------------------
    # 2️⃣  PROFESSIONAL COLOR PALETTE
    # --------------------------------------------------
    colors = [
        "#4C72B0",  # blue (attention)
        "#DD8452",  # orange (ffn)
        "#55A868",  # green (norm)
        "#C44E52",  # red (output)
        "#8172B3"   # fallback
    ][:len(sizes)]

    # Slightly explode largest slice
    explode = [0.06 if i == 0 else 0 for i in range(len(sizes))]

    # --------------------------------------------------
    # 3️⃣  DRAW FIGURE
    # --------------------------------------------------
    plt.figure(figsize=(8, 8))

    wedges, texts, autotexts = plt.pie(
        sizes,
        labels=None,  # keep slices clean
        colors=colors,
        autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
        startangle=90,
        counterclock=False,
        explode=explode,
        wedgeprops=dict(edgecolor="white", linewidth=1.2),
        pctdistance=0.75
    )

    # Make percentage text bold and readable
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight("bold")
        autotext.set_color("white")

    # --------------------------------------------------
    # 4️⃣  LEGEND
    # --------------------------------------------------
    plt.legend(
        wedges,
        labels,
        title="Transformer Components",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=11,
        title_fontsize=12,
        frameon=False
    )

    # --------------------------------------------------
    # 5️⃣  TITLE
    # --------------------------------------------------
    plt.title(
        "Latency Distribution Across Transformer Blocks",
        fontsize=15,
        fontweight="bold",
        pad=20
    )

    plt.axis("equal")
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved polished pie chart → {out_path}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("Running llm.main()...")
    try:
        llm_kv.main()
    except Exception:
        print("llm.main() raised an exception.")
        traceback.print_exc()

    LATENCY = getattr(llm_kv, "LATENCY", None)
    if LATENCY is None:
        print("LATENCY not found.")
        sys.exit(1)

    df = extract_stats_from_latency(LATENCY)
    if df.empty:
        print("No latency data recorded.")
        sys.exit(1)

    os.makedirs("latency_results", exist_ok=True)

    df.to_csv("latency_results/latency_report.csv", index=False)
    print("Saved CSV → latency_results/latency_report.csv")

    plot_latency_logscale(df, "latency_results/latency_log_bar.png")
    plot_latency_pie(df, "latency_results/latency_pie.png")

    print("\nTop by average latency:")
    print(df.sort_values("avg", ascending=False)[
        ["pretty_name", "avg"]
    ].to_string(index=False, float_format="{:.6e}".format))


if __name__ == "__main__":
    main()