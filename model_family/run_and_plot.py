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
# Sweep configuration
# ------------------------------------------------------------
SEQ_LENGTHS    = [512, 1024, 2048, 4096, 8192]
MODEL_TYPES    = llm_kv.ALL_MODEL_TYPES   # ["gpt", "phi", "llama", "gemma", "opt"]
MAX_NEW_TOKENS = 10


# ------------------------------------------------------------
# Name / group mappings
# ------------------------------------------------------------
NAME_MAP = {
    "attn.q_proj":            "Q Projection",
    "attn.k_proj":            "K Projection",
    "attn.v_proj":            "V Projection",
    "attn.gqa_expand":        "GQA Expand",
    "attn.matmul_qk":         "QKᵀ MatMul",
    "attn.apply_causal_mask": "Causal Mask",
    "attn.softmax":           "Softmax",
    "attn.weighted_sum":      "Attention Weighted Sum",
    "attn.out_projection":    "Output Projection",
    "attn.cache_concat":      "KV Cache Concat",
    "block.norm1":            "LayerNorm (Pre-Attn)",
    "block.norm2":            "LayerNorm (Pre-FFN)",
    "ff.linear1":             "FFN Linear 1",
    "ff.gelu":                "GELU Activation",
    "ff.linear2":             "FFN Linear 2",
    "ff.swiglu":              "SwiGLU Activation",
    "model.proj_in":          "Extra Proj In",
    "model.proj_out":         "Extra Proj Out",
    "model.output_head":      "Output Head Projection",
}


def pretty_name(name: str) -> str:
    return NAME_MAP.get(name, name.replace(".", " ").replace("_", " ").title())


def group_block(name: str) -> str:
    if name.startswith("attn."):         return "Self-Attention"
    elif name.startswith("ff."):         return "Feed Forward Network"
    elif name.startswith("block.norm"):  return "Layer Normalization"
    elif name.startswith("model."):      return "Output Head / Extra Proj"
    else:                                return "Other"


# ------------------------------------------------------------
# Extract latency stats -> DataFrame
# ------------------------------------------------------------
def extract_stats(latency_recorder) -> pd.DataFrame:
    rows = []
    for name, stat in latency_recorder.data.items():
        count = int(stat["count"])
        total = float(stat["total"])
        avg   = total / count if count > 0 else 0.0
        rows.append({
            "name":        name,
            "pretty_name": pretty_name(name),
            "group":       group_block(name),
            "count":       count,
            "total":       total,
            "avg":         avg,
            "min":         float(stat["min"]),
            "max":         float(stat["max"]),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Run one benchmark configuration
# ------------------------------------------------------------
def run_one(model_type: str, seq_len: int):
    llm_kv.LATENCY = llm_kv.LatencyRecorder(
        use_cuda=__import__("torch").cuda.is_available()
    )
    llm_kv.MODEL_TYPE     = model_type
    llm_kv.SEQ_LEN        = seq_len
    llm_kv.MAX_NEW_TOKENS = MAX_NEW_TOKENS

    try:
        llm_kv.main()
    except Exception:
        print(f"  [FAIL] model={model_type} seq_len={seq_len}")
        traceback.print_exc()
        return None

    latency = getattr(llm_kv, "LATENCY", None)
    if latency is None or not latency.data:
        print(f"  [SKIP] No latency data — model={model_type} seq_len={seq_len}")
        return None

    return extract_stats(latency)


# ------------------------------------------------------------
# Per-run plots
# ------------------------------------------------------------
def plot_bar(df, out_path, title):
    df_plot = df.sort_values("avg", ascending=False)
    x = np.arange(len(df_plot))
    avg = df_plot["avg"].values
    lower_err = np.maximum(avg - df_plot["min"].values, 1e-12)
    upper_err = np.maximum(df_plot["max"].values - avg, 1e-12)

    plt.figure(figsize=(13, 6))
    plt.bar(x, avg)
    plt.errorbar(x, avg, yerr=[lower_err, upper_err], fmt="none", capsize=4)
    plt.xticks(x, df_plot["pretty_name"], rotation=45, ha="right")
    plt.ylabel("Latency (s) — Log Scale")
    plt.yscale("log")
    plt.title(title)
    plt.grid(axis="y", which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_pie(df, out_path, title):
    grouped = df.groupby("group")["total"].sum().sort_values(ascending=False)
    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"][:len(grouped)]
    explode = [0.06 if i == 0 else 0 for i in range(len(grouped))]

    plt.figure(figsize=(8, 8))
    wedges, _, autotexts = plt.pie(
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
    plt.title(title, fontsize=14, fontweight="bold", pad=20)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Cross-seq-length comparison plots (one model)
# ------------------------------------------------------------
def plot_comparison_avg(seq_dfs, out_path, title):
    records = []
    for seq_len, df in seq_dfs.items():
        for _, row in df.iterrows():
            records.append({"seq_len": seq_len, "pretty_name": row["pretty_name"], "avg": row["avg"]})
    wide = pd.DataFrame(records).pivot(index="seq_len", columns="pretty_name", values="avg")

    plt.figure(figsize=(13, 7))
    for col in wide.columns:
        plt.plot(wide.index, wide[col], marker="o", label=col)
    plt.xlabel("Sequence Length")
    plt.ylabel("Avg Latency (s) — Log Scale")
    plt.yscale("log")
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    plt.grid(which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def plot_comparison_pct(seq_dfs, out_path, title):
    records = []
    for seq_len, df in seq_dfs.items():
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
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, frameon=False)
    plt.grid(which="both", linestyle="--", alpha=0.4)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# ------------------------------------------------------------
# Cross-model comparison at a fixed seq_len
# ------------------------------------------------------------
def plot_model_comparison(results, seq_len, out_dir):
    rows = []
    for model_type in MODEL_TYPES:
        df = results.get((model_type, seq_len))
        if df is not None:
            rows.append({"model": model_type, "total_avg": df["avg"].sum()})
    if not rows:
        return

    df_bar = pd.DataFrame(rows).sort_values("total_avg", ascending=False)
    plt.figure(figsize=(8, 5))
    plt.bar(df_bar["model"], df_bar["total_avg"])
    plt.ylabel("Sum of Avg Latency (s)")
    plt.title(f"Total KV-Cache Latency by Model Family  (seq_len={seq_len})")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(out_dir, f"model_comparison_seq{seq_len}.png")
    plt.savefig(path, dpi=200)
    plt.close()
    print(f"  Saved model comparison -> {path}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    out_dir = "latency_results"
    os.makedirs(out_dir, exist_ok=True)

    results = {}  # (model_type, seq_len) -> DataFrame | None

    for model_type in MODEL_TYPES:
        model_dir = os.path.join(out_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)

        seq_dfs = {}  # seq_len -> df

        for seq_len in SEQ_LENGTHS:
            print(f"\n{'='*52}\n  model={model_type} | seq_len={seq_len}\n{'='*52}")

            df = run_one(model_type, seq_len)
            results[(model_type, seq_len)] = df

            if df is None:
                continue

            seq_dfs[seq_len] = df

            csv_path = os.path.join(model_dir, f"latency_seq{seq_len}.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Saved CSV  -> {csv_path}")

            suffix = f"\n({model_type} | seq_len={seq_len} | KV cache)"
            plot_bar(df,
                     os.path.join(model_dir, f"bar_seq{seq_len}.png"),
                     f"Transformer Op Latency{suffix}")
            plot_pie(df,
                     os.path.join(model_dir, f"pie_seq{seq_len}.png"),
                     f"Latency Distribution{suffix}")

        if len(seq_dfs) > 1:
            plot_comparison_avg(
                seq_dfs,
                os.path.join(model_dir, "comparison_avg_latency.png"),
                f"Avg Latency vs Seq Length — {model_type} (KV cache)",
            )
            plot_comparison_pct(
                seq_dfs,
                os.path.join(model_dir, "comparison_pct_latency.png"),
                f"Latency Share (%) vs Seq Length — {model_type} (KV cache)",
            )

    for seq_len in SEQ_LENGTHS:
        plot_model_comparison(results, seq_len, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
