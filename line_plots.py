# plot_from_csv.py
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


out_dir = "latency_results"


import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9A6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def plot_comparison(all_dfs: dict, out_dir: str):
    records = []
    for seq_len, df in all_dfs.items():
        for _, row in df.iterrows():
            records.append({"seq_len": seq_len, "pretty_name": row["pretty_name"], "avg": row["avg"]})
    wide = pd.DataFrame(records).pivot(index="seq_len", columns="pretty_name", values="avg")

    plt.figure(figsize=(18, 9))
    for col, color in zip(wide.columns, COLORS):
        plt.plot(wide.index, wide[col], marker="o", label=col, color=color, linewidth=2)
    plt.xlabel("Sequence Length", fontsize=25)
    plt.ylabel("Avg Latency (seconds) — Log Scale", fontsize=25)
    plt.yscale("log")
    plt.title("Avg Latency vs Sequence Length per Operation", fontsize=25)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=25, frameon=False)
    plt.grid(which="both", linestyle="--", alpha=0.4)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_avg_latency.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved avg latency plot → {path}")


def plot_comparison_pct(all_dfs: dict, out_dir: str):
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

    plt.figure(figsize=(18, 9))
    for col, color in zip(wide.columns, COLORS):
        plt.plot(wide.index, wide[col], marker="o", label=col, color=color, linewidth=2)
    plt.xlabel("Sequence Length", fontsize=25)
    plt.ylabel("% Share of Total Avg Latency", fontsize=25)
    plt.title("Latency Share (%) vs Sequence Length per Operation", fontsize=25)
    plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=25, frameon=False)
    plt.grid(which="both", linestyle="--", alpha=0.4)
    plt.ylim(0, 100)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_pct_latency.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved % latency plot   → {path}")


def load_csvs(csv_dir: str) -> dict:
    """Load all latency_seqlen_*.csv files, return {seq_len: DataFrame}."""
    all_dfs = {}
    pattern = os.path.join(csv_dir, "latency_seqlen_*.csv")
    paths = sorted(glob.glob(pattern))

    if not paths:
        print(f"No CSV files matching '{pattern}' found.")
        return all_dfs

    for path in paths:
        # Extract seq_len from filename, e.g. latency_seqlen_1024.csv -> 1024
        basename = os.path.basename(path)
        try:
            seq_len = int(basename.replace("latency_seqlen_", "").replace(".csv", ""))
        except ValueError:
            print(f"  Skipping unrecognised filename: {basename}")
            continue

        df = pd.read_csv(path)
        all_dfs[seq_len] = df
        print(f"  Loaded seq_len={seq_len:6d} ← {path}")

    return all_dfs


if __name__ == "__main__":
    print(f"Loading CSVs from '{out_dir}' ...")
    all_dfs = load_csvs(out_dir)

    if len(all_dfs) < 2:
        print("Need at least 2 sequence lengths to plot a comparison.")
    else:
        os.makedirs(out_dir, exist_ok=True)
        plot_comparison(all_dfs, out_dir)
        plot_comparison_pct(all_dfs, out_dir)
        print("Done.")