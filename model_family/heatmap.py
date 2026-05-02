import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
seq_len = 8192
data = []

for model in ["phi", "llama", "gpt"]:
    df = pd.read_csv(f"./latency_results/{model}/latency_seq{seq_len}.csv")
    total = df["total"].sum()

    for _, row in df.iterrows():
        data.append({
            "model": model,
            "component": row["pretty_name"],
            "pct": row["total"] / total * 100
        })

df = pd.DataFrame(data)
pivot = df.pivot(index="component", columns="model", values="pct")
pivot = pivot.fillna(0)
plt.figure(figsize=(10,8))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis")
# plt.title("Latency % Distribution Across Models")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=300)