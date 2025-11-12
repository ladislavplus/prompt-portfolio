import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "reports" / "prompt_eval_results.csv"
OUT = ROOT / "media" / "prompt_eval_chart.png"

def visualize():
    df = pd.read_csv(IN)
    grouped = df.groupby(["task", "model"]).mean().reset_index()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for model in grouped["model"].unique():
        subset = grouped[grouped["model"] == model]
        ax.bar(
            subset["task"] + " (" + model + ")", 
            subset["avg_score"], 
            label=model, 
            alpha=0.7
        )

    ax.set_title("Prompt Evaluation Scores by Task and Model", fontsize=14, weight="bold")
    ax.set_ylabel("Average Score (0–5)")
    ax.set_ylim(0, 5)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT, dpi=200)
    plt.close()
    print(f"✅ Visualization saved -> {OUT}")

if __name__ == "__main__":
    visualize()
