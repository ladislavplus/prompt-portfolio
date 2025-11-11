import json, pandas as pd
from benchmark_runner import benchmark, plot_results, write_markdown_report

def compare_multiple_prompts(config_file, testset_path):
    with open(config_file) as f:
        prompt_files = json.load(f)
    all_results = []
    for pf in prompt_files:
        df = benchmark(pf, testset_path, mock=True)
        df["prompt_name"] = pf.split("/")[-1].replace(".json", "")
        all_results.append(df)
    combined = pd.concat(all_results)
    combined.to_csv("results/combined_comparison.csv", index=False)
    print("✅ Combined results saved to results/combined_comparison.csv")
    avg_df = combined.groupby("prompt_name")["similarity"].mean().reset_index()
    print("\n📊 Average similarity per prompt:")
    print(avg_df)
    return avg_df
import matplotlib.pyplot as plt

def plot_comparison(avg_df, output_path="media/prompt_comparison_chart.png"):
    plt.figure(figsize=(6, 4))
    plt.bar(avg_df["prompt_name"], avg_df["similarity"], color="lightgreen")
    plt.title("Average Similarity Comparison Across Prompts")
    plt.xlabel("Prompt Variant")
    plt.ylabel("Average Similarity (0–1)")
    plt.ylim(0, 1)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Comparison chart saved to {output_path}")

if __name__ == "__main__":
    avg_df = compare_multiple_prompts(
        "projects/prompt-comparison/prompts_to_compare.json",
        "prompts/benchmarks/translation_testset.json"
    )
    plot_comparison(avg_df)