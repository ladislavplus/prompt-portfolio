import json
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from benchmark_runner import benchmark
import os
from datetime import datetime

def compare_multiple_prompts(project_file, benchmark_file, model_alias, output_dir="results"):
    """
    Compares multiple prompts against a single benchmark and model.
    """
    with open(project_file) as f:
        project_data = json.load(f)
    
    prompt_files = project_data["prompts"]
    all_results = []
    
    print(f"🚀 Comparing {len(prompt_files)} prompts from project '{project_data['name']}'...")

    for prompt_file in prompt_files:
        # The benchmark function now returns a tuple: (df, prompt_name, test_name)
        result = benchmark(
            prompt_file=prompt_file,
            testset_file=benchmark_file,
            model_alias=model_alias,
            output_dir=output_dir
        )
        if result:
            df, prompt_name, test_name = result
            df["prompt_name"] = prompt_name
            all_results.append(df)

    if not all_results:
        print("❌ No results were generated from any prompt. Exiting.")
        return None

    combined = pd.concat(all_results)
    
    # Save combined results
    project_name_slug = project_data['name'].replace('_', '-')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_csv_filename = f"comparison_{project_name_slug}_{timestamp}.csv"
    combined_csv_path = os.path.join(output_dir, combined_csv_filename)
    combined.to_csv(combined_csv_path, index=False)
    print(f"✅ Combined results for all prompts saved to {combined_csv_path}")

    avg_df = combined.groupby("prompt_name")["similarity"].mean().reset_index()
    
    print("\n📊 Average similarity per prompt:")
    print(avg_df.to_string(index=False))
    
    return avg_df, project_data['name']

def plot_comparison(avg_df, project_name, output_dir="media"):
    """
    Generates and saves a bar chart comparing the average scores of the prompts.
    """
    chart_filename = f"comparison_{project_name}_chart.png"
    chart_path = os.path.join(output_dir, chart_filename)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_df["prompt_name"], avg_df["similarity"], color="lightgreen")
    plt.title(f"Prompt Comparison: '{project_name}'")
    plt.xlabel("Prompt")
    plt.ylabel("Average Similarity Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"📊 Comparison chart saved to {chart_path}")

def main():
    parser = argparse.ArgumentParser(description="Compare multiple prompts against a single benchmark.")
    parser.add_argument("--project", required=True, help="Path to the project JSON file listing the prompts to compare.")
    parser.add_argument("--benchmark", required=True, help="Path to the benchmark testset JSON file.")
    parser.add_argument("--model-alias", required=True, help="Alias of the model to use from models_config.json.")
    parser.add_argument("--output-dir", default="results", help="Directory to save the output CSV files.")
    parser.add_argument("--media-dir", default="media", help="Directory to save the comparison chart.")
    args = parser.parse_args()

    result = compare_multiple_prompts(
        project_file=args.project,
        benchmark_file=args.benchmark,
        model_alias=args.model_alias,
        output_dir=args.output_dir
    )

    if result:
        avg_df, project_name = result
        plot_comparison(avg_df, project_name, args.media_dir)

if __name__ == "__main__":
    main()