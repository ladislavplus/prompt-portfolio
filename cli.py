import argparse
import json
import pandas as pd
from datetime import datetime
import os

from prompt_toolkit.utils import load_json
from prompt_toolkit.models import get_model_name, run_prompt
from prompt_toolkit.evaluation import score_similarity, evaluate_run
from prompt_toolkit.reporting import (
    save_results_csv,
    plot_benchmark_results,
    write_benchmark_report,
    plot_comparison_results,
    write_run_report,
    plot_run_results,
)

def benchmark_command(args):
    """Runs the benchmark command."""
    print(f"🚀 Running benchmark for prompt '{args.prompt}' with model '{args.model_alias}'...")
    
    models_config = load_json("config/models_config.json")
    model_name = get_model_name(args.model_alias, models_config)
    
    prompt_data = load_json(args.prompt)
    test_data = load_json(args.benchmark)

    results = []
    for case in test_data["test_cases"]:
        output = run_prompt(prompt_data["prompt_text"], model_name, case["input"])
        score = score_similarity(case["expected_output"], output)
        results.append({
            "id": case["id"],
            "input": json.dumps(case["input"]),
            "expected": case["expected_output"],
            "output": output,
            "similarity": score,
        })

    if not results:
        print("❌ No results were generated. Exiting.")
        return

    df = pd.DataFrame(results)
    avg_similarity = df["similarity"].mean()
    df["avg_similarity"] = avg_similarity

    prompt_name_slug = prompt_data['name'].replace('_', '-')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{test_data['name']}_{prompt_name_slug}_{timestamp}.csv"
    save_results_csv(df, args.output_dir, csv_filename)

    print(f"✅ Benchmark complete. Average similarity: {avg_similarity:.3f}")

    chart_path = plot_benchmark_results(df, prompt_data['name'], test_data['name'], args.media_dir)
    write_benchmark_report(df, prompt_data['name'], test_data['name'], chart_path)

def compare_command(args):
    """Runs the compare command."""
    print(f"🚀 Comparing prompts from project '{args.project}'...")

    project_data = load_json(args.project)
    prompt_files = project_data["prompts"]
    all_results = []

    for prompt_file in prompt_files:
        args.prompt = prompt_file
        # This is a simplified version. In a real scenario, you might want to avoid this kind of argument mutation.
        # For this refactoring, it's a pragmatic choice to reuse the benchmark logic.
        # A more advanced implementation could refactor benchmark_command to be more modular.
        
        # Temporarily redirect output to avoid clutter
        import sys
        from io import StringIO
        old_stdout = sys.stdout
        sys.stdout = StringIO()

        benchmark_command(args)

        sys.stdout.seek(0)
        output = sys.stdout.read()
        sys.stdout = old_stdout
        
        # Find the latest result file for this prompt
        list_of_files = os.listdir(args.output_dir)
        full_path = [os.path.join(args.output_dir, i) for i in list_of_files if i.startswith(f"comparison_{project_data['name']}")]
        if not full_path:
            # Fallback for individual benchmark runs if compare fails
            test_data = load_json(args.benchmark)
            prompt_data = load_json(prompt_file)
            prompt_name_slug = prompt_data['name'].replace('_', '-')
            full_path = [os.path.join(args.output_dir, i) for i in list_of_files if i.startswith(f"{test_data['name']}_{prompt_name_slug}")]

        if not full_path:
            print(f"Could not find results file for {prompt_file}")
            continue
            
        latest_file = max(full_path, key=os.path.getctime)
        df = pd.read_csv(latest_file)
        df["prompt_name"] = load_json(prompt_file)["name"]
        all_results.append(df)


    if not all_results:
        print("❌ No results were generated from any prompt. Exiting.")
        return

    combined = pd.concat(all_results)
    
    project_name_slug = project_data['name'].replace('_', '-')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_csv_filename = f"comparison_{project_name_slug}_{timestamp}.csv"
    save_results_csv(combined, args.output_dir, combined_csv_filename)

    avg_df = combined.groupby("prompt_name")["similarity"].mean().reset_index()
    
    print("\n📊 Average similarity per prompt:")
    print(avg_df.to_string(index=False))
    
    plot_comparison_results(avg_df, project_data['name'], args.media_dir)

def run_command(args):
    """Runs the run command."""
    models_config = load_json("config/models_config.json")
    model_alias = args.model_alias or models_config["default_model"]
    model_name = get_model_name(model_alias, models_config)

    print(f"🚀 Running prompts with model: {model_alias} ({model_name})")

    data = pd.read_json(args.data)
    results = []
    for _, row in data.iterrows():
        prompt = f"Task: {row['task']}\nInput: {row['input']}"
        if "target_lang" in row and row["target_lang"]:
            prompt += f"\nTarget language: {row['target_lang']}"
        
        output = run_prompt(prompt, model_name)
        eval_scores = evaluate_run(output, prompt)
        
        results.append({
            "id": row["id"],
            "task": row["task"],
            "output": output,
            "score": eval_scores["avg_score"],
            "factual": eval_scores["clarity"] > 3.5, # Example logic
            "model_alias": model_alias,
        })

    df = pd.DataFrame(results)
    save_results_csv(df, "output/raw", "prompt_runner_results.csv")
    write_run_report(df, model_alias, model_name, "output/summaries")
    plot_run_results(df, model_alias, "media")

def main():
    parser = argparse.ArgumentParser(description="Prompt Portfolio CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Benchmark command
    parser_benchmark = subparsers.add_parser("benchmark", help="Run a benchmark for a single prompt.")
    parser_benchmark.add_argument("--benchmark", required=True, help="Path to the benchmark testset JSON file.")
    parser_benchmark.add_argument("--prompt", required=True, help="Path to the prompt JSON file.")
    parser_benchmark.add_argument("--model-alias", required=True, help="Alias of the model to use from models_config.json.")
    parser_benchmark.add_argument("--output-dir", default="output/raw", help="Directory to save the output CSV file.")
    parser_benchmark.add_argument("--media-dir", default="media", help="Directory to save the chart image.")
    parser_benchmark.set_defaults(func=benchmark_command)

    # Compare command
    parser_compare = subparsers.add_parser("compare", help="Compare multiple prompts against a single benchmark.")
    parser_compare.add_argument("--project", required=True, help="Path to the project JSON file listing the prompts to compare.")
    parser_compare.add_argument("--benchmark", required=True, help="Path to the benchmark testset JSON file.")
    parser_compare.add_argument("--model-alias", required=True, help="Alias of the model to use from models_config.json.")
    parser_compare.add_argument("--output-dir", default="output/raw", help="Directory to save the output CSV files.")
    parser_compare.add_argument("--media-dir", default="media", help="Directory to save the comparison chart.")
    parser_compare.set_defaults(func=compare_command)

    # Run command
    parser_run = subparsers.add_parser("run", help="Run a series of prompts from a dataset.")
    parser_run.add_argument("--data", default="datasets/sample_inputs.json", help="Path to dataset JSON")
    parser_run.add_argument("--model-alias", help="Alias of the model to use from models_config.json")
    parser_run.set_defaults(func=run_command)

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
