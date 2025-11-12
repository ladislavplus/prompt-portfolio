import os
import json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import litellm
from difflib import SequenceMatcher
import argparse
import matplotlib.pyplot as plt

load_dotenv()

def load_models_config(path="models_config.json"):
    """Loads the models configuration file."""
    with open(path, 'r') as f:
        return json.load(f)

def run_prompt(prompt_template, variables, model_name):
    """
    Runs a prompt using the specified model via litellm, with variable substitution.
    """
    prompt = prompt_template.format(**variables)
    try:
        r = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling litellm for model {model_name}: {e}")
        return f"Error: Could not get response from model {model_name}."

def score_similarity(expected, output):
    """Return simple similarity score between 0–1."""
    return round(SequenceMatcher(None, str(expected).lower(), str(output).lower()).ratio(), 3)

def benchmark(prompt_file, testset_file, model_alias, output_dir="results"):
    """
    Runs a benchmark for a given prompt against a testset using a specified model.
    """
    # Load configurations
    models_config = load_models_config()
    if model_alias not in models_config["models"]:
        print(f"❌ Error: Model alias '{model_alias}' not found in models_config.json.")
        return None
    model_info = models_config["models"][model_alias]
    model_name = model_info["litellm_string"]

    with open(prompt_file) as f:
        prompt_data = json.load(f)
    with open(testset_file) as f:
        test_data = json.load(f)

    results = []
    print(f"🚀 Running benchmark for prompt '{prompt_data['name']}' with model '{model_alias}'...")

    for case in test_data["test_cases"]:
        output = run_prompt(prompt_data["prompt_text"], case["input"], model_name)
        score = score_similarity(case["expected_output"], output)
        results.append({
            "id": case["id"],
            "input": json.dumps(case["input"]),
            "expected": case["expected_output"],
            "output": output,
            "similarity": score
        })

    if not results:
        print("❌ No results were generated. Exiting.")
        return None

    df = pd.DataFrame(results)
    avg_similarity = df["similarity"].mean()
    df["avg_similarity"] = avg_similarity

    # Save results
    prompt_name_slug = prompt_data['name'].replace('_', '-')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"{test_data['name']}_{prompt_name_slug}_{timestamp}.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(csv_path, index=False)

    print(f"✅ Benchmark complete. Average similarity: {avg_similarity:.3f}")
    print(f"   Results saved to {csv_path}")
    return df, prompt_data['name'], test_data['name']

def plot_results(df, prompt_name, test_name, output_dir="media"):
    """Generates and saves a bar chart of the benchmark results."""
    chart_filename = f"{test_name}_{prompt_name}_chart.png"
    chart_path = os.path.join(output_dir, chart_filename)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(df["id"], df["similarity"], color="skyblue")
    plt.title(f"Prompt Benchmark: '{prompt_name}' on '{test_name}'")
    plt.xlabel("Test Case ID")
    plt.ylabel("Similarity Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    print(f"📊 Chart saved to {chart_path}")
    return chart_path

def write_markdown_report(df, prompt_name, test_name, chart_path, output_dir="reports"):
    """Writes a Markdown report summarizing the benchmark results."""
    report_filename = f"{test_name}_{prompt_name}_report.md"
    report_path = os.path.join(output_dir, report_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    avg_score = df["similarity"].mean()

    report_content = f"""# 🧠 Prompt Benchmark Report
**Prompt:** `{prompt_name}`  
**Testset:** `{test_name}`  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Average Similarity:** `{avg_score:.3f}`

---

## 📊 Overview Chart
![Benchmark Chart]({os.path.relpath(chart_path, output_dir)})

---

## 📋 Detailed Results

| ID | Input Variables | Expected Output | Model Output | Similarity |
|----|-----------------|-----------------|--------------|------------|
"""
    for _, row in df.iterrows():
        report_content += f"| {row['id']} | `{row['input']}` | {row['expected']} | {row['output']} | **{row['similarity']}** |\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"📝 Markdown report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Run a benchmark for a specific prompt against a testset.")
    parser.add_argument("--benchmark", required=True, help="Path to the benchmark testset JSON file.")
    parser.add_argument("--prompt", required=True, help="Path to the prompt JSON file.")
    parser.add_argument("--model-alias", required=True, help="Alias of the model to use from models_config.json.")
    parser.add_argument("--output-dir", default="results", help="Directory to save the output CSV file.")
    parser.add_argument("--reports-dir", default="reports", help="Directory to save the markdown report.")
    parser.add_argument("--media-dir", default="media", help="Directory to save the chart image.")
    args = parser.parse_args()

    result = benchmark(
        prompt_file=args.prompt,
        testset_file=args.benchmark,
        model_alias=args.model_alias,
        output_dir=args.output_dir
    )

    if result:
        df, prompt_name, test_name = result
        chart_path = plot_results(df, prompt_name, test_name, args.media_dir)
        write_markdown_report(df, prompt_name, test_name, chart_path, args.reports_dir)

if __name__ == "__main__":
    main()