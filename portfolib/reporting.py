import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import textwrap

def save_results_csv(df, output_dir, filename):
    """Saves a DataFrame to a CSV file."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, filename)
    df.to_csv(csv_path, index=False)
    print(f"✅ Results saved to {csv_path}")
    return csv_path

def plot_benchmark_results(df, prompt_name, test_name, model_alias, timestamp, output_dir="media"):
    """Generates and saves a bar chart of the benchmark results."""
    chart_filename = f"{test_name}_{prompt_name}_{model_alias}_{timestamp}_chart.png"
    chart_path = os.path.join(output_dir, chart_filename)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.bar(df["id"], df["similarity"], color="skyblue")
    
    title = f"Prompt Benchmark: '{prompt_name}' on '{test_name}' using model '{model_alias}'"
    plt.suptitle(textwrap.fill(title, width=55), fontsize=12)
    
    plt.xlabel("Test Case ID")
    plt.ylabel("Similarity Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    plt.savefig(chart_path)
    plt.close()
    print(f"📊 Chart saved to {chart_path}")
    return chart_path

def write_benchmark_report(df, prompt_name, test_name, model_alias, timestamp, chart_path, output_dir="output/summaries"):
    """Writes a Markdown report summarizing the benchmark results."""
    report_filename = f"{test_name}_{prompt_name}_{model_alias}_{timestamp}_report.md"
    report_path = os.path.join(output_dir, report_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    avg_score = df["similarity"].mean()

    report_content = f"""# 🧠 Prompt Benchmark Report
**Prompt:** `{prompt_name}`  
**Testset:** `{test_name}`  
**Model:** `{model_alias}`  
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

def plot_comparison_results(avg_df, project_name, model_alias, timestamp, output_dir="media"):
    """
    Generates and saves a bar chart comparing the average scores of the prompts.
    """
    chart_filename = f"comparison_{project_name}_{model_alias}_{timestamp}_chart.png"
    chart_path = os.path.join(output_dir, chart_filename)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_df["prompt_name"], avg_df["similarity"], color="lightgreen")
    
    title = f"Prompt Comparison: '{project_name}' using model '{model_alias}'"
    plt.suptitle(textwrap.fill(title, width=60), fontsize=12)
    
    plt.xlabel("Prompt")
    plt.ylabel("Average Similarity Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(chart_path)
    plt.close()
    print(f"📊 Comparison chart saved to {chart_path}")

def write_run_report(df, model_alias, model_name, output_dir="output/summaries"):
    """Writes a Markdown report summarizing the run results."""
    md_path = os.path.join(output_dir, "prompt_runner_results_eval.md")
    with open(md_path, "w") as f:
        f.write("# Prompt Evaluation Results\n\n")
        f.write(f"**Model Used:** {model_alias} (`{model_name}`)\n\n")
        f.write("| id | task | score | factual |\n")
        f.write("|----|------|-------|--------|\n")
        for _, r in df.iterrows():
            f.write(f"| {r['id']} | {r['task']} | {r['score']} | {r['factual']} |\n")
    print(f"📝 Markdown report saved to {md_path}")

def plot_run_results(df, model_alias, output_dir="media"):
    """Generates and saves a bar chart of the run results."""
    png_path = os.path.join(output_dir, "prompt_runner_results_eval.png")
    tasks = df["task"]
    scores = df["score"]

    plt.figure(figsize=(6,4))
    plt.bar(tasks, scores, color="skyblue")
    plt.ylim(0,5)
    
    title = f"Prompt Scores (Model: {model_alias})"
    plt.suptitle(textwrap.fill(title, width=40))
    
    plt.ylabel("Score (1–5)")
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    
    plt.savefig(png_path)
    plt.close()
    print(f"📊 Chart saved to {png_path}")

def plot_multi_model_comparison_results(avg_df, prompt_name, timestamp, output_dir="media"):
    """
    Generates and saves a bar chart comparing the average scores of different models for a single prompt.
    """
    chart_filename = f"model_comparison_{prompt_name}_{timestamp}_chart.png"
    chart_path = os.path.join(output_dir, chart_filename)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(avg_df["model_alias"], avg_df["similarity"], color="coral")
    
    title = f"Model Comparison for Prompt: '{prompt_name}'"
    plt.suptitle(textwrap.fill(title, width=60), fontsize=12)
    
    plt.xlabel("Model Alias")
    plt.ylabel("Average Similarity Score")
    plt.ylim(0, 1)
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.3f}', va='bottom', ha='center')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    plt.savefig(chart_path)
    plt.close()
    print(f"📊 Model comparison chart saved to {chart_path}")
    return chart_path

def write_multi_model_comparison_report(avg_df, prompt_name, timestamp, chart_path, output_dir="output/summaries"):
    """Writes a Markdown report summarizing the multi-model comparison results."""
    report_filename = f"model_comparison_{prompt_name}_{timestamp}_report.md"
    report_path = os.path.join(output_dir, report_filename)
    os.makedirs(output_dir, exist_ok=True)
    
    report_content = f"""# 🧠 Model Comparison Report
**Prompt:** `{prompt_name}`  
**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  

---

## 📊 Overview Chart
![Model Comparison Chart]({os.path.relpath(chart_path, output_dir)})

---

## 📋 Detailed Results

| Model Alias | Average Similarity |
|-------------|--------------------|
"""
    for _, row in avg_df.iterrows():
        report_content += f"| {row['model_alias']} | **{row['similarity']:.3f}** |\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"📝 Model comparison report saved to {report_path}")