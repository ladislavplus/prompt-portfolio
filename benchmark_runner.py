import os, json, pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from difflib import SequenceMatcher

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def run_prompt(prompt_template, input_text, mock=False):
    prompt = prompt_template.replace("{{input}}", input_text)
    if mock:
        return f"[MOCK OUTPUT for]: {input_text}"
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    return response.output_text

def score_similarity(expected, output):
    """Return simple similarity score between 0–1."""
    return round(SequenceMatcher(None, expected.lower(), output.lower()).ratio(), 3)

def benchmark(prompt_file, testset_file, mock=False):
    with open(prompt_file) as f: prompt_data = json.load(f)
    with open(testset_file) as f: test_data = json.load(f)
    results = []
    for case in test_data:
        output = run_prompt(prompt_data["template"], case["input"], mock)
        score = score_similarity(case["expected"], output)
        results.append({
            "id": case["id"],
            "input": case["input"],
            "expected": case["expected"],
            "output": output,
            "similarity": score
        })
    df = pd.DataFrame(results)
    df["avg_similarity"] = df["similarity"].mean().round(3)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/{prompt_data['id']}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)
    print(f"✅ Benchmark complete. Average similarity: {df['avg_similarity'][0]:.3f}")
    print(f"Results saved to {csv_path}")
    return df


import matplotlib.pyplot as plt

def plot_results(df, output_path):
    plt.figure(figsize=(6, 4))
    plt.bar(df["id"], df["similarity"], color="skyblue")
    plt.title("Prompt Benchmark Similarity Scores")
    plt.xlabel("Test Case ID")
    plt.ylabel("Similarity (0–1)")
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"📊 Chart saved to {output_path}")


def write_markdown_report(df, prompt_name, chart_path, report_path):
    avg_score = df["similarity"].mean().round(3)
    report_content = f"""# 🧠 Prompt Benchmark Report: {prompt_name}

**Date:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Average Similarity:** {avg_score}

---

## 📊 Overview Chart
![Benchmark Chart]({chart_path})

---

## 📋 Detailed Results

| ID | Input | Expected | Output | Similarity |
|----|--------|-----------|---------|-------------|
"""

    for _, row in df.iterrows():
        report_content += f"| {row['id']} | {row['input']} | {row['expected']} | {row['output']} | {row['similarity']} |\n"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"📝 Markdown report written to {report_path}")



if __name__ == "__main__":
    df = benchmark(
        "prompts/library/translation_prompt.json",
        "prompts/benchmarks/translation_testset.json",
        mock=True
    )
    chart_path = "media/translation_en_to_fr_chart.png"
    plot_results(df, chart_path)
    write_markdown_report(
        df,
        "translation_en_to_fr",
        chart_path,
        "reports/translation_en_to_fr_report.md"
    )