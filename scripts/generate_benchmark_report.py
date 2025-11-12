import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "reports" / "prompt_eval_results.csv"
SUMMARY = ROOT / "reports" / "prompt_eval_summary.md"
IMG = ROOT / "media" / "prompt_eval_chart.png"
OUT = ROOT / "reports" / "prompt_benchmark_report.md"

def generate_report():
    now = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    eval_df = pd.read_csv(EVAL)
    avg_scores = eval_df.groupby(["model"]).mean(numeric_only=True)["avg_score"].round(2)
    best_model = avg_scores.idxmax()
    best_score = avg_scores.max()

    report = f"""# 🧮 Prompt Benchmark Report  
**Date:** {now}  

This report summarizes prompt evaluation results from automated batch tests.

---

## 📊 Performance Summary

| Model | Avg. Score |
|--------|-------------|
"""

    for model, score in avg_scores.items():
        report += f"| {model} | {score} |\n"

    report += f"""
---

## 🏅 Top Performing Model
**{best_model}** — average score **{best_score}**

---

## 📈 Evaluation Chart
![Prompt Evaluation Chart]({IMG.relative_to(ROOT)})

---

## 📄 Detailed Task Summary
(see `{SUMMARY.relative_to(ROOT)}` for per-task breakdown)
"""

    OUT.write_text(report, encoding="utf-8")
    print(f"✅ Benchmark report saved -> {OUT}")

if __name__ == "__main__":
    generate_report()
