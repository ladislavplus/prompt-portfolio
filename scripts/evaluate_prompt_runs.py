import pandas as pd
from pathlib import Path
from textblob import TextBlob
import random

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "reports" / "prompt_runs.csv"
OUT_CSV = ROOT / "reports" / "prompt_eval_results.csv"
OUT_MD = ROOT / "reports" / "prompt_eval_summary.md"

# --- Simple scoring heuristics ---
def score_relevance(output, prompt):
    return min(len(set(output.split()) & set(prompt.split())) / max(len(prompt.split()), 1) * 10, 5)

def score_clarity(output):
    blob = TextBlob(output)
    return round(5 - (abs(blob.sentiment.polarity) * 2), 2)

def score_completeness(output):
    length = len(output.split())
    if length < 10: return 1
    elif length < 30: return 3
    else: return 5

def evaluate():
    df = pd.read_csv(IN)
    results = []
    for _, row in df.iterrows():
        r_score = score_relevance(row["output"], row["prompt"])
        c_score = score_clarity(row["output"])
        comp_score = score_completeness(row["output"])
        total = round((r_score + c_score + comp_score) / 3, 2)
        results.append({
            "task": row["task"],
            "model": row["model"],
            "relevance": r_score,
            "clarity": c_score,
            "completeness": comp_score,
            "avg_score": total
        })

    df_scores = pd.DataFrame(results)
    summary = df_scores.groupby(["task", "model"]).mean().round(2).reset_index()

    # Save outputs
    df_scores.to_csv(OUT_CSV, index=False)
    summary.to_markdown(OUT_MD, index=False)

    print(f"✅ Saved scores -> {OUT_CSV}")
    print(f"✅ Saved summary -> {OUT_MD}")
    print(summary.head(3))

if __name__ == "__main__":
    evaluate()
