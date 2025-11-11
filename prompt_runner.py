import os, json, argparse, pandas as pd
from dotenv import load_dotenv
import openai
import matplotlib.pyplot as plt

load_dotenv()

def run_prompt(task, user_input, target_lang=None, use_mock=True):
    if use_mock:
        mocks = {
            "summarize": "AI lets machines act like humans.",
            "translate": "¡Hola mundo!",
            "explain": "Cloud computing uses remote servers."
        }
        return mocks.get(task, "No mock result.")
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    prompt = f"Task: {task}\nInput: {user_input}"
    if target_lang:
        prompt += f"\nTarget language: {target_lang}"
    r = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content.strip()

def score_response(task, output):
    """
    Simple evaluation function.
    Returns score 1–5 and boolean factual flag.
    """
    # placeholder logic, refine later
    score_map = {"summarize":5, "translate":5, "explain":4}
    score = score_map.get(task, 3)
    factual = True if score >= 4 else False
    return score, factual

def main():
    parser = argparse.ArgumentParser(description="Prompt Playground Runner")
    parser.add_argument("--api", action="store_true", help="Use real API instead of mock.")
    parser.add_argument("--out", default="reports/prompt_runner_results.csv", help="Output CSV path.")
    args = parser.parse_args()

    data = pd.read_json("datasets/sample_inputs.json")
    results = []
    for _, row in data.iterrows():
        output = run_prompt(
            task=row["task"],
            user_input=row["input"],
            target_lang=row.get("target_lang"),
            use_mock=not args.api
        )
        score, factual = score_response(row["task"], output)
        results.append({
            "id": row["id"],
            "task": row["task"],
            "output": output,
            "score": score,
            "factual": factual
        })

    # save CSV
    pd.DataFrame(results).to_csv(args.out, index=False)

    # also generate a Markdown summary
    md_path = args.out.replace(".csv", "_eval.md")
    with open(md_path, "w") as f:
        f.write("# Prompt Evaluation Results\n\n")
        f.write("| id | task | score | factual |\n")
        f.write("|----|------|-------|--------|\n")
        for r in results:
            f.write(f"| {r['id']} | {r['task']} | {r['score']} | {r['factual']} |\n")
    print(f"✅ Results saved to {args.out} and {md_path}")

    # simple bar chart of scores
    tasks = [r["task"] for r in results]
    scores = [r["score"] for r in results]

    plt.figure(figsize=(6,4))
    plt.bar(tasks, scores, color="skyblue")
    plt.ylim(0,5)
    plt.title("Prompt Evaluation Scores")
    plt.ylabel("Score (1–5)")
    plt.tight_layout()

    png_path = args.out.replace(".csv", "_eval.png")
    plt.savefig(png_path)
    plt.close()
    print(f"✅ Visualization saved to {png_path}")

if __name__ == "__main__":
    main()
