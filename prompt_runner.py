import os, json, argparse, pandas as pd
from dotenv import load_dotenv
import openai

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
        results.append({"id": row["id"], "task": row["task"], "output": output})
    pd.DataFrame(results).to_csv(args.out, index=False)
    print(f"✅ Results saved to {args.out}")

if __name__ == "__main__":
    main()
