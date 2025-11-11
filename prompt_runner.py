import os, json, pandas as pd
from dotenv import load_dotenv
#from notebooks import  # ignore if not a module
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
    prompt = f"Task: {task}\\nInput: {user_input}"
    if target_lang: prompt += f"\\nTarget language: {target_lang}"
    r = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content.strip()

def main():
    data = pd.read_json("datasets/sample_inputs.json")
    results = []
    for _, row in data.iterrows():
        results.append({
            "id": row["id"],
            "task": row["task"],
            "input": row["input"],
            "output": run_prompt(row["task"], row["input"], row.get("target_lang"), use_mock=True)
        })
    df = pd.DataFrame(results)
    df.to_csv("reports/prompt_runner_results.csv", index=False)
    print("✅ Saved to reports/prompt_runner_results.csv")

if __name__ == "__main__":
    main()