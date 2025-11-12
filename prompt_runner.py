import logging
import os, json, argparse, pandas as pd
from dotenv import load_dotenv
import litellm
import matplotlib.pyplot as plt

load_dotenv()

# basic logging config
logging.basicConfig(
    filename="reports/prompt_runner.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

def load_models_config(path="models_config.json"):
    """Loads the models configuration file."""
    with open(path, 'r') as f:
        return json.load(f)

def run_prompt(task, user_input, model_name, target_lang=None):
    """
    Runs a prompt using the specified model via litellm.
    """
    prompt = f"Task: {task}\nInput: {user_input}"
    if target_lang:
        prompt += f"\nTarget language: {target_lang}"
    
    try:
        r = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}]
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error calling litellm for model {model_name}: {e}", exc_info=True)
        return f"Error: Could not get response from model {model_name}."

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
    parser.add_argument("--out", default="reports/prompt_runner_results.csv", help="Output CSV path.")
    parser.add_argument("--data", default="datasets/sample_inputs.json", help="Path to dataset JSON")
    parser.add_argument("--model-alias", help="Alias of the model to use from models_config.json")
    args = parser.parse_args()

    # Load models config
    models_config = load_models_config()
    
    # Determine model to use
    if args.model_alias:
        model_alias = args.model_alias
        if model_alias not in models_config["models"]:
            print(f"❌ Error: Model alias '{model_alias}' not found in models_config.json.")
            return
    else:
        model_alias = models_config["default_model"]

    model_info = models_config["models"][model_alias]
    model_name = model_info["litellm_string"]
    
    print(f"🚀 Running prompts with model: {model_alias} ({model_name})")

    # load dataset from argument
    data = pd.read_json(args.data)

    try:
        results = []
        for _, row in data.iterrows():
            output = run_prompt(
                task=row["task"],
                user_input=row["input"],
                model_name=model_name,
                target_lang=row.get("target_lang")
            )
            score, factual = score_response(row["task"], output)
            results.append({
                "id": row["id"],
                "task": row["task"],
                "output": output,
                "score": score,
                "factual": factual,
                "model_alias": model_alias
            })

        # save CSV
        pd.DataFrame(results).to_csv(args.out, index=False)

        # also generate a Markdown summary
        md_path = args.out.replace(".csv", "_eval.md")
        with open(md_path, "w") as f:
            f.write("# Prompt Evaluation Results\n\n")
            f.write(f"**Model Used:** {model_alias} (`{model_name}`)\n\n")
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
        plt.title(f"Prompt Scores (Model: {model_alias})")
        plt.ylabel("Score (1–5)")
        plt.tight_layout()

        png_path = args.out.replace(".csv", "_eval.png")
        plt.savefig(png_path)
        plt.close()
        print(f"✅ Visualization saved to {png_path}")
        logging.info(f"Processed {len(results)} prompts successfully with model {model_alias}.")
    except Exception as e:
        logging.error(f"Error during run: {e}", exc_info=True)
        print(f"❌ Error during run: {e}")

if __name__ == "__main__":
    main()
