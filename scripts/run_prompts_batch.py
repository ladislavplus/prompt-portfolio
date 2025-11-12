import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
import time
import random

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "datasets" / "generated_prompts.jsonl"
OUT = ROOT / "reports" / "prompt_runs.csv"

# --- Prompt execution helpers ---

def run_prompt_mock(prompt: str, model: str):
    """Offline mock output for testing without API calls."""
    mock_responses = [
        f"[MOCK:{model}] Summary: {prompt[:120]}...",
        f"[MOCK:{model}] Translation generated successfully.",
        f"[MOCK:{model}] Explanation complete.",
        f"[MOCK:{model}] Extraction result: {{'field':'value'}}"
    ]
    return random.choice(mock_responses), "mock"

def run_prompt_api(prompt: str, model: str):
    """Call OpenAI API if USE_OPENAI_API=true and key is set."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        return resp.choices[0].message.content.strip(), "api"
    except Exception as e:
        return f"[ERROR] {e}", "error"

# --- Main runner ---

def main(models=("gpt-3.5-turbo", "gpt-4-turbo")):
    os.makedirs(OUT.parent, exist_ok=True)
    rows = []

    use_api = os.getenv("USE_OPENAI_API", "false").lower() == "true"
    run_func = run_prompt_api if use_api else run_prompt_mock

    with open(INPUT, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            for model in models:
                start = time.time()
                out, mode = run_func(rec["prompt"], model)
                elapsed = round(time.time() - start, 3)
                rows.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "task": rec["task"],
                    "model": model,
                    "prompt": rec["prompt"],
                    "output": out,
                    "mode": mode,
                    "elapsed_s": elapsed
                })
            # optional: limit runs for quick tests
            if i > 50:
                break

    df = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_csv(OUT)
        df = pd.concat([old, df], ignore_index=True)

    df.to_csv(OUT, index=False)
    print(f"✅ Wrote runs -> {OUT}")
    print(df.head(3))

if __name__ == "__main__":
    main()
