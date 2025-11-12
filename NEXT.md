Goal

Produce reproducible, defensible evidence of prompt-engineering skill by:

Creating a diverse dataset of realistic tasks and prompt templates (3–5 tasks, multiple variants).

Generating many prompt variants programmatically.

Running them across models (mock + real API) and saving outputs.

Implementing an evaluation pipeline with both automatic heuristics and a lightweight human review (Streamlit reviewer app + CSV).

Producing final analytics and visuals you can publish in the repo.

Step A — Create a richer dataset of tasks & templates

Why: the current few mock prompts don’t show breadth. We’ll create templates for 4 task types: Summarize, Translate, Explain (tech), and Instruction-following (e.g., extract). Each template will include variables (length, style, audience). Programmatic generation will produce variants you can test.

Files to add

datasets/bench_templates.json (new)

scripts/generate_prompts.py (new)

output: datasets/generated_prompts.jsonl

Create datasets/bench_templates.json

{
  "tasks": [
    {
      "id": "summarize",
      "title": "Summarize",
      "description": "Summarize a given paragraph at varying lengths and styles.",
      "templates": [
        "Summarize the following text in {length} sentences for a {audience} audience:\n\n\"{source}\"",
        "Write a {length}-sentence summary of the text below, focusing on {focus}:\n\n{source}"
      ],
      "vars": {
        "length": ["1", "2", "3"],
        "audience": ["general", "technical", "executive"],
        "focus": ["key findings", "main argument", "practical implications"]
      }
    },
    {
      "id": "translate",
      "title": "Translate",
      "description": "Translate text into another language with register instructions.",
      "templates": [
        "Translate the following passage into {lang} using a {register} register:\n\n{source}"
      ],
      "vars": {
        "lang": ["French", "Spanish", "German"],
        "register": ["formal", "casual", "technical"]
      }
    },
    {
      "id": "explain",
      "title": "Explain",
      "description": "Explain a technical concept at different depths.",
      "templates": [
        "Explain {concept} in a way a {audience} would understand. Keep it {depth}."
      ],
      "vars": {
        "concept": ["blockchain consensus", "gradient descent", "HTTP cookies"],
        "audience": ["high-school student", "new developer", "non-technical manager"],
        "depth": ["very brief (<=2 sentences)", "concise (3-4 sentences)", "detailed (>=6 sentences)"]
      }
    },
    {
      "id": "extract",
      "title": "Extract",
      "description": "Structured extraction from a text (entities or Q/A).",
      "templates": [
        "Read the passage and extract the following fields as JSON: {fields}.\n\nPassage:\n{source}"
      ],
      "vars": {
        "fields": ["name, date, location", "product, price, warranty", "issue_summary, steps_taken"]
      }
    }
  ]
}


Create scripts/generate_prompts.py

# scripts/generate_prompts.py
import json
import itertools
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "datasets" / "bench_templates.json"
OUT = ROOT / "datasets" / "generated_prompts.jsonl"

# small set of realistic source texts (expand later)
SOURCES = {
    "summarize": [
        "The company reported a 15% year-over-year increase in revenue driven by growth in the Asia region. Margins expanded due to cost optimization initiatives.",
        "Climate models suggest an increasing frequency of extreme weather events linked to ocean temperature anomalies and greenhouse gas concentration."
    ],
    "translate": [
        "Welcome to the yearly conference. We are excited to share our plans for the upcoming year.",
        "Please find the invoice attached. Payment due within 30 days."
    ],
    "explain": [
        "Gradient descent updates model parameters incrementally to reduce loss by following the negative gradient.",
        "HTTP cookies are small pieces of data stored in the browser that help websites remember stateful information."
    ],
    "extract": [
        "John Doe visited Berlin on 2024-06-12 to attend a conference on renewable energy. He reported issues with flight delays.",
        "Product: SuperVac 3000. Price: $199.99. Warranty: 2 years. Customer reported motor noise after 3 months."
    ]
}

def expand_templates():
    with open(TEMPLATES, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    os.makedirs(OUT.parent, exist_ok=True)
    out_f = open(OUT, "w", encoding="utf-8")

    for task in cfg["tasks"]:
        tid = task["id"]
        for tmpl in task["templates"]:
            # build cartesian product of vars
            var_keys = list(task.get("vars", {}).keys())
            var_values = [task.get("vars", {})[k] for k in var_keys]
            for combo in itertools.product(*var_values):
                vars_map = dict(zip(var_keys, combo))
                # pick a random source for the task
                source = random.choice(SOURCES.get(tid, [""]))
                vars_map["source"] = source
                prompt = tmpl.format(**vars_map)
                record = {
                    "task": tid,
                    "template": tmpl,
                    "vars": vars_map,
                    "prompt": prompt
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
    out_f.close()
    print(f"Generated prompts -> {OUT}")

if __name__ == "__main__":
    expand_templates()


Run command

python scripts/generate_prompts.py


Expected

datasets/generated_prompts.jsonl produced (many prompt variants).

Quick inspection: head -n 3 datasets/generated_prompts.jsonl shows JSON lines.

Troubleshooting

If bench_templates.json path error → ensure directory structure and filenames match.

Add more SOURCES entries for realistic variety.

Step B — Run prompt suite across models (automated)

Why: produce outputs you can analyze. Use your existing benchmark runner or add a script that reads generated_prompts.jsonl and writes reports/prompt_runs.csv.

File to add

scripts/run_prompts_batch.py

Code

# scripts/run_prompts_batch.py
import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import time
import random

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "datasets" / "generated_prompts.jsonl"
OUT = ROOT / "reports" / "prompt_runs.csv"

# Minimal run_prompt wrapper (reuses your app logic / can import from project)
def run_prompt_mock(prompt, model):
    # deterministic-ish mock using prompt hash
    return f"[MOCK:{model}] {prompt[:200]}...", "mock"

def run_prompt_api(prompt, model):
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":prompt}],
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip(), "api"
    except Exception as e:
        return f"[ERROR] {e}", "error"

def main(models=("gpt-3.5-turbo", "gpt-4-turbo")):
    os.makedirs(OUT.parent, exist_ok=True)
    rows = []
    use_api = os.getenv("USE_OPENAI_API","false").lower() == "true"

    with open(INPUT, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            prompt = rec["prompt"]
            for model in models:
                start = time.time()
                if use_api:
                    out, mode = run_prompt_api(prompt, model)
                else:
                    out, mode = run_prompt_mock(prompt, model)
                elapsed = round(time.time() - start, 3)
                rows.append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "task": rec["task"],
                    "model": model,
                    "prompt": prompt,
                    "output": out,
                    "mode": mode,
                    "elapsed_s": elapsed
                })
    df = pd.DataFrame(rows)
    if OUT.exists():
        old = pd.read_csv(OUT)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote runs -> {OUT}")

if __name__ == "__main__":
    main()


Run

# Using mock mode (safe, fast)
python scripts/run_prompts_batch.py

# To run with real API (if you want and have quota)
export USE_OPENAI_API=true
python scripts/run_prompts_batch.py


Expected

reports/prompt_runs.csv with many rows (task, model, prompt, output, elapsed).

Use mock mode if you want many iterations without cost.

Troubleshooting

If API errors, check OPENAI_API_KEY in .env and that your openai package version matches usage.

Step C — Define an evaluation rubric + automatic heuristics

Why: employers want to see evaluation methodology. We'll compute several automatic metrics:

Instruction-following (does output mention task keywords?) — heuristic

Conciseness (token/word length compared to requested length)

Fluency (language model perplexity is not available — instead use simple heuristics: punctuation, non-ASCII)

Semantic similarity to the source (if available: use embeddings via OpenAI or a local sentence-transformer; we’ll provide both options)

File

scripts/evaluate_runs.py

Code (basic heuristics + optional embeddings if API available)

# scripts/evaluate_runs.py
import os
import pandas as pd
import numpy as np
from pathlib import Path
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "reports" / "prompt_runs.csv"
OUT = ROOT / "reports" / "prompt_evals.csv"

def instr_following_score(prompt, output):
    # naive: check overlap with words like "summarize","translate","explain","extract"
    keywords = ["summarize", "translate", "explain", "extract", "paraphrase"]
    score = sum(1 for k in keywords if k in prompt.lower() and k in output.lower())
    return min(score, 1)  # 0 or 1

def conciseness_score(prompt, output):
    # prefer shorter for summarize; penalize overly long translations
    pwords = len(prompt.split())
    owords = len(output.split())
    ratio = owords / max(1, pwords)
    if ratio < 0.5:
        return 1.0
    if ratio < 1.5:
        return 0.8
    if ratio < 3.0:
        return 0.5
    return 0.2

def semantic_sim_score(prompt, output):
    # TF-IDF cosine between prompt and output as rough proxy
    vec = TfidfVectorizer().fit_transform([prompt, output])
    cos = cosine_similarity(vec[0:1], vec[1:2])[0][0]
    return float(cos)

def main():
    df = pd.read_csv(IN)
    evals = []
    for _, row in df.iterrows():
        p = row["prompt"]
        o = str(row["output"])
        fscore = instr_following_score(p, o)
        cscore = conciseness_score(p, o)
        sscore = semantic_sim_score(p, o)
        # aggregate (simple average)
        overall = np.mean([fscore, cscore, sscore])
        evals.append({
            **row.to_dict(),
            "instr_following": fscore,
            "conciseness": cscore,
            "semantic_similarity": sscore,
            "overall_score": overall
        })
    outdf = pd.DataFrame(evals)
    outdf.to_csv(OUT, index=False)
    print(f"Wrote evals -> {OUT}")

if __name__ == "__main__":
    main()


Run

python scripts/evaluate_runs.py


Expected

reports/prompt_evals.csv containing per-run scores and overall_score (0–1).

Troubleshooting

sklearn missing → pip install scikit-learn

TF-IDF similarity is only a rough proxy; where possible use embeddings for better semantic similarity.

Step D — Human-in-the-loop reviewer app

Why: automatic metrics are useful but human review is gold for evaluation. We’ll add a compact Streamlit reviewer that presents runs and collects human ratings for: correctness (1–5), usefulness (1–5), safety flag (Y/N), comments. Output saved to reports/human_evals.csv.

File

reviewer/reviewer_app.py

Code

# reviewer/reviewer_app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports" / "prompt_runs.csv"
OUT = ROOT / "reports" / "human_evals.csv"

st.set_page_config(page_title="Prompt Reviewer", layout="wide")
st.title("📝 Prompt Reviewer")

if not DATA.exists():
    st.warning("No prompt runs found. Run scripts/run_prompts_batch.py first.")
    st.stop()

df = pd.read_csv(DATA)
idx = st.number_input("Start index", min_value=0, max_value=max(0, len(df)-1), value=0, step=1)
row = df.iloc[int(idx)]

st.subheader(f"Task: {row['task']} | Model: {row['model']}")
st.markdown("**Prompt:**")
st.code(row["prompt"])
st.markdown("**Output:**")
st.text_area("Output", value=row["output"], height=200)

col1, col2, col3 = st.columns(3)
with col1:
    correctness = st.slider("Correctness (1-5)", 1, 5, 3)
with col2:
    usefulness = st.slider("Usefulness (1-5)", 1, 5, 3)
with col3:
    safety = st.selectbox("Safety OK?", ["yes", "no"])

comments = st.text_area("Reviewer notes / short justification", height=80)

if st.button("Submit Review"):
    out = {
        "timestamp": datetime.utcnow().isoformat(),
        "row_index": int(idx),
        "task": row["task"],
        "model": row["model"],
        "prompt": row["prompt"],
        "output": row["output"],
        "correctness": correctness,
        "usefulness": usefulness,
        "safety": safety,
        "comments": comments
    }
    df_out = pd.DataFrame([out])
    if OUT.exists():
        old = pd.read_csv(OUT)
        df_out = pd.concat([old, df_out], ignore_index=True)
    df_out.to_csv(OUT, index=False)
    st.success("Review saved!")


Run reviewer

cd reviewer
streamlit run reviewer_app.py


How to use

Recruit 3–5 reviewers (peers, colleagues, or friends). They open this app and review randomized indexes.

Each reviewer fills ~20 samples — enough to compute inter-rater reliability and produce meaningful analysis.

Troubleshooting

If you can’t recruit people, do an internal double-blind review (you + 1 peer) — still valuable.

Step E — Compute inter-rater stats & final analytics notebook

Why: employers like metrics: average scores, distributions, inter-rater reliability (Cohen’s kappa), and clear visualizations.

Files

notebooks/02_evaluation_analysis.ipynb (new)

Or a Python script scripts/analysis.py to generate charts

Sketch of analysis tasks

Load reports/prompt_evals.csv and reports/human_evals.csv

Compute per-model, per-task averages (auto vs human)

Plot distributions (histograms/boxplots)

Compute Cohen’s kappa / Krippendorff’s alpha (use sklearn.metrics.cohen_kappa_score for pair comparisons)

Save charts to media/ (e.g., media/eval_scores_by_model.png)

Small scripts/analysis.py example

# scripts/analysis.py
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
import os

OUT = "media"
os.makedirs(OUT, exist_ok=True)

human = pd.read_csv("reports/human_evals.csv")
if human.empty:
    print("No human evaluations found.")
    exit(1)

# example: average correctness by model
avg_by_model = human.groupby("model")["correctness"].mean().sort_values()
fig, ax = plt.subplots(figsize=(6,4))
avg_by_model.plot.barh(ax=ax)
ax.set_xlabel("Avg Correctness (1-5)")
ax.set_title("Human Avg Correctness by Model")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "human_correctness_by_model.png"))
print("Saved chart -> media/human_correctness_by_model.png")

# sample inter-rater: compare reviewer1 vs reviewer2 if both reviewed same row_index
# naive example: pivot by reviewer id if you add one; otherwise compute pairwise kappas per row subset.


Run

python scripts/analysis.py


Expected

Charts saved in media/

Numeric summaries printed to console and saved as CSV in reports/analysis_summary.csv (you can extend the script to save CSVs)

Step F — Write a robust “Methodology” section for the repo

Why: employers need to see how you tested and validated prompts. Create METHODS.md describing:

Data generation process

Models tested

Evaluation metrics (automatic + human; definitions)

Reviewer instructions and recruitment

Reproducibility steps (commands to re-run, environment)

Filename

METHODS.md at repo root

Short template

# Methods — Prompt Engineering Experiments

## Data generation
- Templates: datasets/bench_templates.json
- Generated prompts: datasets/generated_prompts.jsonl (script: scripts/generate_prompts.py)

## Execution
- Batch runner: scripts/run_prompts_batch.py
- Use mock mode by default. To run live, set USE_OPENAI_API=true and ensure OPENAI_API_KEY in .env.

## Automatic evaluation
- scripts/evaluate_runs.py produces reports/prompt_evals.csv
- Metrics: instruction-following (binary), conciseness, semantic similarity (TF-IDF baseline)

## Human evaluation
- Reviewer app: reviewer/reviewer_app.py
- Output: reports/human_evals.csv
- Required fields: correctness (1–5), usefulness (1–5), safety flag, comments
- Recruit 3+ reviewers; randomize indices per reviewer

## Analysis
- scripts/analysis.py and notebooks/02_evaluation_analysis.ipynb

Step G — Produce portfolio artifacts & narrative

Add reports/final_summary.md that includes:

Key metrics (avg overall_score by model)

Human evaluation summary (avg correctness/usefulness)

One or two sample prompt+response pairs (good and bad) with short commentary

Links to charts in /media

Update README.md to point to METHODS.md and reports/final_summary.md.

Quick timeline & minimum viable evidence (what to aim for first)

If you want a practical minimum set of deliverables to validate experience quickly, do these in order (each is one evening’s work):

A — Generate 200 prompt variants using generate_prompts.py.

B — Run batch in mock mode to produce reports/prompt_runs.csv.

C — Run automatic evaluation to produce reports/prompt_evals.csv.

D — Launch reviewer app and get 3 reviewers to each rate 20 items (you’ll get 60 human eval rows).

E — Run analysis.py, create charts and reports/final_summary.md.

F — Add Methods & update README.

This yields: generated dataset, automated runs, human labels, analysis, and a documented methodology — exactly the evidence recruiters want.

Security, privacy & practical notes

If using real user data, remove PII before posting to GitHub. The extract task templates may surface names/dates — redact or use synthetic data.

If using OpenAI API, be mindful of costs. Use mock mode for bulk generation and sample API runs for qualitative checks.

For human reviewers, use personal contacts or a small paid pool (HITs on Mechanical Turk / Upwork) if you want a faster scale; otherwise, colleagues/friends are fine.