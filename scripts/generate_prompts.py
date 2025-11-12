import json
import itertools
import os
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / "datasets" / "bench_templates.json"
OUT = ROOT / "datasets" / "generated_prompts.jsonl"

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
            var_keys = list(task.get("vars", {}).keys())
            var_values = [task["vars"][k] for k in var_keys]
            for combo in itertools.product(*var_values):
                vars_map = dict(zip(var_keys, combo))
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
    print(f"✅ Generated prompts -> {OUT}")

if __name__ == "__main__":
    expand_templates()
