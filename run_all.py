import os
import subprocess

os.makedirs("reports", exist_ok=True)

datasets = [
    "datasets/sample_inputs.json",
    # "datasets/other_prompts.json",  # add more datasets here
]

for ds in datasets:
    out_csv = f"reports/{os.path.splitext(os.path.basename(ds))[0]}_results.csv"
    print(f"Running {ds} → {out_csv}")
    subprocess.run(["python", "prompt_runner.py", "--data", ds, "--out", out_csv], check=True)

print("✅ All runs completed.")
