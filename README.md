# prompt-portfolio

TODO

This repository, "prompt-portfolio," is a Python-based toolkit for managing, testing, and evaluating language model prompts. It enables a systematic, data-driven approach to prompt engineering.

  Core Purpose:

  The main goal is to help developers and prompt engineers test and compare different versions of prompts to see which ones perform best. It
  allows you to:

   * Run Prompts: Execute prompts with various inputs, using either mock data for quick tests or a real language model API (like OpenAI's GPT
     models).
   * Benchmark Prompts: Evaluate a prompt's performance by comparing its output to a set of "expected" or ideal answers. It calculates a
     similarity score for each test case.
   * Compare Prompts: Run several different prompts against the same set of tests to see which one is most effective on average.
   * Generate Reports: Automatically create reports in CSV and Markdown formats, along with charts, to visualize and summarize the results.

  How It's Used:

   1. Define Prompts: You create your prompts in .json files within the prompts/library directory.
   2. Create Test Data: You define your test cases (inputs and expected outputs) in .json files, typically in the datasets or
      prompts/benchmarks directories.
   3. Run Evaluations: You use the Python scripts to carry out the tests:
       * prompt_runner.py: For running a set of tasks from a dataset.
       * benchmark_runner.py: For scoring a single prompt against a benchmark.
       * compare_prompts.py: For comparing the performance of multiple prompts.
       * run_all.py: A convenience script to run multiple test suites at once.
   4. Analyze Results: The output is saved in the reports and results folders, allowing you to analyze which prompts are most effective and
      why.


---

## 💬 Project 4 — Prompt-Lab Web App (`v0.4.0`)

An interactive Streamlit dashboard for testing, evaluating, and visualizing prompts in real time.

### ✨ Features
- Model & task selection (Summarize, Translate, Explain, Paraphrase)
- Live prompt execution (OpenAI API + offline mock)
- Automatic CSV logging to `/reports/prompt_lab_runs.csv`
- Cached history with refresh and sorting
- Performance chart by model & task

### 🚀 Run Locally
```bash
streamlit run app.py
