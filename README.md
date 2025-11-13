# Prompt Portfolio

This repository is a Python-based toolkit for managing, testing, and evaluating language model prompts. It enables a systematic, data-driven approach to prompt engineering.

## Core Purpose

The main goal is to help developers and prompt engineers test and compare different versions of prompts to see which ones perform best. It allows you to:

- **Run Prompts:** Execute prompts with various inputs, using either mock data for quick tests or a real language model API.
- **Benchmark Prompts:** Evaluate a prompt's performance by comparing its output to a set of "expected" or ideal answers. It calculates a similarity score for each test case.
- **Compare Prompts:** Run several different prompts against the same set of tests to see which one is most effective on average.
- **Generate Reports:** Automatically create reports in CSV and Markdown formats, along with charts, to visualize and summarize the results.

## Getting Started

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Set up your environment:**
    Create a `.env` file in the root of the project and add your `OPENAI_API_KEY`:
    ```
    OPENAI_API_KEY="your-api-key"
    ```

## How It's Used

This project now uses a single command-line interface (CLI) tool, `cli.py`, to run all evaluations.

### 1. Define Prompts
Create your prompts in `.json` files within the `prompts/library` directory.

### 2. Create Test Data
Define your test cases (inputs and expected outputs) in `.json` files, typically in the `datasets` or `prompts/benchmarks` directories.

### 3. Run Evaluations
Use the `cli.py` script with one of the following subcommands:

-   **`run`**: For running a set of tasks from a dataset.
    ```bash
    python cli.py run --data datasets/sample_inputs.json --model-alias gpt-4-turbo
    ```

-   **`benchmark`**: For scoring a single prompt against a benchmark.
    ```bash
    python cli.py benchmark --prompt prompts/library/translation_prompt.json --benchmark prompts/benchmarks/translation_testset.json --model-alias gpt-4-turbo
    ```

-   **`compare`**: For comparing the performance of multiple prompts.
    ```bash
    python cli.py compare --project projects/prompt-comparison/prompts_to_compare.json --benchmark prompts/benchmarks/translation_testset.json --model-alias gpt-4-turbo
    ```

### 4. Analyze Results
The output is saved in the `reports` and `results` folders, allowing you to analyze which prompts are most effective and why.

---

## 📁 Project Index

| Project                | Description                               | Status      |
| ---------------------- | ----------------------------------------- | ----------- |
| **Prompt Playground**  | Basic prompt testing & evaluation workflow | ✅ Complete |
| **Prompt Tester CLI**  | Automated prompt scoring & reporting      | ✅ Complete |
| **Prompt Benchmark Suite** | Model comparison across tasks             | ✅ Complete |
| **Prompt-Lab Web App** | Interactive Streamlit prompt tester       | ✅ Complete |

---

## 💬 Project 4 — Prompt-Lab Web App (`v0.4.0`)

An interactive Streamlit dashboard for testing, evaluating, and visualizing prompts in real time.

### ✨ Features

-   Model & task selection (Summarize, Translate, Explain, Paraphrase)
-   Live prompt execution (OpenAI API + offline mock)
-   Automatic CSV logging to `/reports/prompt_lab_runs.csv`
-   Cached history with refresh and sorting
-   Performance chart by model & task

### 🚀 Run Locally

```bash
streamlit run app.py
```