# GEMINI.md - Project Overview

## Project Overview

This project, "prompt-portfolio," is a Python-based toolkit for managing, testing, and evaluating language model (LLM) prompts. It provides a systematic, data-driven approach to prompt engineering, allowing developers to benchmark and compare prompts to optimize performance.

The project consists of three main components:
1.  **`prompt_toolkit` Library:** A core library containing modules for model interaction (`models.py`), prompt evaluation (`evaluation.py`), reporting (`reporting.py`), and utility functions (`utils.py`). It uses `litellm` to interface with various LLM providers.
2.  **Command-Line Interface (`cli.py`):** A powerful CLI built with `argparse` that serves as the main entry point for running experiments. It supports three primary subcommands:
    *   `run`: Executes a series of prompts from a dataset.
    *   `benchmark`: Evaluates a single prompt's performance against a predefined benchmark test set.
    *   `compare`: Compares the performance of multiple prompts against the same benchmark.
3.  **Streamlit Web App (`app.py`):** An interactive web-based dashboard for quick, real-time prompt testing and visualization.

## Building and Running

### 1. Installation

First, install the required Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file in the root of the project to store your API keys. For example:

```
OPENAI_API_KEY="your-api-key"
```

### 3. Running the CLI

The `cli.py` script is the primary tool for evaluation.

**Benchmark a single prompt:**

```bash
python cli.py benchmark --prompt prompts/library/translation_prompt.json --benchmark prompts/benchmarks/translation_testset.json --model-alias gpt120b
```

**Compare multiple prompts:**

```bash
python cli.py compare --project experiments/prompt-comparison/prompts_to_compare.json --benchmark prompts/benchmarks/translation_testset.json --model-alias gpt120b
```

**Run a dataset of prompts:**

```bash
python cli.py run --data datasets/sample_inputs.json --model-alias gpt120b
```

### 4. Running the Web App

To launch the interactive Streamlit dashboard, run:

```bash
streamlit run app.py
```

## Development Conventions

### Directory Structure

*   `config/`: Contains configuration files, primarily `models_config.json`, which defines the available LLM models and their aliases.
*   `datasets/`: Holds JSON files with sample inputs for running prompts.
*   `experiments/`: Contains JSON files that define "projects" or "experiments" for the `compare` command, listing multiple prompts to be tested.
*   `media/`: Default directory for saving generated charts and images from reports.
*   `notebooks/`: Jupyter notebooks for exploration and experimentation.
*   `output/`: The primary directory for all generated results.
    *   `raw/`: Stores raw, machine-readable output from CLI runs (e.g., CSV files).
    *   `summaries/`: Stores human-readable reports and summaries (e.g., Markdown files).
*   `prompts/`: Contains all prompt-related JSON files.
    *   `library/`: Stores reusable prompt templates.
    *   `benchmarks/`: Stores test sets with inputs and expected outputs for benchmarking prompts.
*   `prompt_toolkit/`: The core Python library for the project.

### Workflow

1.  **Define Prompts:** Create new prompt templates in `prompts/library/`.
2.  **Define Benchmarks:** Create test cases in `prompts/benchmarks/`.
3.  **Run Evaluations:** Use `cli.py` to run benchmarks and comparisons.
4.  **Analyze Results:** Check the `output/` directory for raw CSV data and Markdown summary reports.
5.  **Interactive Testing:** Use the Streamlit app (`app.py`) for quick, ad-hoc prompt testing.
