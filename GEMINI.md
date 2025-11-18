# Project Overview

This project, "Prompt Portfolio," is a Python-based toolkit designed for systematic prompt engineering. It allows developers to manage, test, evaluate, and compare language model prompts to optimize their performance. The project includes a command-line interface (CLI) for running experiments and a web-based application for interactive testing.

## Key Technologies

*   **Backend:** Python
*   **Libraries:**
    *   `litellm`: For interacting with various language model APIs.
    *   `pandas`: For data manipulation and analysis.
    *   `matplotlib`: For generating charts and visualizations.
    *   `textblob`: For text processing and similarity scoring.
    *   `argparse`: For building the command-line interface.
    *   `streamlit`: For the web application.

## Architecture

The project is organized into several key directories:

*   `app.py`: A Streamlit web application for interactive prompt testing.
*   `cli.py`: A command-line interface for running benchmarks and comparisons.
*   `portfolib/`: A library containing the core logic for models, evaluation, and reporting.
*   `prompts/`: A directory for storing prompt templates and benchmark test sets.
*   `datasets/`: Contains datasets for testing prompts.
*   `output/`: Stores the raw results and summary reports from the experiments.
*   `media/`: Contains the charts generated from the experiments.
*   `config/`: Contains configuration files, such as `models_config.json`.

## Building and Running

### Prerequisites

*   Python 3.x
*   Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
*   Create a `.env` file in the root directory and add your `OPENAI_API_KEY`:
    ```
    OPENAI_API_KEY="your-api-key"
    ```

### Running the CLI

The main entry point for running experiments is `cli.py`. It has the following commands:

*   **`benchmark`**: Run a benchmark for a single prompt.
    ```bash
    python cli.py benchmark --prompt prompts/library/translation_prompt.json --benchmark prompts/benchmarks/translation_testset.json --model-alias gpt-4
    ```
*   **`compare`**: Compare multiple prompts against a single benchmark.
    ```bash
    python cli.py compare --project experiments/prompt-comparison/prompts_to_compare.json --benchmark prompts/benchmarks/translation_testset.json --model-alias gpt-4
    ```
*   **`compare-models`**: Compare a single prompt across multiple models.
    ```bash
    python cli.py compare-models --prompt prompts/library/translation_prompt.json --benchmark prompts/benchmarks/translation_testset.json --model-aliases gpt-4 gpt-3.5-turbo
    ```
*   **`run`**: Run a series of prompts from a dataset.
    ```bash
    python cli.py run --data datasets/sample_inputs.json --model-alias gpt-4
    ```

### Running the Web App

The project also includes a Streamlit web application for interactive prompt testing.

```bash
streamlit run app.py
```

## Development Conventions

*   **Prompts:** Prompts are defined in `.json` files in the `prompts/library` directory.
*   **Test Data:** Test cases are defined in `.json` files in the `prompts/benchmarks` directory.
*   **Configuration:** Model configurations are stored in `config/models_config.json`.
*   **Output:** The results of the experiments are saved in the `output` directory, with raw data in `output/raw` and summary reports in `output/summaries`. Charts are saved in the `media` directory.
*   **Code Style:** The code follows standard Python conventions (PEP 8).
