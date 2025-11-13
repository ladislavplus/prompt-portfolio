# app.py
import streamlit as st
from dotenv import load_dotenv
import os
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

from prompt_toolkit.models import get_model_name, run_prompt as run_api_prompt
from prompt_toolkit.utils import load_json

# --- Load environment ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Streamlit Page Config ---
st.set_page_config(page_title="Prompt Playground", page_icon="💬", layout="wide")

st.title("💬 Prompt Playground")
st.caption("Interactive prompt testing dashboard for OpenAI models")

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")
models_config = load_json("config/models_config.json")
model_aliases = list(models_config["models"].keys())
model_alias_choice = st.sidebar.selectbox(
    "Choose Model",
    model_aliases,
    index=model_aliases.index(models_config.get("default_model", model_aliases[0]))
)

task_choice = st.sidebar.selectbox(
    "Choose Task",
    ["Summarize", "Translate", "Explain", "Paraphrase"],
    index=0
)

use_api = st.sidebar.checkbox("Use OpenAI API", value=True)
st.sidebar.write(f"**Active Model:** `{model_alias_choice}`")
st.sidebar.write(f"**Mode:** {'API' if use_api else 'Mock'}")

# --- Helper Function ---
def run_prompt(prompt: str, model_alias: str, use_api: bool = False):
    """
    Run prompt through API (if enabled) or mock fallback.
    Returns (response_text, mode)
    """
    if use_api:
        model_name = get_model_name(model_alias)
        response = run_api_prompt(prompt, model_name)
        return response, "api"

    # --- Mock fallback ---
    mock_responses = {
        "Summarize": "Here’s a short summary of your text (mock).",
        "Translate": "Voici la traduction simulée de votre texte (mock).",
        "Explain": "This is a simplified explanation of your content (mock).",
        "Paraphrase": "Here’s a paraphrased version of your text (mock)."
    }
    return mock_responses.get(task_choice, "Mock response."), "mock"

# --- Main Area ---
st.subheader("Enter Your Prompt")
user_input = st.text_area("Prompt Input", height=150, placeholder="Type your prompt here...")

if st.button("🚀 Run Prompt"):
    if not user_input.strip():
        st.warning("Please enter a prompt before running.")
    else:
        with st.spinner("Running prompt..."):
            start = time.time()
            output, mode = run_prompt(user_input, model_alias_choice, use_api)
            elapsed = round(time.time() - start, 2)

        st.info(f"**Model:** {model_alias_choice} | **Task:** {task_choice}")
        st.success(f"Response generated ({mode.upper()} mode, {elapsed}s)")
        st.text_area("🧠 Model Output", value=output, height=200)

        # Save run to reports
        reports_dir = "output/summaries"
        os.makedirs(reports_dir, exist_ok=True)
        out_path = os.path.join(reports_dir, "prompt_lab_runs.csv")

        run_data = pd.DataFrame([{
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_alias_choice,
            "task": task_choice,
            "prompt": user_input,
            "output": output,
            "mode": mode,
            "elapsed_s": elapsed
        }])

        if os.path.exists(out_path):
            old = pd.read_csv(out_path)
            run_data = pd.concat([old, run_data], ignore_index=True)

        run_data.to_csv(out_path, index=False)
        st.write(f"✅ Run saved to `{out_path}`")

# --- History Section ---
st.divider()
st.subheader("📜 Run History")

@st.cache_data
def load_history(path="output/summaries/prompt_lab_runs.csv"):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["timestamp", "model", "task", "prompt", "output", "mode", "elapsed_s"])

history_path = "output/summaries/prompt_lab_runs.csv"
df = load_history(history_path)

col1, col2 = st.columns([1, 4])
with col1:
    if st.button("🔄 Refresh History"):
        st.cache_data.clear()
        st.rerun()

if df.empty:
    st.info("No previous runs yet. Run a prompt to generate history.")
else:
    st.success(f"Loaded {len(df)} total runs.")
    st.dataframe(df.tail(10).sort_values(by="timestamp", ascending=False), use_container_width=True)

    with st.expander("🔍 View Full History Table"):
        st.dataframe(df, use_container_width=True)

st.divider()
st.subheader("📊 Performance Overview")

if not df.empty:
    avg_time = (
        df.groupby(["model", "task"])["elapsed_s"]
        .mean()
        .reset_index()
        .sort_values("elapsed_s", ascending=True)
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    for task in avg_time["task"].unique():
        subset = avg_time[avg_time["task"] == task]
        ax.barh(subset["model"] + " (" + task + ")", subset["elapsed_s"], label=task)

    ax.set_xlabel("Average Response Time (s)")
    ax.set_ylabel("Model + Task")
    ax.set_title("Average Prompt Response Time by Model & Task")
    st.pyplot(fig)

else:
    st.info("No performance data yet. Run a few prompts to populate metrics.")