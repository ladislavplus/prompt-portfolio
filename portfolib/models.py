import litellm
from dotenv import load_dotenv
from .utils import load_json

load_dotenv()

def load_models_config(path="config/models_config.json"):
    """Loads the models configuration file."""
    return load_json(path)

def get_model_name(model_alias, models_config=None):
    """Gets the litellm model string from a model alias."""
    if models_config is None:
        models_config = load_models_config()
    
    if model_alias not in models_config["models"]:
        raise ValueError(f"Model alias '{model_alias}' not found in models_config.json.")
    
    return models_config["models"][model_alias]["litellm_string"]

def run_prompt(user_prompt, model_name, system_prompt=None, variables=None):
    """
    Runs a prompt using the specified model via litellm, with optional variable substitution.
    Supports an optional system prompt.
    """
    if variables:
        user_prompt = user_prompt.format(**variables)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    
    try:
        r = litellm.completion(
            model=model_name,
            messages=messages
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error calling litellm for model {model_name}: {e}")
        return f"Error: Could not get response from model {model_name}."
