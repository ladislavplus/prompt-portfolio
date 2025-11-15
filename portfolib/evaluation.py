import json
import re
from difflib import SequenceMatcher
from textblob import TextBlob

def normalize_value(value):
    """Normalizes a value for comparison by converting to float if possible."""
    if isinstance(value, (int, float)):
        return float(value)
    
    if isinstance(value, str):
        # Lowercase, strip whitespace, and remove common currency/separators
        s = value.lower().strip()
        s = re.sub(r'[$,€,kč,£]', '', s)
        s = s.replace(' ', '').replace(',', '')

        # Attempt to convert to float
        try:
            return float(s)
        except ValueError:
            # If not a number, return the cleaned string
            return s
    
    return value

def score_similarity(expected, output):
    """
    Scores similarity. If expected is JSON, compares data semantically. 
    Otherwise, uses text similarity.
    """
    try:
        expected_json = json.loads(expected)
        
        try:
            output_json = json.loads(output)
            
            if not isinstance(expected_json, dict) or not isinstance(output_json, dict):
                return 1.0 if expected_json == output_json else 0.0

            matching_fields = 0
            total_fields = len(expected_json)
            
            if total_fields == 0:
                return 1.0 if len(output_json) == 0 else 0.0

            for key, expected_val in expected_json.items():
                if key in output_json:
                    output_val = output_json[key]
                    
                    normalized_expected = normalize_value(expected_val)
                    normalized_output = normalize_value(output_val)
                    
                    if normalized_expected == normalized_output:
                        matching_fields += 1
            
            return round(matching_fields / total_fields, 3)

        except json.JSONDecodeError:
            return 0.0
            
    except (json.JSONDecodeError, TypeError):
        return round(SequenceMatcher(None, str(expected).lower(), str(output).lower()).ratio(), 3)

def score_relevance(output, prompt):
    """Scores the relevance of the output to the prompt."""
    return min(len(set(output.split()) & set(prompt.split())) / max(len(prompt.split()), 1) * 10, 5)

def score_clarity(output):
    """Scores the clarity of the output using TextBlob."""
    blob = TextBlob(output)
    return round(5 - (abs(blob.sentiment.polarity) * 2), 2)

def score_completeness(output):
    """Scores the completeness of the output based on its length."""
    length = len(output.split())
    if length < 10: return 1
    elif length < 30: return 3
    else: return 5

def evaluate_run(output, prompt):
    """Evaluates a single prompt run."""
    r_score = score_relevance(output, prompt)
    c_score = score_clarity(output)
    comp_score = score_completeness(output)
    total = round((r_score + c_score + comp_score) / 3, 2)
    return {
        "relevance": r_score,
        "clarity": c_score,
        "completeness": comp_score,
        "avg_score": total
    }
