from difflib import SequenceMatcher
from textblob import TextBlob

def score_similarity(expected, output):
    """Return simple similarity score between 0–1."""
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
