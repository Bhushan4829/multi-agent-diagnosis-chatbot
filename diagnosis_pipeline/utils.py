from typing import List

def normalize(xs: List[float]) -> List[float]:
    """Normalize a list of floats between 0 and 1."""
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    return [(x - lo) / (hi - lo) if hi > lo else 0.5 for x in xs]

def generate_fallback_reasoning(symptoms: List[str], diagnosis: dict) -> str:
    """Simpler fallback reasoning message if LLM fails."""
    return (
        f"Based on symptoms {', '.join(symptoms)}, the most likely condition is {diagnosis['disease']}.\n"
        f"Confidence level: {diagnosis['confidence']:.0%}.\n"
        "Please consult a healthcare provider for a definitive diagnosis."
    )
