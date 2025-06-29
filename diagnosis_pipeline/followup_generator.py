import logging
from typing import List, Dict, Set, Optional
from openai import OpenAI

logger = logging.getLogger(__name__)

QUESTION_DIMS = [
    "missing_symptoms",
    "severity", "onset_duration", "frequency",
    "triggers_relievers", "associated_signs",
    "functional_impact", "risk_factors", "treatment_response"
]

class FollowupGenerator:
    def __init__(self, openai_api_key: str, cooc_matrix, memory, max_followups: int = 3):
        self.client = OpenAI(api_key=openai_api_key)
        self.cooc = cooc_matrix
        self.memory = memory
        self.max_followups = max_followups

    def generate(self, symptoms: List[str], predictions: List[Dict], asked_dims: Set[str],
                 patient_profile: Dict = None, last_user_input: str = "") -> List[str]:

        context = self.memory.load_memory_variables({"input": last_user_input})["chat_history"]
        context_str = "\n".join(f"- {m}" for m in context[-3:]) if context else "No previous context"

        confidence = predictions[0]['confidence'] if predictions else 0.0
        remaining_dims = [d for d in QUESTION_DIMS if d not in asked_dims]
        if not remaining_dims:
            return []

        current_dim = remaining_dims[0]
        asked_dims.add(current_dim)

        if current_dim == "missing_symptoms":
            suspected = [p["disease"] for p in predictions[:3]]
            if confidence > 0.8:
                missing = self.cooc.top_candidates(symptoms, suspected, k=1)
            elif confidence > 0.5:
                missing = self.cooc.top_candidates(symptoms, suspected, k=2)
            else:
                missing = self.cooc.top_candidates(symptoms, suspected, k=3)

            if not missing:
                return []

            if confidence > 0.8:
                prompt = f"""Context: {context_str}
Patient Symptoms: {', '.join(symptoms)}
Potential Diagnoses: {', '.join(suspected)}

Ask one precise yes/no question about whether they have: {missing[0]}"""
            else:
                prompt = f"""Context: {context_str}
Patient Symptoms: {', '.join(symptoms)}
Potential Diagnoses: {', '.join(suspected)}
Missing Symptoms: {', '.join(missing)}

Generate one natural follow-up question asking about these symptoms:"""
        else:
            prompt = f"""You're a doctor conducting a diagnosis interview.

Patient Profile: {patient_profile or 'Not specified'}
Current Symptoms: {', '.join(symptoms)}
Top Diagnosis Considered: {predictions[0]['disease']} (confidence: {confidence:.0%})
Conversation History:
{context_str}

Ask one follow-up question about the patient's {current_dim.replace('_', ' ')},
tailored to the confidence level ({'high' if confidence > 0.8 else 'medium' if confidence > 0.5 else 'low'}):"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a caring medical assistant. Use direct patient-friendly language."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            return [response.choices[0].message.content.strip()]
        except Exception as e:
            logger.error(f"[FollowupGenerator] Error: {e}")
            if current_dim == "missing_symptoms":
                return [f"Are you experiencing {missing[0]}?"]
            return [f"Can you tell me more about your {current_dim.replace('_', ' ')}?"]
