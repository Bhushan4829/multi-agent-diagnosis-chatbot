import logging
import json
from typing import List, Dict, Any
from openai import OpenAI

logger = logging.getLogger(__name__)

class ReasoningGenerator:
    def __init__(self, openai_api_key: str, memory, knowledge_store):
        self.client = OpenAI(api_key=openai_api_key)
        self.memory = memory
        self.knowledge_store = knowledge_store

    def generate(self, symptoms: List[str], diagnosis: Dict, patient_profile: Dict[str, Any],
                 last_user_input: str, max_history: int = 6) -> Dict[str, str]:

        history = self.memory.load_memory_variables({"input": last_user_input})["chat_history"][-max_history:]
        history_block = "\n".join(f"- {m}" for m in history) if history else ""

        demo_section = ""
        if patient_profile:
            demo_section = (
                f"Patient Info: {patient_profile.get('age','?')}y, "
                f"{patient_profile.get('sex','?')}, "
                f"{patient_profile.get('weight','?')}kg, "
                f"{patient_profile.get('height','?')}cm\n\n"
            )

        query = (
            f"Patient Profile: {demo_section}"
            f"Patient presenting with: {', '.join(symptoms)}\n"
            f"Possible diagnosis: {diagnosis['disease']}\n"
            "Required: pathophysiology, diagnostic criteria, differential diagnosis"
        )

        try:
            chunks = self.knowledge_store.similarity_search(query, k=5)
        except Exception as e:
            logger.warning(f"[ReasoningGenerator] Knowledge search failed: {e}")
            chunks = []

        knowledge_text = "\n".join("- " + c.page_content for c in chunks) if chunks else "No specific evidence found"

        # Chain-of-Thought prompt
        prompt = f"""Perform clinical reasoning step-by-step:
Context: {history_block} {demo_section}
Patient presenting with: {', '.join(symptoms)}
Potential Diagnosis: {diagnosis['disease']} (ICD-10: {diagnosis['icd10']}, Confidence: {diagnosis['confidence']:.0%})
Relevant Medical Knowledge: {knowledge_text}

1. Pathophysiological Basis
2. Diagnostic Criteria
3. Symptom Match
4. Differential Diagnosis
5. Evidence Evaluation
6. Confidence Assessment

Final Analysis:"""

        system_msg = (
            "You are a medical assistant. Provide structured clinical reasoning. "
            "Always end with: 'This is not medical advice—please consult a qualified healthcare professional.'"
        )

        try:
            # Step-by-step reasoning
            resp = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            reasoning = resp.choices[0].message.content.strip()

            # Condensed summary
            summary_prompt = (
                f"Here is a detailed clinical reasoning:\n{reasoning}\n\n"
                "Please condense that into one paragraph, ending with the disclaimer:"
                " 'This is not medical advice—please consult a qualified healthcare professional.'"
            )
            sum_resp = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.3
            )
            summary = sum_resp.choices[0].message.content.strip()

            return {
                "steps": reasoning,
                "summary": summary
            }

        except Exception as e:
            logger.error(f"[ReasoningGenerator] Error generating reasoning: {e}")
            return {
                "steps": "Step-by-step reasoning not available.",
                "summary": ""
            }
