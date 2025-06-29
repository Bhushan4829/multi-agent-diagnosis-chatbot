# diagnosis_pipeline/session_orchestrator.py

from typing import Optional, Dict, Any, List, Set
import logging
import re
from collections import defaultdict

from diagnosis_pipeline.medical_assistant import MedicalAssistant
CONFIDENCE_HIGH = 0.8  # Adjust as needed

class ConversationProfile:
    REQUIRED = ['age', 'sex', 'weight', 'height']

    def __init__(self):
        self.data: Dict[str, str] = {}
        self.original_query: Optional[str] = None
        self.demographics_asked = False

    def needs(self) -> bool:
        return any(field not in self.data for field in self.REQUIRED)

    def get_demographics_prompt(self) -> str:
        return (
            "Before we proceed with your diagnosis, I'd like to know some basic information "
            "to provide better care. Could you please share your:\n"
            "- Age (in years)\n"
            "- Sex or gender\n"
            "- Weight (in kg)\n"
            "- Height (in cm)\n\n"
            "For example: 'I'm 35 years old, male, 70 kg, 175 cm'"
        )

    def extract_demographics(self, text: str) -> Dict[str, str]:
        demographics = {}

        if age_match := re.search(r'(\d+)\s*years?', text, re.IGNORECASE):
            demographics['age'] = age_match.group(1)
        elif age_match := re.search(r'\b(\d{2})\b', text):
            demographics['age'] = age_match.group(1)

        if re.search(r'\bmale\b|\bman\b|boy', text, re.IGNORECASE):
            demographics['sex'] = 'male'
        elif re.search(r'\bfemale\b|\bwoman\b|girl', text, re.IGNORECASE):
            demographics['sex'] = 'female'

        if weight_match := re.search(r'(\d+)\s*kg|\b(\d+)\s*kilos', text, re.IGNORECASE):
            demographics['weight'] = weight_match.group(1) or weight_match.group(2)
        elif weight_match := re.search(r'weighs?\s*(\d+)', text, re.IGNORECASE):
            demographics['weight'] = weight_match.group(1)

        if height_match := re.search(r'(\d+)\s*cm|\b(\d+)\s*centimeters', text, re.IGNORECASE):
            demographics['height'] = height_match.group(1) or height_match.group(2)
        elif height_match := re.search(r'\b(\d{3})\s*(?!kg|kilos)', text, re.IGNORECASE):
            demographics['height'] = height_match.group(1)

        return demographics


class SessionOrchestrator:
    def __init__(self, assistant: MedicalAssistant):
        self.assistant = assistant
        self.profile = ConversationProfile()
        self.logger = logging.getLogger(__name__)
        self._in_diagnosis = False
        self._awaiting_demographics = False
        self.pending_symptoms: List[str] = []
        self.last_question: Optional[str] = None
        self.followup_count: int = 0
        self.asked_dims: Set[str] = set()
        self.current_predictions: List[Dict] = []

    def _reset_diagnosis_state(self):
        self._in_diagnosis = False
        self._awaiting_demographics = False
        self.pending_symptoms = []
        self.last_question = None
        self.followup_count = 0
        self.asked_dims.clear()
        self.current_predictions = []

    def _get_final_diagnosis_response(self, top_prediction: Dict) -> str:
        reasoning = self.assistant.generate_reasoning(
            self.pending_symptoms,
            top_prediction,
            self.profile.data,
            last_user_input=self.last_question or ""
        )

        treatment = (
            self.assistant.meta_df.loc[
                self.assistant.meta_df['disease'] == top_prediction['disease'],
                'treatment'
            ].squeeze()
            if not self.assistant.meta_df.empty
            else 'No established treatment found'
        )

        precautions = self.assistant.generate_precautions(
            top_prediction['disease'],
            treatment
        )

        return (
            f"📋 **Diagnosis:** {top_prediction['disease']} (ICD-10: {top_prediction['icd10']}, Confidence: {top_prediction['confidence']:.0%})\n\n"
            f"**Summary:** {reasoning.get('summary', '')}\n\n"
            f"**Treatment:** {treatment}\n\n"
            f"**Precautions:** {precautions}"
        )

    def _evaluate_predictions_and_respond(self) -> str:
        rag_predictions = self.assistant.rag_lookup(self.pending_symptoms, top_k=5)
        llm_predictions = self.assistant.predict_diseases(self.pending_symptoms, top_k=5)

        self.current_predictions = self.assistant.evaluate_predictions(
            rag_predictions,
            llm_predictions,
            self.pending_symptoms
        )

        if not self.current_predictions:
            self._reset_diagnosis_state()
            return "⚠️ Could not determine likely conditions. Please try describing your symptoms differently."

        top_prediction = self.current_predictions[0]

        if (top_prediction['confidence'] < CONFIDENCE_HIGH and self.followup_count < 3):
            followups = self.assistant.generate_followups(
                symptoms=self.pending_symptoms,
                predictions=self.current_predictions,
                asked_dims=self.asked_dims,
                patient_profile=self.profile.data,
                last_user_input=self.last_question or ""
            )

            if followups:
                self.followup_count += 1
                self.last_question = followups[0]
                return f"🤔 {self.last_question}"

        response = self._get_final_diagnosis_response(top_prediction)
        self._reset_diagnosis_state()
        return response

    def handle(self, user_input: str) -> str:
        text = user_input.strip()

        if text.lower() in ['/clear', '/reset']:
            self.assistant.clear_memory()
            self.assistant.clear_knowledge()
            self._reset_diagnosis_state()
            return '🗑️ Chat and memory cleared.'

        if self._awaiting_demographics:
            extracted = self.profile.extract_demographics(text)
            self.profile.data.update(extracted)

            if self.profile.needs():
                missing = [f for f in self.profile.REQUIRED if f not in self.profile.data]
                return f"I still need: {', '.join(missing)}."

            self._awaiting_demographics = False
            self._in_diagnosis = True
            self.pending_symptoms = self.assistant.extract_symptoms(self.profile.original_query)
            return self._evaluate_predictions_and_respond()

        if self._in_diagnosis and self.last_question:
            parsed = self.assistant.analyze_response(self.last_question, text)
            new_symptoms = parsed.get('new_symptoms', [])

            if new_symptoms:
                self.pending_symptoms.extend(new_symptoms)
                self.pending_symptoms = list(set(self.pending_symptoms))

            self.assistant.memory.save_context(
                {"input": self.last_question},
                {"output": text}
            )

            self.last_question = None
            return self._evaluate_predictions_and_respond()

        if not self._in_diagnosis:
            intent = self.assistant.classify_intent(text)

            if intent == 'symptom_diagnosis':
                self.profile.original_query = text

                if self.profile.needs():
                    self._awaiting_demographics = True
                    return self.profile.get_demographics_prompt()

                self._in_diagnosis = True
                self.pending_symptoms = self.assistant.extract_symptoms(text)
                return self._evaluate_predictions_and_respond()

            elif intent == 'patient_history':
                return self.assistant.handle_patient_history(text)

            return self.assistant.handle_chat(text)

        return self.assistant.handle_chat(text)
