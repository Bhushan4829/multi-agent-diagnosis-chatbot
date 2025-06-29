import logging
import json
from openai import OpenAI
from typing import List

logger = logging.getLogger(__name__)

class SymptomExtractor:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    def extract(self, user_text: str) -> List[str]:
        messages = [
            {"role": "system", "content": (
                "You are a medical assistant. "
                "Extract all symptom keywords from the user’s free-text description. "
                "⚠️ Respond _only_ with a JSON array of symptom strings—no explanations or extra text."
            )},
            {"role": "user", "content": user_text}
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            symptoms = json.loads(content)
            if isinstance(symptoms, list):
                return symptoms
            else:
                raise ValueError(f"Expected a JSON list, got: {type(symptoms)}")
        except Exception as e:
            logger.error(f"[SymptomExtractor] Error: {e}")
            return []
