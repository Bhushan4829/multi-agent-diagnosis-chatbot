import torch
import logging
from typing import List, Dict
from diagnosis_pipeline.icd_mapper import ICD10Mapper

logger = logging.getLogger(__name__)

class DiseasePredictor:
    def __init__(self, tokenizer, model, icd_mapper: ICD10Mapper, fine_db):
        self.tokenizer = tokenizer
        self.model = model
        self.icd_mapper = icd_mapper
        self.fine_db = fine_db

    def predict(self, symptoms: List[str], top_k: int = 5) -> List[Dict]:
        """Predict diseases using fine-tuned LLM and map ICD-10 codes."""
        self.model.set_adapter("diagnosis")  # Activate adapter if using PEFT

        prompt = f"### Symptoms:\n{', '.join(symptoms)}\n\n### Diagnosis:\n"
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=32,
                    num_beams=5,
                    num_return_sequences=top_k,
                    early_stopping=True,
                    output_scores=True,
                    return_dict_in_generate=True
                )

            predictions = []
            if hasattr(outputs, 'sequences_scores'):
                for seq, score in zip(outputs.sequences, outputs.sequences_scores):
                    decoded = self.tokenizer.decode(seq, skip_special_tokens=True)
                    disease_name = decoded.split("### Diagnosis:")[-1].strip().lower()
                    disease_name = disease_name.split('\n')[0].strip()

                    local_match = self.fine_db[
                        self.fine_db["disease"].str.lower() == disease_name
                    ]
                    if not local_match.empty:
                        icd_code = local_match.iloc[0]["ICD-10 Code"]
                    else:
                        code_map = self.icd_mapper.get_codes(disease_name)
                        icd_code = code_map.get(disease_name, "Unknown")

                    confidence = torch.exp(score).item()
                    predictions.append({
                        "disease": disease_name,
                        "icd10": icd_code,
                        "confidence": confidence
                    })

            return sorted(predictions, key=lambda x: x["confidence"], reverse=True)

        except Exception as e:
            logger.error(f"[DiseasePredictor] Prediction failed: {e}")
            return []