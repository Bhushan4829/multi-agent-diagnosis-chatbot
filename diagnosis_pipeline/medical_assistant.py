import logging
import pandas as pd
import ast
from typing import Optional, Dict, List, Any, Set
from diagnosis_pipeline.icd_mapper import ICD10Mapper
from diagnosis_pipeline.symptom_extractor import SymptomExtractor
from diagnosis_pipeline.disease_predictor import DiseasePredictor
from diagnosis_pipeline.retriever import MedicalRetriever
from diagnosis_pipeline.followup_generator import FollowupGenerator
from diagnosis_pipeline.reasoning import ReasoningGenerator
from diagnosis_pipeline.utils import generate_fallback_reasoning
from fuzzywuzzy import fuzz
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

class MedicalAssistant:
    def __init__(self,
                 tokenizer,
                 model,
                 gen_tokenizer,
                 gen_model,
                 openai_api_key: str,
                 disease_csv_path: str,
                 pinecone_index=None,
                 memory=None,
                 knowledge_store=None,
                 cooc_matrix=None):

        # Load fine-tuned disease DB
        self.fine_db = pd.read_csv(disease_csv_path)
        self.fine_db['cleaned_symptoms'] = self.fine_db['cleaned_symptoms'].apply(ast.literal_eval)
        self.meta_df = pd.read_csv("/content/drive/MyDrive/merged_diseases.csv")
        self.meta_df["combined_symptoms"] = self.meta_df["combined_symptoms"].apply(ast.literal_eval)

        # Core LLM components
        self.tokenizer = tokenizer
        self.model = model
        self.gen_tokenizer = gen_tokenizer
        self.gen_model = gen_model
        self.openai_api_key = openai_api_key

        # Submodules
        self.icd_mapper = ICD10Mapper(
            client_id=os.getenv("ICD_CLIENT_ID"),
            client_secret=os.getenv("ICD_CLIENT_SECRET")
        )
        self.symptom_extractor = SymptomExtractor(openai_api_key)
        self.predictor = DiseasePredictor(tokenizer, model, self.icd_mapper, self.fine_db)
        self.retriever = MedicalRetriever(openai_api_key, pinecone_index, self.icd_mapper, knowledge_store)
        self.followup_generator = FollowupGenerator(openai_api_key, cooc_matrix, memory)
        self.reasoning_generator = ReasoningGenerator(openai_api_key, memory, knowledge_store)

        # Memory + store
        self.memory = memory
        self.pinecone_index = pinecone_index
        self.knowledge_store = knowledge_store

    def run_diagnosis(self, user_input: str, patient_profile: Optional[Dict[str, Any]] = None) -> Dict:
        symptoms = self.symptom_extractor.extract(user_input)
        if not symptoms:
            return {"status": "error", "message": "Couldn't identify symptoms. Please describe more detail."}

        if patient_profile:
            self.memory.save_context({"input": "patient_profile"}, {"output": str(patient_profile)})

        asked_dims = set()
        followup_count = 0

        for round_i in range(3):
            rag_preds = self.retriever.rag_lookup(symptoms)
            llm_preds = self.predictor.predict(symptoms)

            # Combine by ICD-10 + fuzzy match
            combined = {}
            for pred in rag_preds + llm_preds:
                key = pred["icd10"]
                if key not in combined or pred["confidence"] > combined[key]["confidence"]:
                    combined[key] = pred

            final_preds = sorted(combined.values(), key=lambda x: x["confidence"], reverse=True)
            top = final_preds[0] if final_preds else None

            if not top:
                break

            if top["confidence"] >= 0.8:
                break
            elif top["confidence"] >= 0.5 and followup_count >= 2:
                break
            elif followup_count >= 3:
                break

            followups = self.followup_generator.generate(
                symptoms, final_preds, asked_dims,
                patient_profile, user_input
            )
            if not followups:
                break
            followup_count += 1
            return {
                "status": "followup",
                "followup_questions": followups,
                "current_predictions": final_preds[:3]
            }

        if not final_preds:
            return {"status": "error", "message": "No conditions could be determined."}

        top = final_preds[0]
        reasoning = self.reasoning_generator.generate(
            symptoms, top, patient_profile, last_user_input=user_input
        )

        treatment = self.meta_df.loc[self.meta_df["disease"] == top["disease"], "treatment"].squeeze()
        precautions = f"Please consult your doctor about precautions for {top['disease']}."

        self.memory.save_context({"input": "final_diagnosis"},
                                 {"output": f"{top['disease']} (Confidence: {top['confidence']:.0%})"})

        return {
            "status": "complete",
            "symptoms": symptoms,
            "diagnosis": top,
            "summary": reasoning.get("summary", ""),
            "treatment": treatment,
            "precautions": precautions,
            "confidence": top["confidence"],
            "disclaimer": "⚠️ AI-generated — consult a licensed provider."
        }