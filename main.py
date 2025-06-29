# main.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from diagnosis_pipeline.load_models import (
    load_symptom_extraction_model,
    load_disease_prediction_model,
    load_reasoning_pipeline
)
from diagnosis_pipeline.medical_assistant import MedicalAssistant
from diagnosis_pipeline.session_orchestrator import SessionOrchestrator
import os
from dotenv import load_dotenv

load_dotenv()

# ───────── Load models ──────────
symptom_tokenizer, symptom_model = load_symptom_extraction_model()
disease_tokenizer, disease_model = load_disease_prediction_model()
gen_model = load_reasoning_pipeline()

# ───────── Init assistant ───────
assistant = MedicalAssistant(
    tokenizer=symptom_tokenizer,
    model=symptom_model,
    gen_tokenizer=disease_tokenizer,
    gen_model=gen_model,
    disease_csv_path="data/disease_prediction_cleaned_deduplicated.csv",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    pinecone_index_name=os.getenv("PINECONE_INDEX")
)

# ───────── Init session manager ───────
session = SessionOrchestrator(assistant)

# ───────── FastAPI Setup ───────
app = FastAPI()

class Query(BaseModel):
    message: str

@app.post("/chat")
async def chat_handler(query: Query):
    reply = session.handle(query.message)
    return {"response": reply}
