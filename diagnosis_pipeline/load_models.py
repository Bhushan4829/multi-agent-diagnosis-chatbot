# # diagnosis_pipeline/load_models.py

# from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
# import torch
# import os
# from dotenv import load_dotenv
# import bitsandbytes
# load_dotenv()
# HF_TOKEN = os.getenv("HF_TOKEN")

# def load_symptom_extraction_model():
#     tokenizer = AutoTokenizer.from_pretrained(
#         "bhushan4829/medalpaca-7b-symptom-disease-diagnosis_wr",
#         token=HF_TOKEN
#     )
#     model = AutoModelForSequenceClassification.from_pretrained(
#         "bhushan4829/medalpaca-7b-symptom-disease-diagnosis_wr",
#         token=HF_TOKEN
#     )
#     return tokenizer, model

# def load_disease_prediction_model():
#     tokenizer = AutoTokenizer.from_pretrained(
#         "Bhushan4829/medalpaca-7b-disease-predictor",
#         token=HF_TOKEN
#     )
#     model = AutoModelForSequenceClassification.from_pretrained(
#         "Bhushan4829/medalpaca-7b-disease-predictor",
#         token=HF_TOKEN
#     )
#     return tokenizer, model

# def load_reasoning_pipeline():
#     pipe = pipeline(
#         "text-generation",
#         model="deepseek-ai/deepseek-coder-6.7b-base",
#         device_map="auto" if torch.cuda.is_available() else None,
#         token=HF_TOKEN
#     )
#     return pipe

# diagnosis_pipeline/load_models.py

import os
import torch
import logging
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from peft import PeftModel
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN")
BASE_MODEL_REPO = "medalpaca/medalpaca-7b"
PEFT_MODEL_REPO = "bhushan4829/medalpaca-7b-symptom-disease-diagnosis_wr"
CHAT_MODEL_REPO = "meta-llama/Llama-3.2-1B-Instruct"
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/mnt/models")  # your attached EBS path
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = MODEL_CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR
def load_models():
    """Load base and generation models with caching and quantization."""
    logger.info("Loading models...")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    try:
        # Base model for diagnosis
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_REPO,
            cache_dir=MODEL_CACHE_DIR,
            token=HF_TOKEN,
            use_fast=False,
            padding_side="right"
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_REPO,
            cache_dir=MODEL_CACHE_DIR,
            token=HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=quant_config
        )
        base_model.generation_config.pad_token_id = tokenizer.eos_token_id
        base_model.config.pad_token_id = tokenizer.eos_token_id

        model = PeftModel.from_pretrained(
            base_model,
            PEFT_MODEL_REPO,
            adapter_name="diagnosis",
            cache_dir=MODEL_CACHE_DIR,
            token=HF_TOKEN
        )
        model.eval()

        # Chat model for reasoning and conversation
        gen_tokenizer = AutoTokenizer.from_pretrained(
            CHAT_MODEL_REPO,
            cache_dir=MODEL_CACHE_DIR,
            token=HF_TOKEN,
            use_fast=True,
            padding_side="right"
        )
        gen_model = AutoModelForCausalLM.from_pretrained(
            CHAT_MODEL_REPO,
            cache_dir=MODEL_CACHE_DIR,
            token=HF_TOKEN,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        gen_model.eval()

        return tokenizer, model, gen_tokenizer, gen_model

    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise
