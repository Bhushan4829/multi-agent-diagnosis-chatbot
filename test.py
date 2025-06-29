# Testing the ICD10Mapper
# from diagnosis_pipeline.icd_mapper import ICD10Mapper
# from dotenv import load_dotenv
# import os

# load_dotenv()

# icd_mapper = ICD10Mapper(
#     client_id=os.getenv("ICD_CLIENT_ID"),
#     client_secret=os.getenv("ICD_CLIENT_SECRET")
# )

# print(icd_mapper.get_codes("fever"))

# Loading the models

import os
import torch
from dotenv import load_dotenv
from diagnosis_pipeline.load_models import load_models

load_dotenv()

def main():
    print("👟 Loading models …")
    tokenizer, model, gen_tokenizer, gen_model = load_models()

    # Print basic info
    print("\n=== Diagnosis Model ===")
    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Model:     {model.__class__.__name__}")
    print(f"Device:    {next(model.parameters()).device}")
    print(f"4-bit?     {'nf4' in str(model)}")

    print("\n=== Chat Model ===")
    print(f"Tokenizer: {gen_tokenizer.__class__.__name__}")
    print(f"Model:     {gen_model.__class__.__name__}")
    print(f"Device:    {next(gen_model.parameters()).device}")

    # Quick forward pass to confirm it works (no gradients)
    prompt = "### Symptoms:\nfever, headache\n\n### Diagnosis:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=8)
    decoded = tokenizer.decode(out[0], skip_special_tokens=True)
    print("\n🩺 Sample output:\n", decoded)

if __name__ == "__main__":
    main()