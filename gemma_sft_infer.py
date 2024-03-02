import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["HF_TOKEN"] = "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"
model_id = "/mnt/d/010Code/009LLM/Gemma/outputs"
tokenize_id = "/mnt/d/010Code/009LLM/Gemma/outputs"
# model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(tokenize_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0},
                                             token=os.environ['HF_TOKEN'])

text = "Quote: Imagination is more"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
