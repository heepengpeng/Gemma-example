import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["HF_TOKEN"] = "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"
model_id = "/mnt/e/002hfmodel/hub/models--google--gemma-2b/snapshots/9d067f00def958594aaa16b39a65b07d69ca655b"
# model_id = "google/gemma-2b"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HF_TOKEN'])
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0},
                                             token=os.environ['HF_TOKEN'])

text = "Quote: Imagination is more"
device = "cuda:0"
inputs = tokenizer(text, return_tensors="pt").to(device)

outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

os.environ["WANDB_DISABLED"] = "true"

from peft import LoraConfig

lora_config = LoraConfig(
    r=8,
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

from datasets import load_dataset
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

import transformers
from trl import SFTTrainer


def formatting_func(example):
    text = f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}"
    return [text]


trainer = SFTTrainer(
    model=model,
    train_dataset=data["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        max_steps=10,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit"
    ),
    peft_config=lora_config,
    formatting_func=formatting_func,
)
trainer.train()

trainer.save_model("outputs")
