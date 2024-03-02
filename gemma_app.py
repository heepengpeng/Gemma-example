import torch
from flask import Flask, jsonify, request
from huggingface_hub import login
from transformers import AutoTokenizer, pipeline

app = Flask(__name__)

login(token="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")
model = r"E:\trl\gemma-finetuned-openassistant\checkpoint-20000"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)


@app.route('/v1/llm', methods=['POST'])
def chat_with_gemma():
    messages = request.get_json()["messages"]
    prompt = pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    response = outputs[0]["generated_text"][len(prompt):]
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host="127.0.0.1", port=8000)
