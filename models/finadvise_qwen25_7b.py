import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "nguyentd/FinancialAdvice-Qwen2.5-7B"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto").to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def generation(question):
    # Put your input here:
    user_input = f"Please answer the given question, and provide a step-by-step solution. Using the format: Step 1: ..., Step 2: ..., ...\n The question is:\n{question}"
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=2048
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response