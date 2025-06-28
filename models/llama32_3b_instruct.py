import transformers
import torch

model_id = "meta-llama/Llama-3.2-3B-Instruct"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

def generation(question):

    messages = [
        {"role": "system", "content": "You are a finance chatbot."},
        {"role": "user", "content": f"Please answer the given question, and provide a step-by-step solution. Using the format: Step 1: ..., Step 2: ..., ...\n The question is:\n{question}"}
    ]

    outputs = pipeline(
        messages,
        max_new_tokens=2048,
    )

    return outputs[0]["generated_text"][-1]['content']

