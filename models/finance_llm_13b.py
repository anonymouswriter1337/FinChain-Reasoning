import os
import logging
from vllm import LLM, SamplingParams

model_id = "AdaptLLM/finance-LLM-13B"

logging.info("Loading Model")
# Initialize vLLM model
llm = LLM(model=model_id, 
          dtype="bfloat16",                             # Match your original setup
          tensor_parallel_size=4,                       # Change to >1 if using multi-GPU
          max_num_seqs=32, 
          gpu_memory_utilization=0.85, 
          download_dir=os.getenv("HF_CACHE")+"/models") # HuggingFace Cache Directory for Models

# Sampling config
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=4096,
)

system_prompt = ""
prompt_template = """<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_prompt} [/INST]"""

# Core batch generation function
def batch_generate(questions):
    prompts = []
    logging.info("Formatting prompt batch")
    for question in questions:
        user_prompt = f"Please answer the given question, and provide a step-by-step solution. Using the format: Step 1: ..., Step 2: ..., ...\n The question is:\n{question}"
        prompts.append(prompt_template.format(system_prompt=system_prompt, user_prompt=user_prompt))

    outputs = llm.generate(prompts, sampling_params)

    # Extract generated completions
    return [
        o.outputs[0].text.strip()
        for o in outputs
    ]
