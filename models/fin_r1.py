from vllm import LLM, SamplingParams
from typing import List

# --- Setup ---
model_name = "SUFE-AIFLM-Lab/Fin-R1"

llm = LLM(
    model=model_name,
    tensor_parallel_size=2,        # Set >1 for multi-GPU
    gpu_memory_utilization=0.85,
    dtype="float16",               # Use float16 to save memory
    max_model_len=4096             # Optional
)

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=4096,
)

# --- Core Batch Generation Function ---
def batch_generate(questions: List[str]) -> List[str]:
    prompts = [
        f"Please answer the given question, and provide a step-by-step solution. "
        f"Using the format: Step 1: ..., Step 2: ..., ...\nThe question is:\n{q}"
        for q in questions
    ]

    outputs = llm.generate(prompts, sampling_params)

    # Extract just the generated answers
    answers = [
        o.outputs[0].text.strip().split("\n\nThinking\n\n")[0]
        for o in outputs
    ]
    return answers