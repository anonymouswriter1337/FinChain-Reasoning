from vllm import LLM, SamplingParams
from typing import List, Dict

# Model ID
model_id = "meta-math/MetaMath-13B-V1.0"

# Initialize vLLM
llm = LLM(
    model=model_id,
    tensor_parallel_size=2,  # Adjust as needed
    gpu_memory_utilization=0.9,
    dtype="bfloat16",        # Use float16 if bfloat16 is unsupported
    max_model_len=4096,
)

# Sampling configuration
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

    # Extract just the generated answers. Although the model is not a reasoner...
    answers = [
        o.outputs[0].text.strip().split("\n\nThinking\n\n")[0]
        for o in outputs
    ]
    return answers

# Single example generation
def generation(question: str) -> str:
    return batch_generate([question])[0]