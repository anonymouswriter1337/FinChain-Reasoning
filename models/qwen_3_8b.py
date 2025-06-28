from vllm import LLM, SamplingParams

# Initialize vLLM model
llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=2,         # Change to >1 if using multi-GPU
    gpu_memory_utilization=0.85,
    dtype="bfloat16",               # Match your original setup
    max_model_len=4096              # Supports longer generation
)

# Sampling config
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=4096,
)

# Core batch generation function
def batch_generate(questions):
    prompts = [
        f"Please answer the given question, and provide a step-by-step solution. "
        f"Using the format: Step 1: ..., Step 2: ..., ...\n"
        f"The question is:\n{q}" for q in questions
    ]

    outputs = llm.generate(prompts, sampling_params)

    # Extract generated completions
    return [
        o.outputs[0].text.strip().split("\n\nThinking\n\n")[0]
        for o in outputs
    ]

