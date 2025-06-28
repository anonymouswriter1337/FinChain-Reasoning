from vllm import LLM, SamplingParams
from typing import List, Dict

# Model ID
model_id = "Qwen/Qwen2.5-7B"

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

# Build chat-style message
def build_conversations(questions: List[str]) -> List[List[Dict[str, str]]]:
    conversations = []
    for question in questions:
        chat = [
            # Uncomment if Qwen expects a system prompt
            # {"role": "system", "content": "You are a finance chatbot."},
            {"role": "user", "content": (
                "Please answer the given question, and provide a step-by-step solution. "
                "Using the format: Step 1: ..., Step 2: ..., ...\n"
                f"The question is:\n{question}"
            )}
        ]
        conversations.append(chat)
    return conversations

# Batch generation using vLLM
def batch_generate(questions: List[str]) -> List[str]:
    conversations = build_conversations(questions)
    outputs = llm.chat(conversations, sampling_params=sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]

# Single example generation
def generation(question: str) -> str:
    return batch_generate([question])[0]
