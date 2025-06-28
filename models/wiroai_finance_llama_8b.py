from vllm import LLM, SamplingParams
from typing import List, Dict

# Model ID
model_id = "WiroAI/WiroAI-Finance-Llama-8B"

# Initialize vLLM engine
llm = LLM(
    model=model_id,
    tensor_parallel_size=2,         # Multi-GPU setting
    gpu_memory_utilization=0.9,
    dtype="bfloat16",               # Use "float16" if needed
    max_model_len=4096,
)

# Attempt to extract known stop token IDs
tokenizer = llm.get_tokenizer()
stop_ids = [tokenizer.eos_token_id]

# Try to include <|eot_id|> if it exists
try:
    stop_ids.append(tokenizer.convert_tokens_to_ids("<|eot_id|>"))
except Exception:
    pass  # Safe fallback if token not found

# Sampling configuration
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=4096,
    stop_token_ids=stop_ids,
)

# Batch prompt constructor using system + user chat format
def build_conversations(questions: List[str]) -> List[List[Dict[str, str]]]:
    conversations = []
    for question in questions:
        conversation = [
            # {"role": "system", "content": "You are a finance chatbot developed by Wiro AI."},
            {"role": "user", "content": (
                "Please answer the given question, and provide a step-by-step solution. "
                "Using the format: Step 1: ..., Step 2: ..., ...\n"
                f"The question is:\n{question}"
            )},
        ]
        conversations.append(conversation)
    return conversations

# Batch generation function
def batch_generate(questions: List[str]) -> List[str]:
    conversations = build_conversations(questions)
    outputs = llm.chat(conversations, sampling_params=sampling_params)
    return [o.outputs[0].text.strip() for o in outputs]

# Single-call generation wrapper
def generation(question: str) -> str:
    return batch_generate([question])[0]


