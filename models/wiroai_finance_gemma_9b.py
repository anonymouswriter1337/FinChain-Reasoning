from vllm import LLM, SamplingParams
from typing import List, Dict

# Model ID
model_id = "WiroAI/WiroAI-Finance-Gemma-9B"

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


# from vllm import LLM, SamplingParams
# from typing import List

# # Initialize LLM
# model_id = "WiroAI/WiroAI-Finance-Gemma-9B"

# llm = LLM(
#     model=model_id,
#     tensor_parallel_size=2,
#     gpu_memory_utilization=0.85,
#     dtype="bfloat16",
#     max_model_len=4096,
# )

# sampling_params = SamplingParams(
#     temperature=0.7,
#     top_p=0.95,
#     max_tokens=4096,
#     stop_token_ids=[
#         llm.get_tokenizer().eos_token_id,
#         llm.get_tokenizer().convert_tokens_to_ids("<|eot_id|>"),
#     ],
# )

# # Format message to ChatML prompt
# def chat_prompt(question: str) -> str:
#     return (
#         "<|user|>\n"
#         f"Please answer the given question, and provide a step-by-step solution. "
#         f"Using the format: Step 1: ..., Step 2: ..., ...\n"
#         f"The question is:\n{question}\n\n"
#         "<|assistant|>\n"
#     )

# # Extract assistant's response
# def extract_assistant_response(text: str) -> str:
#     if "<|assistant|>" in text:
#         return text.split("<|assistant|>")[-1].strip()
#     return text.strip()

# # Batch generation
# def batch_generate(questions: List[str]) -> List[str]:
#     prompts = [chat_prompt(q) for q in questions]
#     outputs = llm.generate(prompts, sampling_params)
#     return [extract_assistant_response(o.outputs[0].text) for o in outputs]

# # Single example wrapper
# def generation(question: str) -> str:
#     return batch_generate([question])[0]


# import transformers
# import torch


# model_id = "WiroAI/WiroAI-Finance-Gemma-9B"

# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map="auto",
# )

# pipeline.model.eval()

# def generation(question):

#     messages = [
#         {"role": "user", "content": f"Please answer the given question, and provide a step-by-step solution. Using the format: Step 1: ..., Step 2: ..., ...\n The question is:\n{question}"
#     },
#     ]

#     terminators = [
#         pipeline.tokenizer.eos_token_id,
#         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs = pipeline(
#         messages,
#         max_new_tokens=2048,
#         eos_token_id=terminators,
#         do_sample=True,
#         temperature=0.9,
#     )

#     return outputs[0]["generated_text"][-1]['content']
