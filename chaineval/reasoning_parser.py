import openai, os
import json
from tqdm import tqdm
import asyncio
# from async_openai import AsyncOpenAI

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set!")

# Set your OpenAI key
client = openai.AsyncOpenAI(api_key=api_key)


def build_prompt(narrative: str) -> str:
    return f"""
You are a precise financial reasoning assistant. Your task is to extract exact financial reasoning steps from a free-form explanation and convert them into clearly labeled, structured steps. Focus only on the computational steps that directly contribute to the final answer. Omit any redundant comments or human-like speculation.

Extract the financial reasoning steps from the following narrative and return them in a structured format:

\"\"\"{narrative}\"\"\"

Return your answer in this format:
Step 1: ...
Step 2: ...
...
"""

# models_to_parse = [
#     "deepseek_r1_distill_llama_8b.jsonl",
#     "deepseek_r1_distill_llama_70b.jsonl",
#     "deepseek_r1_distill_qwen_7b.jsonl",
#     "deepseek_r1_distill_qwen_32b.jsonl",
#     "fino1_8b.jsonl",
#     "fin_r1.jsonl",
#     "qwen_3_8b.jsonl",
#     "wizardmath_7b.jsonl",
#     "metamath_13b.jsonl"
# ]

gpt_models = [
    "gpt-4.1-mini.jsonl",
    "gpt-4o-mini.jsonl",
    "o4-mini.jsonl",
    "o3-mini.jsonl",
    "gpt-4.1.jsonl"
]

# # Dhruv's models
# models_to_parse = [
#     "deepseek_r1_distill_llama_8b.jsonl",
#     "deepseek_r1_distill_llama_70b.jsonl",
#     "deepseek_r1_distill_qwen_7b.jsonl",
#     "deepseek_r1_distill_qwen_32b.jsonl",
#     "fino1_8b.jsonl"
# ]

# # Zhuohan's models
# models_to_parse = [
#     "fin_r1.jsonl",
#     "qwen_3_8b.jsonl",
#     "wizardmath_7b.jsonl",
#     "metamath_13b.jsonl"
# ]


async def extract_reasoning_steps_async(narrative: str, model="gpt-4o-mini") -> str:
    prompt = build_prompt(narrative)
    chat_completion = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a financial reasoning extraction tool."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return chat_completion.choices[0].message.content.strip()

async def process_file(model_file):
    with open(f'../results/{model_file}', 'r') as f_pred:
        with open(f'../results/reasoning_parsed/{model_file}', 'w') as f_out:
            lines = f_pred.readlines()
            tasks = []
            for line in lines:
                json_line = json.loads(line)
                narrative = json_line['generation']
                tasks.append((json_line, extract_reasoning_steps_async(narrative)))
            
            for json_line, task in tqdm(tasks, desc="Parsing reasoning steps for: " + model_file, total=len(tasks)):
                reasoning_steps = await task
                json_line['generation_parsed'] = reasoning_steps
                f_out.write(json.dumps(json_line) + '\n')

async def main():
    await asyncio.gather(*[process_file(model_file) for model_file in gpt_models])

if __name__ == "__main__":
    asyncio.run(main())

# def extract_reasoning_steps(narrative: str, model="gpt-4o-mini") -> str:
#     prompt = build_prompt(narrative)
#     chat_completion = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system", "content": "You are a financial reasoning extraction tool."},
#             {"role": "user", "content": prompt}
#         ],
#         temperature=0
#     )
#     return chat_completion.choices[0].message.content.strip()

# for model_file in models_to_parse:
#     with open(f'../results/{model_file}', 'r') as f_pred:
#         with open(f'../results/reasoning_parsed/{model_file}', 'w') as f_out:
#             for line in tqdm(f_pred, desc="Parsing reasoning steps for: " + model_file, total = 2700):
#                 json_line = json.loads(line)
#                 narrative = json_line['generation']
#                 reasoning_steps = extract_reasoning_steps(narrative)
#                 json_line['generation_parsed'] = reasoning_steps
#                 f_out.write(json.dumps(json_line) + '\n')