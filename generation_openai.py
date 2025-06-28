

import argparse
import json
import asyncio
import os
from tqdm import tqdm
from glob import glob
from openai import AsyncOpenAI, RateLimitError, OpenAIError

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set!")

# ----------------- Args -----------------
parser = argparse.ArgumentParser(description='OpenAI: Generate all topics & subtopics into one output.')
parser.add_argument('--model', type=str, default='o4-mini', help='OpenAI model name')
parser.add_argument('--api_key', type=str, default=api_key, help='OpenAI API key')
args = parser.parse_args()

# ----------------- Load All Questions -----------------
all_data = []
input_files = sorted(glob('data/testset/*/*.jsonl'))

print(f"📂 Found {len(input_files)} files... Reading all data.")

for input_file in input_files:
    topic = input_file.split(os.sep)[-2]
    subtopic = os.path.splitext(os.path.basename(input_file))[0]

    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)
            item["topic"] = topic
            item["subtopic"] = subtopic
            all_data.append(item)

questions = [item["question"] for item in all_data]

# ----------------- Prompt Format -----------------
def format_prompt(q: str) -> str:
    return (
        "Please answer the given question, and provide a step-by-step solution. "
        "Using the format: Step 1: ..., Step 2: ..., ...\n"
        f"The question is:\n{q}"
    )

# ----------------- Async OpenAI Generation -----------------
client = AsyncOpenAI(api_key=args.api_key)
semaphore = asyncio.Semaphore(300)  # tune based on your rate limit

async def call_openai(prompt: str):
    retry = 0
    while retry <= 5:
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model=args.model,
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=0.7,
                    # top_p=0.95,
                    max_completion_tokens=4096,
                )
                raw_output = response.choices[0].message.content.strip()
                return raw_output.split("\n\nThinking\n\n")[0]
        except RateLimitError:
            await asyncio.sleep(2 ** retry)
            retry += 1
        except OpenAIError as e:
            return f"[ERROR]: {str(e)}"
    return "[FAILED]: Too many retries"

async def generate_all():
    print(f"🚀 Generating {len(questions)} questions using OpenAI model '{args.model}'...")
    prompts = [format_prompt(q) for q in questions]
    
    results = [None] * len(prompts)  # Placeholder list

    pbar = tqdm(total=len(prompts), desc="🧠 Generating")

    async def indexed_task(i, prompt):
        result = await call_openai(prompt)
        results[i] = result
        pbar.update(1)

    await asyncio.gather(*(indexed_task(i, p) for i, p in enumerate(prompts)))
    pbar.close()

    return results

# ----------------- Run & Save -----------------
async def main():
    generations = await generate_all()

    output_file = f'results/{args.model}.jsonl'
    os.makedirs("results", exist_ok=True)

    with open(output_file, 'w') as fw:
        for item, generation in tqdm(zip(all_data, generations), total=len(all_data), desc="✍️ Writing all results"):
            item["generation"] = generation
            item["model"] = args.model
            json.dump(item, fw, ensure_ascii=False)
            fw.write('\n')

    print(f"✅ All done. Results saved to: {output_file}")

if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("✋ Interrupted by user.")




