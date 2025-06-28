# import argparse, json, tqdm, importlib, os

# # Setup argument parser
# parser = argparse.ArgumentParser(description='Process command line arguments.')
# # deepseek_r1_distill_llama_8b  fino1_8b  llama_3.1_8b_instruct  wiroai_finance_gemma_8b  wiroai_finance_qwen_7b  wiroai_finance_llama_8b
# parser.add_argument('--model', type=str, help='The model to be used for generation.', default='qwen_3_8b')
# parser.add_argument('--topic', type=str, help='The topic to be used for the input/output file names.', default='ci')
# args = parser.parse_args()

# # Dynamically import the model
# model = importlib.import_module(f"models.{args.model}")

# # Open the input file and output file based on the model and topic arguments
# input_file = f'data/testset/{args.topic}.jsonl'
# output_file = f'results/{args.topic}/{args.model}.jsonl'

# # Ensure the output directory exists
# os.makedirs(os.path.dirname(output_file), exist_ok=True)

# with open(input_file, 'r') as f:
#     with open(output_file, 'a') as fw:
#         for line in tqdm.tqdm(f.readlines()):
#             data = json.loads(line)
#             question = data['question']
#             data['generation'] = model.generation(question)
#             data['model'] = args.model
#             json.dump(data, fw, ensure_ascii=False)
#             fw.write('\n')
#             fw.flush()

# import argparse
# import json
# import importlib
# import os
# from tqdm import tqdm

# # ----------------- Args -----------------
# parser = argparse.ArgumentParser(description='vLLM-based full batch generation script.')
# parser.add_argument('--model', type=str, default='qwen_3_8b', help='The model module name under models/')
# parser.add_argument('--topic', type=str, default='investment_analysis', help='The topic name')
# parser.add_argument('--subtopic', type=str, default='ci', help='Subtopic name')
# args = parser.parse_args()

# # ----------------- Load Model -----------------
# model = importlib.import_module(f"models.{args.model}")  # Must expose batch_generate(questions: List[str]) -> List[str]

# # ----------------- Paths -----------------
# input_file = f'data/testset/{args.topic}/{args.subtopic}.jsonl'
# output_file = f'results/{args.topic}/{args.subtopic}/{args.model}.jsonl'
# os.makedirs(os.path.dirname(output_file), exist_ok=True)

# # ----------------- Read Questions -----------------
# with open(input_file, 'r') as f:
#     all_data = [json.loads(line) for line in f]

# questions = [item['question'] for item in all_data]

# # ----------------- Generate Answers via vLLM -----------------
# print(f"🚀 Generating {len(questions)} questions using {args.model}...")
# generations = model.batch_generate(questions)

# # ----------------- Write Results -----------------
# with open(output_file, 'w') as fw:
#     for item, generation in tqdm(zip(all_data, generations), total=len(all_data)):
#         item['generation'] = generation
#         item['model'] = args.model
#         json.dump(item, fw, ensure_ascii=False)
#         fw.write('\n')

# import argparse
# import json
# import importlib
# import os
# from tqdm import tqdm
# from glob import glob

# # ----------------- Args -----------------
# parser = argparse.ArgumentParser(description='vLLM-based full batch generation across all topics.')
# parser.add_argument('--model', type=str, default='qwen_3_8b', help='The model module name under models/')
# args = parser.parse_args()

# # ----------------- Load Model Once -----------------
# model = importlib.import_module(f"models.{args.model}")  # Must expose batch_generate()

# # ----------------- Find All JSONL Files in testset/*/*.jsonl -----------------
# input_files = sorted(glob('data/testset/*/*.jsonl'))  # topic/subtopic.jsonl

# print(f"📂 Found {len(input_files)} files to process.")

# # ----------------- Process All -----------------
# for input_file in input_files:
#     # Parse topic/subtopic names
#     parts = input_file.split(os.sep)
#     topic = parts[-2]
#     subtopic = os.path.splitext(parts[-1])[0]

#     output_file = f'results/{topic}/{subtopic}/{args.model}.jsonl'
#     os.makedirs(os.path.dirname(output_file), exist_ok=True)

#     # Skip if output already exists
#     if os.path.exists(output_file):
#         print(f"✅ Skipping '{topic}/{subtopic}' (already exists)")
#         continue

#     # Read input
#     with open(input_file, 'r') as f:
#         all_data = [json.loads(line) for line in f]

#     if not all_data:
#         print(f"⚠️ Empty file: {input_file}")
#         continue

#     questions = [item['question'] for item in all_data]

#     # Generate
#     print(f"🚀 Generating {len(questions)} questions for '{topic}/{subtopic}'...")
#     try:
#         generations = model.batch_generate(questions)
#     except Exception as e:
#         print(f"❌ Failed on {topic}/{subtopic}: {e}")
#         continue

#     # Save output
#     with open(output_file, 'w') as fw:
#         for item, generation in tqdm(zip(all_data, generations), total=len(all_data), desc=f"✍️ Writing {topic}/{subtopic}"):
#             item['generation'] = generation
#             item['model'] = args.model
#             json.dump(item, fw, ensure_ascii=False)
#             fw.write('\n')

#     print(f"📁 Done: {output_file}")

import argparse
import json
import importlib
import os
from tqdm import tqdm
from glob import glob

# ----------------- Args -----------------
parser = argparse.ArgumentParser(description='vLLM: Generate all topics & subtopics into one output.')
parser.add_argument('--model', type=str, default='gemma_2_9b_instruct', help='The model module name under models/')
args = parser.parse_args()

# ----------------- Load Model -----------------
model = importlib.import_module(f"models.{args.model}")  # Must expose batch_generate()

# ----------------- Collect All Questions -----------------
all_data = []
input_files = sorted(glob('data/testset/*/*.jsonl'))  # topic/subtopic.jsonl

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

# ----------------- Generate -----------------
print(f"🚀 Generating {len(questions)} questions using model '{args.model}'...")
generations = model.batch_generate(questions)

# ----------------- Save to Single Output File -----------------
output_file = f'results/{args.model}.jsonl'
os.makedirs("results", exist_ok=True)

with open(output_file, 'w') as fw:
    for item, generation in tqdm(zip(all_data, generations), total=len(all_data), desc="✍️ Writing all results"):
        item["generation"] = generation
        item["model"] = args.model
        json.dump(item, fw, ensure_ascii=False)
        fw.write('\n')

print(f"✅ All done. Results saved to: {output_file}")

