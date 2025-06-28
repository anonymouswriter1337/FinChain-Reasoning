import random
import re
import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.schema import HumanMessage
import os
import pandas as pd
from tqdm import tqdm
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import evaluate
import re
import warnings
warnings.filterwarnings("ignore")

reasoning_models = [
    "deepseek_r1_distill_llama_8b.jsonl",
    "deepseek_r1_distill_llama_70b.jsonl",
    "deepseek_r1_distill_qwen_7b.jsonl",
    "deepseek_r1_distill_qwen_32b.jsonl",
    "fino1_8b.jsonl",
    "fin_r1.jsonl",
    "qwen_3_8b.jsonl",
    "wizardmath_7b.jsonl",
    "metamath_13b.jsonl"
]

# non_reasoning_models = [
#     "llama3p1_8b_instruct.jsonl",
#     "gemma_2_27b_instruct.jsonl",
#     "financeconnect_13b.jsonl",
#     "qwen_25_math_7b.jsonl",
#     "mathstral_7b.jsonl",
#     "llama3p1_70b_instruct.jsonl",
#     "gemma_2_9b_instruct.jsonl",
#     "qwen_2p5_7b.jsonl",
#     "finance_llm.jsonl",
#     "qwen3_30b_a3b.jsonl",
#     "mistral_7b_instruct_v0p3.jsonl",
#     "llama3p3_70b_instruct.jsonl",
#     "finance_llm_13b.jsonl",
#     "wiroai_finance_gemma_9b.jsonl",
#     "gemma_3_27b_instruct.jsonl",
#     "mixtral_8x7b_instruct_v0p1.jsonl",
#     "wiroai_finance_qwen_7b.jsonl",
#     "wiroai_finance_llama_8b.jsonl",
# ]

# models_left = [
#     "llama3p3_70b_instruct.jsonl",
#     "qwen_2p5_7b.jsonl",
#     "llama3p1_70b_instruct.jsonl",
#     "qwen_25_math_7b.jsonl",
#     "qwen3_30b_a3b.jsonl"
# ]

gpt_models = [
    "gpt-4.1-mini.jsonl",
    # "gpt-4o-mini.jsonl",
    "o4-mini.jsonl",
    "o3-mini.jsonl",
    "gpt-4.1.jsonl"
]

# to_exclude = ['qwen_25_math_7b.jsonl']

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
bertscorer = BERTScorer(model_type = 'allenai/longformer-base-4096', device='mps')
rougescorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
bleuscorer = evaluate.load("bleu")

def convert_to_float(text):
    # Remove non-numeric characters and convert to float
    # Extract numbers and handle special characters
    text = text.replace('~', '').strip()  # Remove approximate symbol
    text = text.replace('$', '').replace(',', '')  # Remove dollar sign and commas
    numbers = re.findall(r'[\d.]+', text)
    if not numbers:
        return None
    
    try:
        number = float(numbers[0])
    except:
        return None

    # Handle multipliers

    if 'billion' in text.lower():
        number *= 1000000000
    elif 'million' in text.lower():
        number *= 1000000
    elif 'thousand' in text.lower():
        number *= 1000

    return number

def extract_final_answer(step):
    final_answer_match = re.search(r'=\s*([^=]+)$', step)
    final_answer = final_answer_match.group(1).strip() if final_answer_match else ''
    final_answer = final_answer.split('\n')[0]
    final_answer = convert_to_float(final_answer)
    return final_answer


def extract_steps(text):
    if '\nuser\n' in text:
        text = text.split('\nuser\n')[0]

    # Find all instances of "Step" at the beginning of a line or after newline
    steps = re.finditer(r'(?:^|\n)(\s*)(\**)(#*)(\s*)Step(-*)(\s*)(\d+)', text)
    step_starts = [step.span()[0] for step in steps]
    if len(step_starts) == 0:
        steps = re.finditer(r'(?:^|\n)\s*(\**)(\d+)(.*)', text)
        step_starts = [step.span()[0] for step in steps]
    step_spans = [(step_starts[idx], step_starts[idx+1]) for idx in range(len(step_starts) - 1)]
    # Add the last step span
    if len(step_starts) > 0:
        step_spans.append((step_starts[-1], len(text)))
    step_strings = [text[start:end].strip() for start, end in step_spans]
    # Remove leading "Step X" from each step
    step_strings = [re.sub(r'(?i)Step\s+\d+\s*:', '', step).strip() for step in step_strings]
    
    # Find stepwise final answers
    step_final_answers = []
    for step in step_strings:
        # Find the last '=' and extract everything after it until the end
        step_final_answer = extract_final_answer(step)
        step_final_answers.append(step_final_answer)

    # Find final answer
    final_answer = None
    for step_final_answer in reversed(step_final_answers):
        if step_final_answer is not None:
            final_answer = step_final_answer
            break
    
    # # Find final answer
    # # Find the last step
    # if len(step_strings) > 0:

    #     final_answer_step = None
    #     for potential_step in reversed(step_strings):
    #         # Replace 
    #         if '=' in potential_step:
    #             final_answer_step = potential_step
    #             break

    #     if final_answer_step is not None:
    #         # last_step = step_strings[-1]
    #         # Find the last '=' and extract everything after it until the end
    #         final_answer = extract_final_answer(final_answer_step)
    #     else:
    #         final_answer = None

    # else:
    #     final_answer = None

    # Remove extra spaces
    step_strings = [re.sub(r'\s+', ' ', step) for step in step_strings]

    return step_strings, step_final_answers, final_answer

def preprocess_document(document):
    return re.sub(r"\s+", " ", document)

def evaluate_trace(gt, pred):

    if gt is None or pred is None:
        return None, None, None, None, None, None, None, None, None, None
    
    rouge_score = rougescorer.score(gt, pred)
    rouge1, rouge2, rougeL, rougeLsum = rouge_score['rouge1'].fmeasure, rouge_score['rouge2'].fmeasure, rouge_score['rougeL'].fmeasure, rouge_score['rougeLsum'].fmeasure

    bert_score = bertscorer.score([gt], [pred])[2][0].item()

    try:
        bleu_score = bleuscorer.compute(predictions=[pred], references=[gt])['bleu']
    except Exception as e:
        print(e)
        bleu_score = 0

    # bert_score  = 0
    # bleu_score = 0
    # rouge1 = 0
    # rouge2 = 0
    # rougeL = 0
    # rougeLsum = 0

    gt_steps, gt_step_final_answers, gt_final_answer = extract_steps(gt)
    pred_steps, pred_step_final_answers, pred_final_answer = extract_steps(pred)

    if len(gt_steps) == 0 or len(pred_steps) == 0:
        print(gt)
        print(pred)
        return 0, 0, 0, 0, rouge1, rouge2, rougeL, rougeLsum, bert_score, bleu_score

    gt_steps_embeddings = model.encode(gt_steps)
    pred_steps_embeddings = model.encode(pred_steps)

    step_final_answer_correctness = [[0 for _ in range(len(pred_steps))] for _ in range(len(gt_steps))]

    for id_i, gt_step_answer in enumerate(gt_step_final_answers):
        if gt_step_answer is None:
            continue
        for id_j, pred_step_answer in enumerate(pred_step_final_answers):
            if pred_step_answer is None:
                continue
            if abs(gt_step_answer - pred_step_answer) / (gt_step_answer + 0.0001) < 0.1:
                step_final_answer_correctness[id_i][id_j] = 1
                break
        else:
            step_final_answer_correctness[id_i][id_j] = 0


    similarity_matrix = cosine_similarity(gt_steps_embeddings, pred_steps_embeddings)
    similarity_matrix = np.multiply(similarity_matrix, step_final_answer_correctness)
    
    step_final_answer_correctness = np.max(np.array(step_final_answer_correctness), axis = 1)
    
    max_similarities_backward = np.max(similarity_matrix, axis=1)
    max_similarities_forward = np.max(similarity_matrix, axis=0)
    binarized_similarity_backward = max_similarities_backward > 0.6
    binarized_similarity_forward = max_similarities_forward > 0.6
    recall = float(np.sum(binarized_similarity_backward) / len(gt_steps))
    precision = float(np.sum(binarized_similarity_forward) / len(pred_steps))

    step_final_answer_correct_acc = step_final_answer_correctness.sum() / len(pred_steps) if len(pred_steps) > 0 else 0

    # Check final answer match
    if gt_final_answer is None or pred_final_answer is None:
        final_answer_match = 0
    else:
        final_answer_match = int(abs(gt_final_answer - pred_final_answer)/(gt_final_answer + 0.0001) < 0.05)

    return recall, precision, final_answer_match, step_final_answer_correct_acc, rouge1, rouge2, rougeL, rougeLsum, bert_score, bleu_score

# Non-reasoning models
for model_file in os.listdir('../results/'):
    if model_file.endswith('.jsonl') and model_file not in reasoning_models:

        # if model_file not in models_left or model_file in to_exclude:
        #     print(f"Skipping {model_file} as calculation already run or to be tuned.")
        #     continue

        if model_file not in gpt_models:
            print(f"Skipping {model_file} as it is not a gpt model.")
            continue

        none_counts = 0

        with open(f'../evals/{model_file}', 'w') as f_eval:
            with open(f'../results/{model_file}', 'r') as f_pred:
                for line in tqdm(f_pred, desc=model_file, total=2700):
                    json_line = json.loads(line)
                    recall, precision, final_answer_match, step_final_answer_correct_acc, rouge1, rouge2, rougeL, rougeLsum, bertscore, bleuscore = evaluate_trace(json_line['solution'], json_line['generation'])
                    
                    if recall is None or precision is None or final_answer_match is None:
                        none_counts += 1
                        print(none_counts)

                    result_json_line = {}
                    result_json_line['seed'] = json_line['seed']
                    result_json_line['id'] = json_line.get('id', None)
                    result_json_line['level'] = json_line['level']
                    result_json_line['topic'] = json_line['topic']
                    result_json_line['subtopic'] = json_line['subtopic']
                    result_json_line['model'] = json_line['model']

                    result_json_line['recall'] = recall
                    result_json_line['precision'] = precision
                    result_json_line['final_answer_match'] = final_answer_match
                    result_json_line['step_final_answer_correct_acc'] = step_final_answer_correct_acc
                    result_json_line['rouge1'] = rouge1
                    result_json_line['rouge2'] = rouge2
                    result_json_line['rougeL'] = rougeL
                    result_json_line['rougeLsum'] = rougeLsum
                    result_json_line['bertscore'] = bertscore
                    result_json_line['bleuscore'] = bleuscore
                    f_eval.write(json.dumps(result_json_line) + '\n')

print("Non-reasoning models evaluation completed.")
# ## Reasoning models
# for model_file in os.listdir('../results/reasoning_parsed/'):
#     if model_file.endswith('.jsonl'):

#         # if model_file in non_reasoning_models or model_file in models_left:
#         #     print(f"Skipping {model_file} as calculation already run or to be tuned.")
#         #     continue

#         if model_file not in gpt_models:
#             print(f"Skipping {model_file} as it is not a gpt model.")
#             continue

#         none_counts = 0

#         with open(f'../evals/reasoning_parsed/{model_file}', 'w') as f_eval:
#             with open(f'../results/reasoning_parsed/{model_file}', 'r') as f_pred:
#                 for line in tqdm(f_pred, desc=model_file, total=2700):
#                     json_line = json.loads(line)
#                     recall, precision, final_answer_match, step_final_answer_correct_acc, rouge1, rouge2, rougeL, rougeLsum, bertscore, bleuscore = evaluate_trace(json_line['solution'], json_line['generation_parsed'])
                    
#                     if recall is None or precision is None or final_answer_match is None:
#                         none_counts += 1
#                         print(none_counts)

#                     result_json_line = {}
#                     result_json_line['seed'] = json_line['seed']
#                     result_json_line['id'] = json_line.get('id', None)
#                     result_json_line['level'] = json_line['level']
#                     result_json_line['topic'] = json_line['topic']
#                     result_json_line['subtopic'] = json_line['subtopic']
#                     result_json_line['model'] = json_line['model']

#                     result_json_line['recall'] = recall
#                     result_json_line['precision'] = precision
#                     result_json_line['final_answer_match'] = final_answer_match
#                     result_json_line['step_final_answer_correct_acc'] = step_final_answer_correct_acc
#                     result_json_line['rouge1'] = rouge1
#                     result_json_line['rouge2'] = rouge2
#                     result_json_line['rougeL'] = rougeL
#                     result_json_line['rougeLsum'] = rougeLsum
#                     result_json_line['bertscore'] = bertscore
#                     result_json_line['bleuscore'] = bleuscore
#                     f_eval.write(json.dumps(result_json_line) + '\n')