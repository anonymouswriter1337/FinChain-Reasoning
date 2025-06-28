from huggingface_hub import snapshot_download
import os

model_id_list = [
                # "mistralai/Mixtral-8x22B-Instruct-v0.1", 
                "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
                "AdaptLLM/finance-LLM-13B"]

for model_id in model_id_list:
    snapshot_download(repo_id=model_id, 
                        cache_dir=os.getenv("HF_CACHE")+"/models")
