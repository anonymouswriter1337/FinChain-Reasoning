import os
import time
from openai import OpenAI

model_id = "o3-mini"

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY is not set")

openai = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generation(question):

    messages = [
        {"role": "system", "content": "You are a finance chatbot"},
        {"role": "user", "content":f"Please answer the given question, and provide a step-by-step solution. Using the format: Step 1: ..., Step 2: ..., ...\n The question is:\n{question}"
    },
    ]

    model_reply = None
    for retry in range(5):
        try:
            response = openai.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=1024,
                temperature=0.7,
            ) if model_id.startswith("gpt") else openai.chat.completions.create(
                model=model_id,
                messages=messages,
                max_completion_tokens=8192,
            )
            model_reply = response.choices[0].message.content
            break
        except Exception as e:
            print(f"[Retry: {retry}] Error in HighSchoolTeacherOpenAI: {str(e)}")
            if retry == 4:
                raise e
            else:
                # Sleep for 30 sec
                time.sleep(30)
    
    return model_reply
