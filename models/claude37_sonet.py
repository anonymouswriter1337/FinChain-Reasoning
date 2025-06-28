import os
import time
import anthropic

model_id = "claude-3-7-sonnet-20250219"

if not os.getenv("ANTHROPIC_API_KEY"):
    raise ValueError("ANTHROPIC_API_KEY API key not set.")

claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def generation(question):

    messages = [
        {"role": "user", "content":f"Please answer the given question, and provide a step-by-step solution. Using the format: Step 1: ..., Step 2: ..., ...\n The question is:\n{question}"},
    ]

    model_reply = None
    for retry in range(5):
        try:
            response = claude.messages.create(
                model=model_id,
                messages=messages,
                max_tokens=8192,
            )
            model_reply = response.content[0].text
            break
        except Exception as e:
            print(f"[Retry: {retry}] Error in Claude-3.7-Sonet Chat completion: {str(e)}")
            if retry == 4:
                raise e
            else:
                # Sleep for 30 sec
                time.sleep(30)
    
    return model_reply
